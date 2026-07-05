"""Factory that builds a Modal app for disaggregated SLIME training via Stitch.

The app has two halves:
- Server: a Modal Flash pool of SGLang servers with a stitch weight-sync sidecar
- Trainer: a clustered SLIME/Ray job that publishes sparse weight deltas through
  a Modal Volume bulletin board
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import cloudpickle
from modal import App, Secret, Volume
import modal
import modal.experimental

from modal_training_gym.common import (
    COMMON_TRAINING_GYM_TAGS,
    hf_secrets,
    modal_tag_value,
)
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.framework import (
    mount_tools_dir,
    resolve_caller_module,
)
from modal_training_gym.common.modal_refs import register_modal_cloudpickle_reducers
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.train_recipes.slime_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    SlimeRecipe,
)
from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe
from modal_training_gym.frameworks.slime.launcher import (
    SLIME_ROOT,
    _build_slime_base_image,
    _PATCH_BRIDGE_PER_TOKEN_LOSS_B64,
)

MINUTES = 60
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
SERVER_STARTUP_TIMEOUT = 35 * MINUTES

# Stitch is installed from git into the Modal image.
STITCH_REPO_URL = "https://github.com/modal-projects/stitch.git"
STITCH_REPO_REF = "main"


def build_stitch_app(
    *,
    training_run_id: str,
    stitch: StitchRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
    group_id: str | None = None,
) -> App:
    """Return a Modal App with Flash rollout pool + clustered Trainer."""
    app_name = name or f"stitch-{training_run_id}"

    SlimeRecipe._validate_custom_model_architecture(model)
    SlimeRecipe._validate_dataset(dataset)

    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)
    register_modal_cloudpickle_reducers()

    # ── Images ────────────────────────────────────────────────────────────────
    base_image = _build_slime_base_image()

    if getattr(stitch, "megatron_to_hf_mode", None) == "bridge":
        base_image = base_image.run_commands(
            f"echo {_PATCH_BRIDGE_PER_TOKEN_LOSS_B64} | base64 -d | python3",
        )

    # Install stitch + sidecar deps into the image
    image = base_image.pip_install(
        f"stitch @ git+{STITCH_REPO_URL}@{STITCH_REPO_REF}",
        "fastapi",
        "httpx",
        "uvicorn",
        "xxhash",
    ).add_local_python_source("modal_training_gym", copy=True)
    image = mount_tools_dir(image)

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(
        f"stitch-{training_run_id}-data",
        create_if_missing=True,
    )
    checkpoints_volume = Volume.from_name(
        f"stitch-{training_run_id}-checkpoints",
        create_if_missing=True,
    )
    delta_volume = Volume.from_name(
        stitch.delta_volume_name, create_if_missing=True, version=2
    )

    train_volumes: dict[str, Volume] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        str(CHECKPOINTS_PATH): checkpoints_volume,
        stitch.delta_bulletin_root: delta_volume,
    }

    model_name = model.model_name
    rollout_concurrency = stitch.sglang_server_concurrency

    # ── App ──────────────────────────────────────────────────────────────────
    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": "stitch",
        "_modal_model_name": modal_tag_value(model.model_name),
        **stitch.app_tags,
    }
    if stitch.wandb is not None:
        tags["_modal_wandb_project"] = modal_tag_value(stitch.wandb.project)
        if stitch.wandb.group:
            tags["_modal_wandb_group"] = modal_tag_value(stitch.wandb.group)
    app = App(app_name, tags=tags)

    # ── SGLang server args ────────────────────────────────────────────────────
    sglang_extra_args: dict[str, str] = {
        "--served-model-name": model_name,
        "--dtype": "bfloat16",
        "--cuda-graph-max-bs": str(rollout_concurrency),
        "--max-running-requests": str(rollout_concurrency),
        "--trust-remote-code": "",
        **(stitch.sglang_server_args or {}),
    }

    # ── Flash Rollout Pool ────────────────────────────────────────────────────

    @app.cls(
        image=image,
        gpu=f"{stitch.rollout_gpu_type}:{stitch.rollout_num_gpus_per_engine}",
        cloud=stitch.cloud,
        region=stitch.region,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            stitch.delta_bulletin_root: delta_volume,
        },
        min_containers=stitch.rollout_min_containers,
        timeout=40 * MINUTES,
        scaledown_window=15 * MINUTES,
        include_source=False,
        secrets=hf_secrets(),
    )
    @modal.experimental.http_server(
        port=SIDECAR_PORT,
        proxy_regions=stitch.rollout_proxy_regions,
        exit_grace_period=25,
        startup_timeout=SERVER_STARTUP_TIMEOUT,
    )
    @modal.concurrent(target_inputs=rollout_concurrency)
    class Server:
        """SGLang rollout server with stitch weight-sync sidecar."""

        @modal.enter()
        def startup(self) -> None:
            import time
            import urllib.error
            import urllib.request

            from huggingface_hub import snapshot_download

            # Ensure model weights are present locally
            snapshot_download(model_name)

            # Build SGLang launch command
            sglang_cmd = [
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                model_name,
                "--port",
                str(SGLANG_PORT),
                "--tp",
                str(stitch.rollout_num_gpus_per_engine),
            ]
            for flag, val in sglang_extra_args.items():
                sglang_cmd.append(flag)
                if val:
                    sglang_cmd.append(val)

            self._sglang_proc = subprocess.Popen(
                sglang_cmd,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        str(i) for i in range(stitch.rollout_num_gpus_per_engine)
                    ),
                },
            )

            # Wait for SGLang to be ready
            deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
            while time.monotonic() < deadline:
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{SGLANG_PORT}/health", method="GET"
                    )
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            break
                except (urllib.error.URLError, OSError, TimeoutError):
                    pass
                if self._sglang_proc.poll() is not None:
                    raise RuntimeError(
                        f"SGLang exited with code {self._sglang_proc.returncode}"
                    )
                time.sleep(5.0)
            else:
                raise TimeoutError("SGLang did not start within timeout")

            # Start the stitch sidecar for weight sync
            base_checkpoint_dir = snapshot_download(model_name, local_files_only=True)
            sidecar_env = {
                **os.environ,
                "SIDECAR_PORT": str(SIDECAR_PORT),
                "SGLANG_PORT": str(SGLANG_PORT),
                "BULLETIN_ROOT": stitch.delta_bulletin_root,
                "LOCAL_CHECKPOINT_DIR": "/local-checkpoint",
                "BASE_CHECKPOINT_DIR": base_checkpoint_dir,
                "DELTA_VOLUME_NAME": stitch.delta_volume_name,
                "SIDECAR_COMMIT_MODE": stitch.sidecar_commit_mode,
                "SIDECAR_DEBUG_REQUESTS": "1" if stitch.sidecar_debug_requests else "",
            }
            self._sidecar_proc = subprocess.Popen(
                ["python3", "-m", "cookbook.slime_disagg.sidecar"],
                env=sidecar_env,
            )

            # Wait for sidecar health
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{SIDECAR_PORT}/health", method="GET"
                    )
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            print(f"Server ready: model={model_name}")
                            return
                except (urllib.error.URLError, OSError, TimeoutError):
                    pass
                if self._sidecar_proc.poll() is not None:
                    raise RuntimeError(
                        f"Sidecar exited with code {self._sidecar_proc.returncode}"
                    )
                time.sleep(2.0)
            raise TimeoutError("Sidecar did not start within timeout")

        @modal.exit()
        def stop(self) -> None:
            for proc in (
                getattr(self, "_sidecar_proc", None),
                getattr(self, "_sglang_proc", None),
            ):
                if proc is None:
                    continue
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    # ── Trainer Cluster ───────────────────────────────────────────────────────
    n_train_nodes = stitch.actor_num_nodes
    gpu_spec = f"{stitch.gpu_type}:{stitch.actor_num_gpus_per_node}"

    train_secrets: list[Secret] = []
    if stitch.wandb is not None:
        train_secrets.append(Secret.from_name(stitch.wandb.modal_wandb_secret_name))

    @app.cls(
        image=image,
        gpu=gpu_spec,
        memory=stitch.memory,
        cloud=stitch.cloud,
        region=stitch.region,
        volumes=train_volumes,
        secrets=train_secrets or None,
        timeout=24 * 60 * MINUTES,
        experimental_options={"efa_enabled": True},
        include_source=False,
    )
    @modal.experimental.clustered(n_train_nodes, rdma=True)
    class Trainer:
        """SLIME trainer cluster with stitch delta-publish hooks."""

        @modal.enter()
        def start_ray(self) -> None:
            from modal_training_gym.common.ray_cluster import ModalRayCluster
            from modal_training_gym.frameworks.slime.modal_helpers.utils import (
                get_modal_cluster_context,
            )

            rank, master_addr, my_ip, _ = get_modal_cluster_context(n_train_nodes)
            self.rank = rank
            os.environ.update(
                {
                    "SLIME_HOST_IP": my_ip,
                    "SGLANG_HOST_IP": my_ip,
                    "HOST_IP": my_ip,
                    **stitch.environment,
                }
            )
            cluster = ModalRayCluster()
            cluster.discover_cluster(n_train_nodes)
            cluster.start_ray()
            self.cluster = cluster

        @modal.method()
        def train(self) -> None:
            """Run training. Rank 0 drives; others provide Ray workers."""
            import asyncio

            for vol in train_volumes.values():
                vol.reload()

            if not self.cluster.is_head:
                asyncio.run(self.cluster.wait_forever())
                return

            from modal_training_gym.frameworks.slime.modal_helpers.utils import (
                build_train_cmd,
                prepare_slime_config,
            )
            from stitch.providers.modal import resolve_flash_gateway_url

            # Resolve the Flash gateway URL for rollout traffic
            rollout_endpoint_url = resolve_flash_gateway_url(app_name, "Server")

            # Fresh run_id per launch for bulletin board partitioning
            run_id = uuid.uuid4().hex[:12]

            # Configure stitch-specific overrides on the recipe
            object.__setattr__(stitch, "rollout_endpoint_url", rollout_endpoint_url)
            object.__setattr__(
                stitch,
                "update_weight_disk_dir",
                f"{stitch.delta_bulletin_root}/{run_id}",
            )
            object.__setattr__(
                stitch,
                "custom_delta_pre_push_path",
                "stitch.trainers.slime.publish_delta_version",
            )
            object.__setattr__(
                stitch,
                "custom_rollout_request_hook_path",
                "stitch.trainers.slime.rollout_request_weight_version_hook",
            )

            extra_cfg = dict(stitch.extra_config or {})
            extra_cfg["update_weight_delta_volume_name"] = stitch.delta_volume_name
            extra_cfg["rollout_modal_flash_app_name"] = app_name
            extra_cfg["rollout_modal_flash_server_cls_name"] = "Server"
            extra_cfg["run_id"] = run_id
            object.__setattr__(stitch, "extra_config", extra_cfg)

            tmpdir = tempfile.mkdtemp()
            prepare_slime_config(stitch, model, tmpdir)
            cmd = build_train_cmd(stitch, SLIME_ROOT, model=model, dataset=dataset)

            # Claim the rollout pool before training starts
            from stitch.bulletin import FilesystemBulletinBoard
            from stitch.providers.modal import commit_volume

            board = FilesystemBulletinBoard(
                Path(stitch.delta_bulletin_root), layout="slime"
            )
            board.claim(run_id)
            commit_volume(stitch.delta_volume_name)

            print(
                f"Training: nodes={n_train_nodes}, "
                f"rollout_endpoint={rollout_endpoint_url}, "
                f"run_id={run_id}"
            )
            print(f"Command: {cmd}")
            env = {**os.environ, **stitch.environment}
            subprocess.run(["bash", "-lc", cmd], check=True, env=env)

    # ── Setup entrypoints ─────────────────────────────────────────────────────

    @app.function(
        image=image,
        volumes={str(HF_CACHE_PATH): hf_cache_volume},
        timeout=2 * 60 * MINUTES,
        secrets=hf_secrets(),
        include_source=False,
        name="download",
    )
    def download() -> None:
        from huggingface_hub import snapshot_download

        hf_cache_volume.reload()
        snapshot_download(repo_id=model_name)
        hf_cache_volume.commit()

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * MINUTES,
        secrets=hf_secrets(),
        include_source=False,
        name="prepare_dataset",
    )
    def prepare_dataset() -> None:
        data_volume.reload()
        prompt_data, eval_paths = SlimeRecipe._resolve_data_paths(dataset)
        dataset.prepare(prompt_data, eval_paths)
        data_volume.commit()

    @app.local_entrypoint()
    def launch() -> None:
        """Deploy the app and run training."""
        download.remote()
        prepare_dataset.remote()
        trainer = Trainer()
        trainer.train.remote()

    return app
