"""Factory that builds a Modal app for a disaggregated slime run via stitch.

Unlike the colocated slime launcher (``build_slime_app``, driven by
``modal run ::train``), the disaggregated flow is **deploy-based**: a persistent
Modal Flash pool of SGLang rollout servers plus a clustered trainer that
publishes sparse weight deltas to a Modal Volume bulletin board the pool syncs
from. Build the app with :func:`build_stitch_app`, ``modal deploy`` it, then
spawn a run through the ``launch_train`` local entrypoint::

    app = build_stitch_app(model=..., dataset=..., recipe=StitchRecipe(...))

    uv run modal deploy -m <module_with_app>
    uv run modal run -m <module_with_app>::launch_train
    uv run modal run -m <module_with_app>::smoke_flash_pool

This packages the stitch ``slime_disagg`` cookbook (``modal_train.py``) around a
training-gym ``StitchRecipe`` + ``ModelConfig`` + ``DatasetConfig``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from types import SimpleNamespace

import modal
import modal.experimental

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS, modal_tag_value
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.stitch_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    YAML_CONFIG_FIELDS,
    StitchRecipe,
    fields_to_argv,
)

MINUTES = 60
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
RAY_PORT = 6379
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
# Ephemeral host-local full HF checkpoint the sidecar patches in place per delta.
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
SLIME_ROOT = "/root/slime"

# stitch supplies the bulletin protocol + sidecar + slime hooks. modal-projects
# repos are public, so a plain build-time pip install needs no token. Pin the
# commit so a moving tip can't silently change the cached image layer.
STITCH_REPO_URL = "https://github.com/modal-projects/stitch.git"
STITCH_REPO_REF = "1486e2e80540fcef72a78a6ecb98aa229c6139fc"


class _ShippedSlimeConfig:
    """Runtime carrier for the slime args shipped to a deployed Trainer.

    ``launch_train`` resolves a :class:`StitchRecipe` + model + dataset locally
    into a plain field dict and spawns it as data, so new or edited recipes run
    without a redeploy. The Trainer rebuilds this carrier, injects the per-run
    fields (rollout endpoint, bulletin dir, custom config), then materializes
    YAML configs and builds the ``train.py`` command from :meth:`cli_args`.
    """

    _CONTROL = {"async_mode", "slime_model_script"}

    def __init__(
        self, fields: dict, *, async_mode: bool, slime_model_script: str
    ) -> None:
        for key, val in fields.items():
            setattr(self, key, val)
        self.async_mode = async_mode
        self.slime_model_script = slime_model_script

    def cli_args(self) -> list[str]:
        fields = {k: v for k, v in vars(self).items() if k not in self._CONTROL}
        return fields_to_argv(fields)


def _stitch_base_image(recipe: StitchRecipe) -> modal.Image:
    """The slime-fork trainer image, also used for the rollout pool + sidecar."""
    image = (
        modal.Image.from_registry(recipe.slime_image_tag)
        .entrypoint([])
        # The base image bakes an HF cache; drop it so the mounted cache volume
        # at the same path isn't shadowed.
        .run_commands(f"rm -rf {HF_CACHE_PATH}")
        # Replace the bundled slime with the pinned fork (generic HTTP rollout
        # endpoint + publish-only disk-delta hooks).
        .run_commands(
            f"rm -rf {SLIME_ROOT}"
            f" && git clone --depth 1 {recipe.slime_repo_url} {SLIME_ROOT}"
            f" && cd {SLIME_ROOT}"
            f" && git fetch --depth 1 origin {recipe.slime_repo_ref}"
            f" && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {SLIME_ROOT}"
        )
        # Reinstall megatron-core in compat editable mode so `megatron.training`
        # (which slime's megatron backend imports) is importable.
        .run_commands(
            "cd /root/Megatron-LM"
            " && python3 -m pip install --no-deps -e . --config-settings editable_mode=compat"
        )
        .pip_install(
            "autoinference-utils==0.2.0",  # SGLang server lifecycle for the pool
            "fastapi",  # stitch sidecar
            "httpx",  # stitch sidecar
            "uvicorn",  # stitch sidecar
            # slime is installed --no-deps, but the sidecar's host-side disk-delta
            # apply needs these (zstd decompress + xxh3/blake3 checksums).
            "zstandard",
            "xxhash",
            "blake3",
        )
        .pip_install(f"stitch @ git+{STITCH_REPO_URL}@{STITCH_REPO_REF}")
        .env(
            {
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                # Fallbacks for the bulletin hooks when a value isn't threaded
                # through the slime args namespace.
                "DELTA_BULLETIN_ROOT": recipe.delta_bulletin_root,
            }
        )
    )
    if recipe.image_run_commands:
        image = image.run_commands(*recipe.image_run_commands)
    if recipe.image_env:
        image = image.env(recipe.image_env)
    if recipe.image_overlay is not None:
        image = recipe.image_overlay(image)
    # Mount the package so the trainer, the `python3 -m …stitch.sidecar`
    # subprocess, and the Ray workers can import the vendored spine + hooks.
    image = image.add_local_python_source("modal_training_gym", copy=True)
    return image


def build_stitch_app(
    *,
    model: ModelConfig,
    dataset: DatasetConfig,
    recipe: StitchRecipe,
    name: str | None = None,
) -> modal.App:
    """Return a deployable Modal App for disaggregated slime training.

    Defines a ``Server`` Flash-pool class, a clustered ``Trainer`` class,
    ``download_model`` / ``prepare_dataset`` functions, and ``launch_train`` /
    ``smoke_flash_pool`` local entrypoints.
    """
    StitchRecipe._resolve_data_paths(dataset)  # validate dataset paths resolve

    app_name = name or recipe.name or f"stitch-{modal_tag_value(model.model_name)}"
    delta_volume_name = recipe.delta_volume_name or f"stitch-delta-bulletin-{app_name}"
    delta_bulletin_root = recipe.delta_bulletin_root
    model_name = model.model_path or model.model_name
    rollout_concurrency = recipe.sglang_server_concurrency
    n_train_nodes = recipe.actor_num_nodes

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": Framework.STITCH.value,
        "_modal_job_type": "training",
        **{str(k): str(v) for k, v in recipe.app_tags.items()},
    }
    if recipe.wandb is not None:
        if recipe.wandb.project:
            tags["wandb_project"] = modal_tag_value(recipe.wandb.project)
        if recipe.wandb.group:
            tags["wandb_group"] = modal_tag_value(recipe.wandb.group)

    image = _stitch_base_image(recipe)

    # Structural SGLang args (derived from the recipe) merged under the recipe's
    # per-model tuning. The disk-delta branch applies deltas host-side, so no
    # engine-side delta server args are passed.
    sglang_server_args = {
        "--served-model-name": model_name,
        "--dtype": "bfloat16",
        "--cuda-graph-max-bs": str(rollout_concurrency),
        "--max-running-requests": str(rollout_concurrency),
        "--trust-remote-code": "",
        **dict(recipe.sglang_server_args),
    }
    warmup_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    hf_cache_volume = modal.Volume.from_name(
        "huggingface-cache", create_if_missing=True
    )
    data_volume = modal.Volume.from_name(
        f"stitch-data-{app_name}", create_if_missing=True
    )
    checkpoints_volume = modal.Volume.from_name(
        f"stitch-checkpoints-{app_name}", create_if_missing=True
    )
    delta_volume = modal.Volume.from_name(
        delta_volume_name, create_if_missing=True, version=2
    )
    train_volumes = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        str(CHECKPOINTS_PATH): checkpoints_volume,
        delta_bulletin_root: delta_volume,
    }

    hf_secret = modal.Secret.from_name("huggingface-secret")
    train_secrets = [hf_secret]
    if recipe.wandb is not None:
        train_secrets.append(modal.Secret.from_name("wandb-secret"))

    memory = recipe.memory
    app = modal.App(app_name, tags=tags)

    with image.imports():
        from autoinference_utils.endpoint import (
            SGLangEndpoint,
            warmup_chat_completions,
        )

    @app.cls(
        image=image,
        gpu=f"{recipe.gpu_type}:{recipe.rollout_num_gpus_per_engine}",
        cloud=recipe.cloud,
        region=recipe.region,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            delta_bulletin_root: delta_volume,
        },
        secrets=[hf_secret],
        min_containers=recipe.rollout_min_containers,
        timeout=40 * MINUTES,
        scaledown_window=15 * MINUTES,
        serialized=True,
    )
    @modal.experimental.http_server(
        port=SIDECAR_PORT,
        proxy_regions=recipe.proxy_regions,
        exit_grace_period=25,
        startup_timeout=SERVER_STARTUP_TIMEOUT,
    )
    @modal.concurrent(target_inputs=rollout_concurrency)
    class Server:
        """One SGLang rollout server plus the stitch weight-sync sidecar."""

        @modal.enter()
        def startup(self) -> None:
            from modal_training_gym.frameworks.stitch import sidecar_process

            self.endpoint = SGLangEndpoint(
                model_path=model_name,
                worker_port=SGLANG_PORT,
                tp=recipe.rollout_num_gpus_per_engine,
                extra_server_args=sglang_server_args,
                health_timeout=SERVER_STARTUP_TIMEOUT,
                health_poll_interval=10.0,
            )
            self.endpoint.start()
            warmup_chat_completions(
                port=SGLANG_PORT,
                payload=warmup_payload,
                successful_requests=2,
                request_timeout=120.0,
                max_attempts_per_request=3,
            )
            # Deltas apply host-side onto a copy of the base checkpoint; the base
            # resolves to the same HF cache snapshot the SGLang server loaded.
            from huggingface_hub import snapshot_download

            base_checkpoint_dir = snapshot_download(model_name, local_files_only=True)
            self.sidecar = sidecar_process.start_sglang_sidecar(
                sidecar_port=SIDECAR_PORT,
                sglang_port=SGLANG_PORT,
                bulletin_root=delta_bulletin_root,
                local_checkpoint_dir=LOCAL_CHECKPOINT_PATH,
                base_checkpoint_dir=base_checkpoint_dir,
                volume_name=delta_volume_name,
                commit_mode=recipe.sidecar_commit_mode,
                debug_requests=recipe.sidecar_debug_requests,
            )
            sidecar_process.wait_http(
                f"http://127.0.0.1:{SIDECAR_PORT}/health",
                self.sidecar,
                SERVER_STARTUP_TIMEOUT,
            )
            print(
                f"Rollout server ready: model={model_name}, "
                f"target_inputs={rollout_concurrency}"
            )

        @modal.exit()
        def stop(self) -> None:
            from modal_training_gym.frameworks.stitch import sidecar_process

            sidecar_process.terminate_process(getattr(self, "sidecar", None))
            if hasattr(self, "endpoint"):
                self.endpoint.stop()

    @app.cls(
        image=image,
        gpu=f"{recipe.gpu_type}:{recipe.actor_num_gpus_per_node}",
        memory=memory,
        cloud=recipe.cloud,
        region=recipe.region,
        volumes=train_volumes,
        secrets=train_secrets,
        timeout=24 * 60 * MINUTES,
        startup_timeout=20 * MINUTES,
        scaledown_window=30 * MINUTES,
        experimental_options={"efa_enabled": True},
        serialized=True,
    )
    @modal.experimental.clustered(n_train_nodes, rdma=True)
    class Trainer:
        """slime actor cluster. The Ray cluster comes up once per container in
        enter(), so back-to-back runs reuse it instead of rebuilding it."""

        @modal.enter()
        def start_ray(self) -> None:
            from modal_training_gym.frameworks.stitch import ray_cluster

            rank, master_addr, my_ip = ray_cluster.get_modal_cluster_context(
                n_train_nodes
            )
            self.rank = rank
            os.environ.update(
                {
                    "SLIME_HOST_IP": my_ip,
                    "SGLANG_HOST_IP": my_ip,
                    "HOST_IP": my_ip,
                    "MASTER_ADDR": master_addr,
                    "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
                    "no_proxy": f"127.0.0.1,{master_addr},{my_ip}",
                    "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
                    "DELTA_VOLUME_NAME": delta_volume_name,
                    "DELTA_APP_NAME": app_name,
                    "DELTA_SERVER_CLS_NAME": "Server",
                    "DELTA_BULLETIN_ROOT": delta_bulletin_root,
                    **{str(k): str(v) for k, v in recipe.environment.items()},
                }
            )
            if rank == 0:
                ray_cluster.start_ray_head(my_ip, n_train_nodes, ray_port=RAY_PORT)
            else:
                ray_cluster.start_ray_worker(my_ip, master_addr, ray_port=RAY_PORT)

        @modal.method()
        def train(self, payload: dict) -> None:
            from stitch.providers.modal import resolve_flash_gateway_url

            from modal_training_gym.frameworks.stitch import (
                bulletin_hooks,
                trainer_helpers,
            )

            for volume in train_volumes.values():
                volume.reload()
            # Rank 0 drives; other ranks only need the Ray worker from enter().
            if self.rank != 0:
                return

            cfg = _ShippedSlimeConfig(
                payload["fields"],
                async_mode=payload["async_mode"],
                slime_model_script=payload["slime_model_script"],
            )
            cfg.rollout_endpoint_url = resolve_flash_gateway_url(app_name, "Server")
            # Fresh run id per launch: slime writes this run's chain under
            # <bulletin_root>/<run_id>/weight_v{N}/ and the canonical `latest`
            # pointer is self-identifying, so a new run never collides with a
            # finished one — no manual bulletin reset needed.
            run_id = uuid.uuid4().hex[:12]
            cfg.update_weight_disk_dir = f"{delta_bulletin_root}/{run_id}"
            # stitch's publish hooks read these off the slime args namespace;
            # merge over any user extra_config already on custom_config_path.
            custom_config = dict(getattr(cfg, "custom_config_path", None) or {})
            custom_config.update(
                {
                    "update_weight_delta_volume_name": delta_volume_name,
                    "rollout_modal_flash_app_name": app_name,
                    "rollout_modal_flash_server_cls_name": "Server",
                    "run_id": run_id,
                }
            )
            cfg.custom_config_path = custom_config

            trainer_helpers.prepare_config(cfg, tempfile.mkdtemp(), YAML_CONFIG_FIELDS)
            cmd = trainer_helpers.build_train_cmd(
                cfg, SLIME_ROOT, model_script_attr="slime_model_script"
            )

            # Claim the pool for this run before slime publishes: write the empty
            # pointer and wake the pool so every replica resets to base now.
            bulletin_hooks.claim_pool(
                SimpleNamespace(
                    update_weight_disk_dir=cfg.update_weight_disk_dir,
                    **custom_config,
                )
            )

            print(
                f"Training on {app_name}: nodes={n_train_nodes}, "
                f"rollout_endpoint={cfg.rollout_endpoint_url}"
            )
            print(f"Command: {cmd}")
            subprocess.run(["bash", "-lc", cmd], check=True)

    @app.function(
        image=image,
        volumes={str(HF_CACHE_PATH): hf_cache_volume},
        timeout=2 * 60 * MINUTES,
        secrets=[hf_secret],
        serialized=True,
    )
    def download_model() -> None:
        model.download()
        hf_cache_volume.commit()

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * MINUTES,
        secrets=[hf_secret],
        serialized=True,
    )
    def prepare_dataset() -> None:
        data_volume.reload()
        prompt_data, eval_paths = StitchRecipe._resolve_data_paths(dataset)
        dataset.prepare(prompt_data, eval_paths)
        data_volume.commit()

    return app


def spawn_training_run(
    *,
    app_name: str,
    recipe: StitchRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
) -> str:
    """Spawn a training run on a deployed stitch app (call from a
    ``@app.local_entrypoint``). Training args ship as data, so recipe edits run
    without a redeploy; infra changes (GPU, nodes, pool size, Volume names)
    still require ``modal deploy``. Returns the spawned call's object id."""
    from modal.exception import NotFoundError

    payload = recipe.to_payload(model=model, dataset=dataset)
    try:
        trainer = modal.Cls.from_name(app_name, "Trainer")()
        call = trainer.train.spawn(payload)
    except NotFoundError:
        raise SystemExit(
            f"App {app_name!r} is not deployed. Run:\n"
            f"  uv run modal deploy -m <module_with_app>"
        )
    print(f"Spawned train on {app_name}: {call.object_id}")
    return call.object_id


def smoke_flash_pool(
    *,
    app_name: str,
    model: ModelConfig,
    recipe: StitchRecipe,
    weight_version: int = 0,
    timeout_seconds: int = 30 * MINUTES,
) -> None:
    """Check the deployed Flash pool serves completions at a weight version, via
    the gateway and each container directly (call from a
    ``@app.local_entrypoint``)."""
    from modal_training_gym.frameworks.stitch import trainer_helpers

    trainer_helpers.smoke_flash_pool(
        app_name=app_name,
        cls_name="Server",
        model_name=model.model_path or model.model_name,
        weight_version=weight_version,
        expect_min_containers=recipe.rollout_min_containers,
        timeout_seconds=timeout_seconds,
    )
