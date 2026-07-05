"""Factory that builds a Modal app for disaggregated stitch training.

Creates a Modal App with:
- A ``Server`` Flash class: SGLang rollout servers with stitch weight-sync sidecars
- ``download`` / ``prepare_dataset`` functions (same interface as the slime launcher)
- A ``train`` function: brings up a Ray cluster and runs SLIME with stitch hooks

Reference: https://github.com/modal-projects/stitch/tree/main/cookbook/slime_disagg
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import PurePosixPath
from typing import Any

import cloudpickle
from modal import App, Image, Secret, Volume

from modal_training_gym.common import (
    COMMON_TRAINING_GYM_TAGS,
    hf_secrets,
    modal_tag_value,
)
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.framework import mount_tools_dir, resolve_caller_module
from modal_training_gym.common.modal_refs import register_modal_cloudpickle_reducers
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.ray_cluster import ModalRayCluster, clustered_if
from modal_training_gym.common.status import StitchStatus
from modal_training_gym.train_recipes.slime_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    SlimeRecipe,
    YAML_CONFIG_FIELDS,
    JSON_CONFIG_FIELDS,
)
from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe

SLIME_ROOT = "/root/slime"

MINUTES = 60
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

# Pin the base slime image by digest (same as the slime launcher).
# Tag: nightly-dev-20260701a
SLIME_IMAGE = (
    "slimerl/slime@sha256:"
    "512b6bed52d3ffd7b8d76c7238ed2bf43446cfadd8aa03a1ea4a39646c92ebf3"
)


def _build_stitch_image(recipe: StitchRecipe) -> Image:
    """Build the base image with SLIME + stitch + sidecar dependencies."""
    image = (
        Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands(f"rm -rf {HF_CACHE_PATH}")
    )

    # Replace the bundled slime with the fork branch that has generic HTTP
    # rollout endpoint and disk-delta hooks.
    if recipe.slime_fork_url and recipe.slime_fork_ref:
        image = image.run_commands(
            f"rm -rf {SLIME_ROOT}"
            f" && git clone --depth 1 {recipe.slime_fork_url} {SLIME_ROOT}"
            f" && cd {SLIME_ROOT}"
            f" && git fetch --depth 1 origin {recipe.slime_fork_ref}"
            f" && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {SLIME_ROOT}"
        )
        # Fix megatron editable install so megatron.training is importable.
        image = image.run_commands(
            "cd /root/Megatron-LM"
            " && python3 -m pip install --no-deps -e ."
            " --config-settings editable_mode=compat"
        )

    # Install stitch and sidecar dependencies.
    image = image.pip_install(
        f"stitch @ git+{recipe.stitch_repo_url}@{recipe.stitch_repo_ref}",
        "autoinference-utils==0.2.0",
        "fastapi",
        "httpx",
        "uvicorn",
        "zstandard",
        "xxhash",
        "blake3",
    )

    image = image.env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )

    image = image.add_local_python_source("modal_training_gym", copy=True)
    image = mount_tools_dir(image)

    if recipe.local_slime:
        image = image.add_local_dir(
            recipe.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )

    return image


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
    """Return a Modal App with disaggregated SLIME training via stitch."""
    app_name = name or f"stitch-{type(stitch).__name__.lstrip('_').lower()}"
    volume_prefix = f"stitch-{type(stitch).__name__.lstrip('_').lower()}"

    SlimeRecipe._validate_custom_model_architecture(model)
    SlimeRecipe._validate_dataset(dataset)

    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)
    register_modal_cloudpickle_reducers()

    n_train_nodes = stitch.actor_num_nodes
    if stitch.use_critic or stitch.advantage_estimator == "ppo":
        n_train_nodes += stitch.critic_num_nodes or stitch.actor_num_nodes
    model_name = model.model_path or model.model_name
    rollout_concurrency = getattr(stitch, "sglang_server_concurrency", 64)
    rollout_gpu_type = stitch.effective_rollout_gpu_type()

    # ── Images ────────────────────────────────────────────────────────────────
    image = _build_stitch_image(stitch)

    # ── Volumes ───────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{volume_prefix}-data", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        f"{volume_prefix}-checkpoints", create_if_missing=True
    )

    delta_volume_name = stitch.delta_volume_name or f"stitch-delta-{app_name}"
    delta_volume = Volume.from_name(
        delta_volume_name, create_if_missing=True, version=2
    )

    train_volumes: dict[str | PurePosixPath, Any] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        str(CHECKPOINTS_PATH): checkpoints_volume,
        stitch.delta_bulletin_root: delta_volume,
    }

    # ── App ───────────────────────────────────────────────────────────────────
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
    gpu_spec = f"{stitch.gpu_type}:{stitch.actor_num_gpus_per_node}"

    # ── SGLang server args ────────────────────────────────────────────────────
    sglang_server_args: dict[str, str] = {
        "--served-model-name": model_name,
        "--dtype": "bfloat16",
        "--cuda-graph-max-bs": str(rollout_concurrency),
        "--max-running-requests": str(rollout_concurrency),
        "--trust-remote-code": "",
        **stitch.sglang_server_extra_args,
    }

    warmup_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # ── Capture for closures ──────────────────────────────────────────────────
    _sidecar_commit_mode = stitch.sidecar_commit_mode
    _sidecar_debug_requests = stitch.sidecar_debug_requests
    _delta_bulletin_root = stitch.delta_bulletin_root
    _rollout_num_gpus_per_engine = stitch.rollout_num_gpus_per_engine

    # ── Server class (Flash SGLang + stitch sidecar) ──────────────────────────

    import modal

    @app.cls(
        image=image,
        gpu=f"{rollout_gpu_type}:{stitch.rollout_num_gpus_per_engine}",
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
        serialized=True,
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
            from autoinference_utils.endpoint import (
                SGLangEndpoint,
                warmup_chat_completions,
            )

            self.endpoint = SGLangEndpoint(
                model_path=model_name,
                worker_port=SGLANG_PORT,
                tp=_rollout_num_gpus_per_engine,
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

            from huggingface_hub import snapshot_download

            base_checkpoint_dir = snapshot_download(model_name, local_files_only=True)
            self.sidecar = _start_sidecar(
                sidecar_port=SIDECAR_PORT,
                sglang_port=SGLANG_PORT,
                bulletin_root=_delta_bulletin_root,
                local_checkpoint_dir=LOCAL_CHECKPOINT_PATH,
                base_checkpoint_dir=base_checkpoint_dir,
                volume_name=delta_volume_name,
                commit_mode=_sidecar_commit_mode,
                debug_requests=_sidecar_debug_requests,
            )
            _wait_http(
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
            _terminate_process(getattr(self, "sidecar", None))
            if hasattr(self, "endpoint"):
                self.endpoint.stop()

    # ── Download function ─────────────────────────────────────────────────────

    @app.function(
        image=image,
        volumes={str(HF_CACHE_PATH): hf_cache_volume},
        timeout=2 * 60 * MINUTES,
        secrets=hf_secrets(),
        include_source=False,
        serialized=True,
        name="download",
    )
    def download(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        from modal_training_gym.common.status_reporter import (
            enqueue_framework_status,
            flush as flush_status_reporter,
        )

        if training_run_id:
            enqueue_framework_status(
                training_run_id,
                StitchStatus.DOWNLOAD_MODEL.value,
                url=framework_status_url or None,
                token=framework_status_token or None,
                is_active=True,
            )
        hf_cache_volume.reload()
        model.download()
        hf_cache_volume.commit()
        if training_run_id:
            flush_status_reporter(timeout_seconds=2.0)

    # ── Prepare dataset function ──────────────────────────────────────────────

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * MINUTES,
        secrets=hf_secrets(),
        include_source=False,
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        data_volume.reload()
        prompt_data, eval_paths = SlimeRecipe._resolve_data_paths(dataset)
        dataset.prepare(prompt_data, eval_paths)
        dataset.validate_prepared(prompt_data)
        for ep in (eval_paths or {}).values():
            dataset.validate_prepared(ep)
        data_volume.commit()

    # ── Train function ────────────────────────────────────────────────────────

    train_secrets: list[Secret] = list(hf_secrets())
    if stitch.wandb is not None:
        train_secrets.append(Secret.from_name(stitch.wandb.modal_wandb_secret_name))

    @app.function(
        image=image,
        gpu=gpu_spec,
        memory=stitch.memory,
        cloud=stitch.cloud,
        region=stitch.region,
        volumes=train_volumes,
        timeout=24 * 60 * MINUTES,
        startup_timeout=20 * MINUTES,
        scaledown_window=30 * MINUTES,
        secrets=train_secrets or None,
        experimental_options={"efa_enabled": True},
        include_source=False,
        serialized=True,
        name="train",
    )
    @clustered_if(n_train_nodes > 1, n_train_nodes, gpu_type=stitch.gpu_type)
    def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        from modal_training_gym.common.status_reporter import (
            enqueue_framework_status,
            flush as flush_status_reporter,
        )

        if framework_status_url:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_URL"] = framework_status_url
        if framework_status_token:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"] = framework_status_token

        for volume in train_volumes.values():
            volume.reload()

        cluster = ModalRayCluster()
        cluster.discover_cluster(n_train_nodes)

        os.environ["SLIME_HOST_IP"] = cluster.node_ip
        os.environ["SGLANG_HOST_IP"] = cluster.node_ip
        os.environ["HOST_IP"] = cluster.node_ip
        os.environ.update(stitch.environment or {})

        cluster.start_ray()

        if not cluster.is_head:
            import asyncio

            asyncio.get_event_loop().run_until_complete(cluster.wait_forever())
            return {}

        if training_run_id:
            enqueue_framework_status(
                training_run_id,
                StitchStatus.TRAINING.value,
                url=framework_status_url or None,
                token=framework_status_token or None,
                is_active=True,
            )

        from stitch.providers.modal import resolve_flash_gateway_url

        rollout_url = resolve_flash_gateway_url(app_name, "Server")

        # Build the config namespace for slime's CLI.
        run_id = uuid.uuid4().hex[:12]
        cfg = _build_train_config(
            stitch,
            model=model,
            dataset=dataset,
            rollout_url=rollout_url,
            run_id=run_id,
            delta_bulletin_root=_delta_bulletin_root,
            delta_volume_name=delta_volume_name,
            app_name=app_name,
        )

        # Claim the pool before training starts.
        from modal_training_gym.frameworks.stitch._hooks import claim_pool

        claim_pool(cfg)

        tmpdir = tempfile.mkdtemp()
        _prepare_slime_config(cfg, tmpdir)
        cmd = _build_train_cmd(cfg, SLIME_ROOT)

        print(f"Training: nodes={n_train_nodes}, rollout_endpoint={rollout_url}")
        print(f"Command: {cmd}")
        subprocess.run(["bash", "-lc", cmd], check=True)

        if training_run_id:
            flush_status_reporter(timeout_seconds=2.0)

        return {
            "training_run_id": training_run_id,
            "modal_app_id": modal_app_id,
        }

    return app


# ── Helper functions ──────────────────────────────────────────────────────────


def _start_sidecar(
    *,
    sidecar_port: int,
    sglang_port: int,
    bulletin_root: str,
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    volume_name: str,
    commit_mode: str,
    debug_requests: bool = False,
) -> subprocess.Popen:
    """Launch the stitch weight-sync sidecar as a subprocess."""
    cmd = [
        "python3",
        "-m",
        "modal_training_gym.frameworks.stitch._sidecar",
        "--host",
        "0.0.0.0",
        "--port",
        str(sidecar_port),
        "--upstream-url",
        f"http://127.0.0.1:{sglang_port}",
        "--bulletin-root",
        bulletin_root,
        "--local-checkpoint-dir",
        local_checkpoint_dir,
        "--base-checkpoint-dir",
        base_checkpoint_dir,
        "--volume-name",
        volume_name,
        "--commit-mode",
        commit_mode,
    ]
    if debug_requests:
        cmd.append("--debug-requests")
    print("Starting sidecar:", " ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def _wait_http(url: str, process: subprocess.Popen | None, timeout: int) -> None:
    """Wait for an HTTP endpoint to become healthy."""
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"process exited while waiting for {url}: code={process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}; last error: {last_error}")


def _terminate_process(process: subprocess.Popen | None) -> None:
    """Gracefully terminate a subprocess."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


class _TrainNamespace:
    """Minimal namespace that the stitch hooks and slime CLI read from."""

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def cli_args(self) -> list[str]:
        import json as _json

        out: list[str] = []
        for key, val in vars(self).items():
            if key.startswith("_") or key in _CLI_SKIP:
                continue
            if val is None or val is False or val == "":
                continue
            flag = f"--{key.replace('_', '-')}"
            if val is True:
                out.append(flag)
            elif isinstance(val, dict) and key in JSON_CONFIG_FIELDS:
                out += [flag, _json.dumps(val)]
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out


_CLI_SKIP = {
    "recipe_type",
    "environment",
    "async_mode",
    "wandb",
    "name",
    "app_tags",
    "capture_trace",
    "trace_sample_limit",
    "image_overlay",
    "local_slime",
    "memory",
    "cloud",
    "region",
    "checkpoint",
    "custom_rm_function",
    "custom_generate_function",
    "custom_rollout_log_function",
    "custom_eval_rollout_log_function",
    "rollout_function",
    "custom_megatron_before_log_prob_hook",
    "custom_megatron_before_train_step_hook",
    "sglang_request_params",
    "slime_model_script",
    "source_hf_checkpoint",
    "megatron_conversion_hf_checkpoint",
    "patch_files",
    "image_run_commands",
    "image_env",
    "train_function_kwargs",
    "conversion_pipeline_model_parallel_size",
    "conversion_tensor_model_parallel_size",
}

# StitchRecipe-only fields that are NOT slime CLI args.
_STITCH_SKIP = {
    "rollout_min_containers",
    "rollout_proxy_regions",
    "rollout_gpu_type",
    "sidecar_commit_mode",
    "sidecar_debug_requests",
    "delta_volume_name",
    "delta_bulletin_root",
    "sglang_server_extra_args",
    "stitch_repo_url",
    "stitch_repo_ref",
    "slime_fork_url",
    "slime_fork_ref",
    "update_weight_delta_encoding",
    "update_weight_delta_checksum",
    "rollout_request_weight_version_mode",
    "rollout_request_weight_version_lag",
    "rollout_request_retry_attempts",
    "rollout_request_retry_sleep",
    "rollout_session_affinity_header",
    "sglang_server_concurrency",
}


def _build_train_config(
    recipe: StitchRecipe,
    *,
    model: ModelConfig,
    dataset: DatasetConfig,
    rollout_url: str,
    run_id: str,
    delta_bulletin_root: str,
    delta_volume_name: str,
    app_name: str,
) -> _TrainNamespace:
    """Build a namespace with all slime CLI fields + stitch-specific config."""
    fields = recipe._fields(dataset=dataset, model=model)

    # Remove StitchRecipe-only fields (not slime CLI args).
    for key in _STITCH_SKIP:
        fields.pop(key, None)

    # Stitch-specific overrides
    fields["rollout_endpoint_url"] = rollout_url
    fields["update_weight_disk_dir"] = f"{delta_bulletin_root}/{run_id}"

    # Stitch hooks (publish + rollout request gating)
    fields["custom_delta_pre_push_path"] = (
        "modal_training_gym.frameworks.stitch._hooks.commit_and_wake"
    )
    fields["custom_rollout_request_hook_path"] = (
        "stitch.trainers.slime.rollout_request_weight_version_hook"
    )

    # Pass stitch config to hooks via custom_config_path (YAML dict).
    custom_config = fields.get("custom_config_path") or {}
    if isinstance(custom_config, str):
        custom_config = {}
    custom_config.update(
        {
            "update_weight_delta_volume_name": delta_volume_name,
            "rollout_modal_flash_app_name": app_name,
            "rollout_modal_flash_server_cls_name": "Server",
            "run_id": run_id,
            "rollout_request_weight_version_mode": recipe.rollout_request_weight_version_mode,
            "rollout_request_weight_version_lag": recipe.rollout_request_weight_version_lag,
            "rollout_request_retry_attempts": recipe.rollout_request_retry_attempts,
            "rollout_request_retry_sleep": recipe.rollout_request_retry_sleep,
            "rollout_session_affinity_header": recipe.rollout_session_affinity_header,
        }
    )
    fields["custom_config_path"] = custom_config

    cfg = _TrainNamespace(**fields)
    cfg.environment = dict(recipe.environment or {})
    cfg.async_mode = recipe.async_mode
    cfg.slime_model_script = recipe.slime_model_script
    return cfg


def _prepare_slime_config(cfg: _TrainNamespace, tmpdir: str) -> None:
    """Resolve HF repo IDs and materialize inline YAML configs."""
    from huggingface_hub import snapshot_download
    import yaml

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(cfg, attr, None)) and not str(val).startswith("/"):
            setattr(cfg, attr, snapshot_download(val, local_files_only=True))

    for field_name in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(cfg, field_name, None), dict):
            path = os.path.join(tmpdir, f"{field_name}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(cfg, field_name, path)


def _build_train_cmd(cfg: _TrainNamespace, slime_root: str) -> str:
    """Build the SLIME training command."""
    train_script = f"{slime_root}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    cli_args = cfg.cli_args()
    model_script = cfg.slime_model_script
    if model_script:
        inner = (
            f"source {slime_root}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} "
            f"{shlex.join(cli_args)}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(cli_args)}"
