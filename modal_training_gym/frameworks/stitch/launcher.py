"""Factory that builds a Modal app for a disaggregated slime run via stitch.

Same shape as the colocated launchers (``build_slime_app`` / ``build_miles_app``):
:func:`build_stitch_app` returns a ``modal.App`` with ``download``,
``prepare_dataset``, and ``train``, so a run is one call::

    TrainConfig(model=..., dataset=..., recipe=StitchRecipe(...)).train()

What differs is what the app contains: rollouts are served by a Modal Flash pool
of SGLang replicas (the ``Server`` class, brought up with the app) that self-sync
to sparse weight deltas the clustered ``train`` function publishes to a Modal
Volume bulletin board. The trainer reaches the pool through its Flash gateway,
resolved from the in-app class handle — so the single ``train()`` call works in an
ephemeral run, with no separate ``modal deploy`` step.

The app is still deployable (``modal deploy``) when a pool should outlive a single
run; only then can the publish hook wake replicas by app name — otherwise they
pick the new pointer up on their next reconcile poll.

This packages the stitch ``slime_disagg`` cookbook (``cookbook/slime_disagg``)
around a training-gym ``StitchRecipe`` + ``ModelConfig`` + ``DatasetConfig``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from pathlib import PurePosixPath
from types import SimpleNamespace

import cloudpickle
import modal
import modal.experimental

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS, modal_tag_value
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.framework import Framework, resolve_caller_module
from modal_training_gym.common.modal_refs import register_modal_cloudpickle_reducers
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.ray_cluster import clustered_if
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.run import (
    TrainingRun,
    TrainingRunStatus,
    record_wandb_attempt,
    wandb_run_id_for_attempt,
)
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.wandb import preflight_wandb
from modal_training_gym.frameworks.stitch import serving_image
from modal_training_gym.train_recipes.stitch_recipe.pins import (
    SLIME_ROOT,
    STITCH_REPO_REF,
    STITCH_REPO_URL,
)
from modal_training_gym.train_recipes.stitch_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    HOOK_CONFIG_FIELDS,
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
# Where each replica tees its sidecar log (its own volume — see below).
ROLLOUT_LOG_PATH = "/rollout-logs"


class _SlimeArgs:
    """Runtime carrier for the slime args the trainer runs with.

    The recipe + model + dataset resolve to a plain field dict
    (:meth:`StitchRecipe.to_payload`); ``train`` rebuilds this carrier, injects
    the per-run fields (rollout endpoint, bulletin dir, custom config), then
    materializes YAML configs and builds the ``train.py`` command from
    :meth:`cli_args`.
    """

    _CONTROL = {"async_mode", "slime_model_script"}

    # Per-run fields the trainer injects (the rest come from the field dict).
    rollout_endpoint_url: str
    update_weight_disk_dir: str
    custom_config_path: dict | str

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


def _stitch_trainer_image(recipe: StitchRecipe) -> modal.Image:
    """The slime-fork trainer image. The rollout pool serves on a different image
    (:func:`serving_image.build_serving_image`): it installs no trainer package, and
    it needs the SGLang fork that exposes ``/stage_weight_update``."""
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
            "httpx",  # stitch's pool client (wake fan-out)
            # slime is installed --no-deps, but the trainer-side delta ENCODER needs
            # these (zstd compress + xxh3/blake3 checksums).
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
    # Mount the package so the trainer and the Ray workers can import the hooks.
    image = image.add_local_python_source("modal_training_gym", copy=True)
    return image


def _resolve_container_app_id() -> str:
    """Best-effort Modal app id from inside the running Trainer container, used
    as a fallback when the client didn't thread one in. A spawned deployed
    function does not reliably get ``MODAL_APP_ID`` in its env, so also consult
    the container's bound App object."""
    app_id = os.environ.get("MODAL_APP_ID", "")
    if app_id:
        return app_id
    try:
        container_app = modal.App._get_container_app()
        return (container_app.app_id if container_app else "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _record_run_started(
    *,
    run_id: str,
    recipe: StitchRecipe,
    model: ModelConfig | None,
    dataset: DatasetConfig | None,
    config_fields: dict,
    modal_app_id: str = "",
) -> TrainingRun | None:
    """Write a ``RUNNING`` :class:`TrainingRun` to the ``training-gym-metadata``
    Volume so the disagg run shows up in the dashboard (the deployed app is
    already tagged for auto-discovery; this adds the run record slime writes for
    itself in the colocated flow). Best-effort: a metadata hiccup must never take
    down the training run, so failures are logged and swallowed."""
    try:
        modal_app_id = modal_app_id or _resolve_container_app_id()
        wandb_block: dict = {}
        if recipe.wandb is not None:
            # Resolve the W&B entity for a dashboard deep-link; keep the run
            # alive if the probe fails (bad key / no access).
            entity = recipe.wandb.entity
            try:
                entity = preflight_wandb(recipe.wandb) or entity
            except Exception as exc:  # noqa: BLE001
                print(f"W&B preflight for dashboard deep-link failed: {exc}")
            wandb_block = {
                "project": recipe.wandb.project,
                "group": recipe.wandb.group,
                "entity": entity,
                "run_id": wandb_run_id_for_attempt(run_id, 1),
            }
        config_summary = {
            "model": {"model_name": model.model_name} if model else {},
            "dataset": (
                {
                    "hf_repo": getattr(dataset, "hf_repo", ""),
                    "name": type(dataset).__name__,
                }
                if dataset
                else {}
            ),
            "recipe": config_fields,
            "wandb": wandb_block,
            "lr": recipe.lr,
            "global_batch_size": recipe.global_batch_size,
        }
        created_at = int(time.time())
        run_record = TrainingRun(
            training_run_id=run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_dashboard_url(modal_app_id),
            framework=Framework.STITCH,
            config=config_summary,
            status=TrainingRunStatus.RUNNING,
            created_at=created_at,
            started_at=created_at,
        )
        if wandb_block:
            record_wandb_attempt(
                run_record,
                entity=wandb_block["entity"],
                project=wandb_block["project"],
                group=wandb_block["group"],
                run_id=wandb_block["run_id"],
                attempt_count=1,
            )
        run_record.save()
        print(f"TrainingRun recorded for dashboard: {run_id}")
        return run_record
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to record TrainingRun {run_id} for dashboard: {exc}")
        return None


def _record_run_finished(
    run_record: TrainingRun | None, status: TrainingRunStatus
) -> None:
    """Stamp the terminal status + duration on the dashboard run record.
    Best-effort, mirroring :func:`_record_run_started`."""
    if run_record is None:
        return
    try:
        finished_at = int(time.time())
        run_record.status = status
        run_record.ended_at = finished_at
        if run_record.completed_at is None:
            run_record.completed_at = finished_at
        run_record.duration_seconds = max(0, finished_at - run_record.started_at)
        run_record.save()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to finalize dashboard TrainingRun: {exc}")


def build_stitch_app(
    *,
    model: ModelConfig,
    dataset: DatasetConfig,
    recipe: StitchRecipe,
    training_run_id: str = "",
    name: str | None = None,
    group_id: str | None = None,
) -> modal.App:
    """Build the Modal App for disaggregated slime training.

    Returns an app with ``download``, ``prepare_dataset``, and ``train`` (same
    surface as ``build_slime_app``), plus the ``Server`` Flash-pool class that
    serves rollouts. ``train`` brings the pool's gateway up, claims it for the
    run, and drives slime; :class:`~modal_training_gym.common.train.TrainConfig`
    calls it for :class:`StitchRecipe` recipes.
    """
    StitchRecipe._resolve_data_paths(dataset)  # validate dataset paths resolve

    # Serialize the caller's module by value so inline ModelConfig/DatasetConfig
    # subclasses defined in a user script reach the containers.
    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)
    register_modal_cloudpickle_reducers()

    app_name = recipe.name or name or f"stitch-{modal_tag_value(model.model_name)}"
    # Volumes are keyed by recipe (not by run) so runs of the same recipe reuse
    # the same dataset / checkpoints / bulletin board.
    volume_prefix = f"stitch-{modal_tag_value(type(recipe).__name__)}"
    delta_volume_name = recipe.delta_volume_name or f"{volume_prefix}-delta-bulletin"
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

    image = _stitch_trainer_image(recipe)
    server_image = serving_image.build_serving_image(
        hf_cache_path=str(HF_CACHE_PATH),
        delta_volume_name=delta_volume_name,
        bulletin_root=delta_bulletin_root,
        runtime=recipe.sglang_runtime,
    )

    # Structural SGLang args (derived from the recipe) merged under the recipe's
    # per-model tuning. Deltas are applied by the engine behind
    # /stage_weight_update, so no engine-side delta server args are passed.
    sglang_server_args = {
        "--served-model-name": model_name,
        "--dtype": "bfloat16",
        "--cuda-graph-max-bs-decode": str(rollout_concurrency),
        "--max-running-requests": str(rollout_concurrency),
        "--trust-remote-code": "",
        **dict(recipe.sglang_server_args),
    }

    hf_cache_volume = modal.Volume.from_name(
        "huggingface-cache", create_if_missing=True
    )
    data_volume = modal.Volume.from_name(
        f"{volume_prefix}-data", create_if_missing=True
    )
    checkpoints_volume_name = f"{volume_prefix}-checkpoints"
    checkpoints_volume = modal.Volume.from_name(
        checkpoints_volume_name, create_if_missing=True
    )
    delta_volume = modal.Volume.from_name(
        delta_volume_name, create_if_missing=True, version=2
    )
    sglang_cache_volume = modal.Volume.from_name("sglang-cache", create_if_missing=True)
    # Durable per-replica sidecar logs. Deliberately NOT the bulletin volume: an
    # open file there makes the reconciler's Volume.reload() fail with "there are
    # open files preventing the operation". v2 so writes are visible without an
    # explicit commit from a replica that may be killed at any time.
    rollout_log_volume = modal.Volume.from_name(
        f"{volume_prefix}-rollout-logs", create_if_missing=True, version=2
    )
    train_volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
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

    @app.cls(
        image=server_image,
        gpu=f"{recipe.gpu_type}:{recipe.rollout_num_gpus_per_engine}",
        cloud=recipe.cloud,
        region=recipe.region,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            serving_image.SGLANG_CACHE_PATH: sglang_cache_volume,
            delta_bulletin_root: delta_volume,
            ROLLOUT_LOG_PATH: rollout_log_volume,
        },
        secrets=[hf_secret],
        min_containers=recipe.rollout_min_containers,
        max_containers=recipe.rollout_max_containers,
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
            from modal_training_gym.frameworks.stitch import server

            server.serve_startup(
                self,
                model_name=model_name,
                sglang_args=sglang_server_args,
                disk_load_format=recipe.sidecar_disk_load_format,
                tp=recipe.rollout_num_gpus_per_engine,
                concurrency=rollout_concurrency,
                sidecar_port=SIDECAR_PORT,
                sglang_port=SGLANG_PORT,
                bulletin_root=delta_bulletin_root,
                local_checkpoint_dir=LOCAL_CHECKPOINT_PATH,
                delta_update_mode=recipe.sglang_delta_update_mode,
                volume_name=delta_volume_name,
                commit_mode=recipe.sidecar_commit_mode,
                flush_cache_on_commit=recipe.sidecar_flush_cache_on_commit,
                debug_requests=recipe.sidecar_debug_requests,
                log_dir=ROLLOUT_LOG_PATH,
                startup_timeout=SERVER_STARTUP_TIMEOUT,
            )

        @modal.exit()
        def stop(self) -> None:
            from modal_training_gym.frameworks.stitch import server

            server.serve_stop(self)

    @app.function(
        image=image,
        gpu=f"{recipe.gpu_type}:{recipe.actor_num_gpus_per_node}",
        memory=memory,
        cloud=recipe.cloud,
        region=recipe.region,
        volumes=train_volumes,
        secrets=train_secrets,
        timeout=24 * 60 * MINUTES,
        startup_timeout=20 * MINUTES,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train",
    )
    @clustered_if(True, n_train_nodes, gpu_type=recipe.gpu_type)
    def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
        rollout_endpoint_url: str = "",
    ) -> dict:
        """Bring up Ray, claim the rollout pool for this run, and drive slime."""
        del modal_app_url  # derived from modal_app_id in the run record
        from modal_training_gym.frameworks.stitch import (
            bulletin_hooks,
            ray_cluster,
            sidecar_process,
            trainer_helpers,
        )

        rank, master_addr, my_ip = ray_cluster.get_modal_cluster_context(n_train_nodes)
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
        if framework_status_url:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_URL"] = framework_status_url
        if framework_status_token:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"] = framework_status_token
        sidecar_process.start_host_mem_monitor()  # per-node host-RAM trace

        # Rank 0 drives the run; the other ranks only host Ray workers, and stay
        # alive until Modal tears the cluster down with rank 0's input.
        if rank != 0:
            ray_cluster.start_ray_worker(my_ip, master_addr, ray_port=RAY_PORT)
            ray_cluster.wait_for_teardown()
            return {}
        ray_cluster.start_ray_head(my_ip, n_train_nodes, ray_port=RAY_PORT)
        for volume in (hf_cache_volume, data_volume, checkpoints_volume):
            volume.reload()

        payload = recipe.to_payload(model=model, dataset=dataset)
        cfg = _SlimeArgs(
            payload["fields"],
            async_mode=payload["async_mode"],
            slime_model_script=payload["slime_model_script"],
        )
        # The pool's Flash gateway, resolved by whoever launched this call (see
        # _PoolAwareTrain). Falls back to a lookup by app name, which only works
        # against a deployed pool.
        cfg.rollout_endpoint_url = (
            rollout_endpoint_url or trainer_helpers.deployed_gateway_url(app_name)
        )
        # Flash holds requests through a cold-starting pool, but slime's first
        # rollout would otherwise meet engines that are still loading.
        trainer_helpers.await_gateway_ready(
            cfg.rollout_endpoint_url, timeout_seconds=SERVER_STARTUP_TIMEOUT
        )
        # Fresh run id per launch: slime writes this run's chain under
        # <bulletin_root>/<run_id>/weight_v{N}/ and the canonical `latest`
        # pointer is self-identifying, so a new run never collides with a
        # finished one — no manual bulletin reset needed.
        run_id = uuid.uuid4().hex[:12]
        cfg.update_weight_disk_dir = f"{delta_bulletin_root}/{run_id}"
        # stitch's publish + request hooks read these off the slime args
        # namespace; merge over any user extra_config already on
        # custom_config_path.
        custom_config = dict(getattr(cfg, "custom_config_path", None) or {})
        custom_config.update(
            {field: getattr(recipe, field) for field in sorted(HOOK_CONFIG_FIELDS)}
        )
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

        record_id = training_run_id or run_id
        run_record = _record_run_started(
            run_id=record_id,
            recipe=recipe,
            model=model,
            dataset=dataset,
            config_fields=payload["fields"],
            modal_app_id=modal_app_id,
        )
        wandb_run_id = ""
        if recipe.wandb is not None:
            # Force slime's W&B run to use the same id recorded in the
            # dashboard deep-link (slime/wandb honor these env vars). Without
            # this, wandb autogenerates a run id and the dashboard link 404s.
            wandb_run_id = wandb_run_id_for_attempt(record_id, 1)
            os.environ["WANDB_RUN_ID"] = wandb_run_id
            os.environ["WANDB_RESUME"] = "allow"
            if recipe.wandb.entity:
                os.environ["WANDB_ENTITY"] = recipe.wandb.entity
        # Tee slime's output to the checkpoints volume: a container's log window
        # only keeps the tail, so a failure whose traceback scrolled past (slime
        # retries rollout requests loudly) is otherwise unreadable afterwards.
        trainer_log = CHECKPOINTS_PATH / "logs" / f"{record_id}-trainer.log"
        trainer_log.parent.mkdir(parents=True, exist_ok=True)
        status = TrainingRunStatus.COMPLETED
        try:
            subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"set -o pipefail; ({cmd}) 2>&1 | tee -a {trainer_log}",
                ],
                check=True,
            )
        except BaseException:
            status = TrainingRunStatus.FAILED
            raise
        finally:
            _record_run_finished(run_record, status)
            checkpoints_volume.commit()

        result = TrainResult(
            app_name=app_name,
            framework=Framework.STITCH,
            training_run_id=record_id,
            checkpoint_dir=str(recipe.save),
            checkpoints_volume_name=checkpoints_volume_name,
            checkpoints_mount_path=str(CHECKPOINTS_PATH),
            model_config=model,
            wandb_project=recipe.wandb.project if recipe.wandb else "",
            wandb_entity=recipe.wandb.entity if recipe.wandb else "",
            wandb_training_run_id=wandb_run_id,
            group_id=group_id or "",
            extra={"rollout_endpoint_url": cfg.rollout_endpoint_url, "run_id": run_id},
        )
        result.save()
        checkpoints_volume.commit()
        return result._to_dict()

    @app.function(
        image=image,
        volumes={str(HF_CACHE_PATH): hf_cache_volume},
        timeout=2 * 60 * MINUTES,
        secrets=[hf_secret],
        serialized=True,
        name="download",
    )
    def download() -> None:
        model.download()
        hf_cache_volume.commit()

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * MINUTES,
        secrets=[hf_secret],
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset() -> None:
        data_volume.reload()
        prompt_data, eval_paths = StitchRecipe._resolve_data_paths(dataset)
        dataset.prepare(prompt_data, eval_paths)
        data_volume.commit()

    # Expose the functions as attributes (app.train, app.download, …) the way the
    # other launchers do, so callers address them without the registry.
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)
    app.train = _PoolAwareTrain(train, Server)  # pyright: ignore[reportAttributeAccessIssue]

    return app


class _PoolAwareTrain:
    """``app.train`` proxy that resolves the rollout pool's gateway client-side.

    The trainer can't discover it itself: an ephemeral app can't be looked up by
    name (``flash_get_containers`` / ``Cls.from_name`` need a deployed app), its
    containers' app object exposes no sibling objects, and a ``Cls`` handle can't
    be captured in the trainer's closure (Modal refuses to serialize unhydrated
    objects). The launching client, inside ``app.run()``, does have the hydrated
    handle — so it passes the gateway URL in as an argument.
    """

    def __init__(self, fn: modal.Function, server_cls: modal.Cls) -> None:
        self._fn = fn
        self._server_cls = server_cls

    def _with_gateway(self, kwargs: dict) -> dict:
        from modal_training_gym.frameworks.stitch import trainer_helpers

        if not kwargs.get("rollout_endpoint_url"):
            kwargs["rollout_endpoint_url"] = trainer_helpers.flash_gateway_url(
                self._server_cls
            )
        return kwargs

    def spawn(self, *args, **kwargs) -> modal.FunctionCall:
        return self._fn.spawn(*args, **self._with_gateway(kwargs))

    def remote(self, *args, **kwargs):
        return self._fn.remote(*args, **self._with_gateway(kwargs))

    def __getattr__(self, name: str):
        return getattr(self._fn, name)


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
