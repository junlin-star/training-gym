import asyncio
import base64
import hashlib
import inspect
import os
import shlex
import subprocess
import shutil
import tempfile
import textwrap
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS, hf_secrets
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.dataset import DatasetConfig, HarborDataset
from modal_training_gym.common.framework import (
    Framework,
    mount_tools_dir,
    resolve_caller_module,
)
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.ray_cluster import ModalRayCluster
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.train_recipes.miles_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    MilesConfig,
)
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.frameworks.miles.modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
    prepare_miles_config,
    resolve_checkpoint_ref,
)

MILES_ROOT = "/root/miles"
HARBOR_PKG_VERSION = "0.6.6"

_MILES_PATCHES = Path(__file__).parent / "modal_helpers" / "patches"
_PATCH_SGLANG_ABORT_B64 = encode_patch("patch_sglang_abort", _MILES_PATCHES)


def _build_miles_base_image(miles: MilesConfig) -> Image:
    image = (
        Image.from_registry(miles.docker_image)
        .entrypoint([])
        .run_commands(
            f"rm -rf {HF_CACHE_PATH} 2>/dev/null || true",
            f"echo {_PATCH_SGLANG_ABORT_B64} | base64 -d | python3",
        )
    )
    if miles.image_env:
        image = image.env(miles.image_env)
    if miles.image_run_commands:
        image = image.run_commands(*miles.image_run_commands)
    return image


def _has_torch_dist_checkpoint(save_path: str) -> bool:
    if not os.path.isdir(save_path):
        return False

    tracker_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker_path):
        try:
            with open(tracker_path) as f:
                marker = f.read().strip()
        except OSError:
            marker = ""
        if marker == "release":
            return os.path.isdir(os.path.join(save_path, "release"))
        if marker.isdigit():
            iter_dir = f"iter_{int(marker):07d}"
            return os.path.isdir(os.path.join(save_path, iter_dir))

    try:
        return any(
            entry.is_dir()
            and (entry.name == "release" or entry.name.startswith("iter_"))
            for entry in os.scandir(save_path)
        )
    except OSError:
        return False


def build_miles_app(
    *,
    training_run_id: str,
    miles: MilesConfig,
    model: ModelConfig,
    dataset: DatasetConfig,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
) -> App:
    app_name = name or miles.name or f"miles-{type(miles).__name__.lstrip('_').lower()}"

    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file and os.path.isfile(mod_file):
            caller_script = os.path.abspath(mod_file)

    image = _build_miles_base_image(miles)

    for patch_file in miles.patch_files:
        image = image.add_local_file(
            patch_file,
            remote_path=f"/tmp/{os.path.basename(patch_file)}",
            copy=True,
        )

    if miles.local_miles:
        image = image.add_local_dir(
            miles.local_miles,
            remote_path=MILES_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )

    if miles.image_overlay is not None:
        image = miles.image_overlay(image)
        miles.image_overlay = None

    if isinstance(dataset, HarborDataset):
        image = image.uv_pip_install(f"harbor=={HARBOR_PKG_VERSION}")

    image = image.add_local_python_source("modal_training_gym", copy=True)
    image = image.uv_pip_install("randomname")
    image = mount_tools_dir(image)

    if caller_script is not None:
        caller_module_name = os.path.splitext(os.path.basename(caller_script))[0]
        image = image.add_local_file(
            caller_script,
            remote_path=f"/root/{caller_module_name}.py",
            copy=True,
        )

    def _set_custom_config_value(key: str, value: str) -> None:
        cfg = (
            dict(miles.custom_config_path or {})
            if isinstance(miles.custom_config_path, dict)
            else {}
        )
        cfg[key] = value
        miles.custom_config_path = cfg

    def _ship_callable(
        fn: Any,
        *,
        fallback_name: str,
        set_path: Callable[[str], None],
    ) -> None:
        nonlocal image
        if fn is None:
            return
        fn_mod = getattr(fn, "__module__", None) or ""
        if fn_mod.startswith("modal_training_gym"):
            return
        try:
            fn_file = os.path.abspath(inspect.getfile(fn))
        except (TypeError, OSError):
            fn_file = None
        if fn_file and os.path.isfile(fn_file) and fn_file != caller_script:
            fn_module_name = os.path.splitext(os.path.basename(fn_file))[0]
            image = image.add_local_file(
                fn_file,
                remote_path=f"/root/{fn_module_name}.py",
                copy=True,
            )
            set_path(f"{fn_module_name}.{getattr(fn, '__name__', fallback_name)}")
            return
        fn_name = getattr(fn, "__name__", fallback_name)
        try:
            payload = base64.b64encode(cloudpickle.dumps(fn)).decode("ascii")
        except Exception:
            module_src = textwrap.dedent(inspect.getsource(fn))
        else:
            module_src = textwrap.dedent(
                f"""
                import base64
                import cloudpickle

                {fn_name} = cloudpickle.loads(base64.b64decode({payload!r}))
                """
            ).lstrip()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=f"notebook_{fallback_name}_",
            delete=False,
        ) as tmp:
            tmp.write(module_src)
            tmp_path = tmp.name
        mod_name = os.path.splitext(os.path.basename(tmp_path))[0]
        image = image.add_local_file(
            tmp_path,
            remote_path=f"/root/{mod_name}.py",
            copy=True,
        )
        set_path(f"{mod_name}.{fn_name}")

    _ship_callable(
        miles.custom_rm_function,
        fallback_name="custom_rm",
        set_path=lambda path: _set_custom_config_value("custom_rm_path", path),
    )
    _ship_callable(
        miles.custom_generate_function,
        fallback_name="custom_generate",
        set_path=lambda path: _set_custom_config_value(
            "custom_generate_function_path", path
        ),
    )
    miles.custom_rm_function = None
    miles.custom_generate_function = None

    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    checkpoints_volume_name = (
        checkpoint.checkpoints_volume_name
        if checkpoint is not None and checkpoint.checkpoints_volume_name
        else f"{app_name}-checkpoints"
    )
    checkpoints_mount_path = (
        checkpoint.checkpoints_mount_path.rstrip("/") or "/"
        if checkpoint is not None and checkpoint.checkpoints_mount_path
        else str(CHECKPOINTS_PATH).rstrip("/")
    )
    checkpoints_volume = Volume.from_name(
        checkpoints_volume_name, create_if_missing=True
    )
    if checkpoint is not None and checkpoint.path and not model.model_path:
        model.model_path = checkpoint.path

    all_volumes: dict[str | PurePosixPath, Any] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        checkpoints_mount_path: checkpoints_volume,
    }

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": "miles",
        **miles.app_tags,
    }
    if miles.wandb is not None:
        tags["_modal_wandb_project"] = miles.wandb.project
        if miles.wandb.group:
            tags["_modal_wandb_group"] = miles.wandb.group

    app = App(app_name, tags=tags)
    gpu_spec = f"{miles.gpu_type}:{miles.actor_num_gpus_per_node}"

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=4 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="download",
    )
    def download():
        hf_cache_volume.reload()
        checkpoints_volume.reload()
        model.download()
        miles.download_model()
        miles.post_process_model()
        hf_cache_volume.commit()
        checkpoints_volume.commit()

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=4 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        data_volume.reload()
        prompt_data, eval_paths = MilesConfig._resolve_data_paths(dataset)
        if dataset.always_prepare and os.path.exists(prompt_data):
            import shutil

            data_dir = os.path.dirname(prompt_data)
            print(f"always_prepare=True - removing {data_dir}")
            shutil.rmtree(data_dir, ignore_errors=True)
        dataset.prepare(prompt_data, eval_paths)
        dataset.validate_prepared(prompt_data)
        for ep in (eval_paths or {}).values():
            dataset.validate_prepared(ep)
        data_volume.commit()

    convert_nnodes = get_checkpoint_conversion_policy(miles, model=model)[0]
    convert_multi_node = convert_nnodes > 1

    @app.function(
        image=image,
        gpu=gpu_spec,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        experimental_options={"efa_enabled": True} if convert_multi_node else {},
        serialized=True,
        name="convert_checkpoint",
    )
    @clustered(convert_nnodes, rdma=convert_multi_node)
    def convert_checkpoint():
        if getattr(miles, "megatron_to_hf_mode", None) == "bridge":
            print("Bridge mode - no conversion needed.")
            return

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        conversion_hf_checkpoint = (
            getattr(miles, "megatron_conversion_hf_checkpoint", None)
            or getattr(miles, "hf_checkpoint", "")
            or model.model_path
            or model.model_name
        )
        hf_path = resolve_checkpoint_ref(conversion_hf_checkpoint)
        save_path = str(miles.ref_load)
        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
            miles, model=model
        )

        if _has_torch_dist_checkpoint(save_path):
            print(
                f"Found existing torch_dist checkpoint at {save_path}; skipping conversion."
            )
            return

        if num_nodes == 1:
            node_rank, master_addr, nnodes = 0, "127.0.0.1", 1
        else:
            import modal.experimental

            info = modal.experimental.get_cluster_info()
            node_rank = info.rank
            master_addr = info.container_ipv4_ips[0]
            nnodes = len(info.container_ipv4_ips)

        torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
        if nnodes > 1:
            torchrun_args += [
                f"--nnodes={nnodes}",
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                "--master-port=12355",
            ]

        import importlib.util

        spec = importlib.util.find_spec(
            "modal_training_gym.frameworks.miles.modal_helpers.convert_hf_to_torch_dist"
        )
        convert_script = (
            spec.origin
            if spec is not None and num_nodes > 1
            else f"{MILES_ROOT}/tools/convert_hf_to_torch_dist.py"
        )
        if not convert_script:
            raise RuntimeError("Miles checkpoint conversion script not found")

        ep = getattr(miles, "expert_model_parallel_size", 1) or 1
        if miles.miles_model_script and ep > 1:
            # Strip only parallelism flags from MODEL_ARGS so the only
            # source of TP/PP/EP/MTP is extra_args.  Keep --spec and
            # architecture flags (--attention-output-gate, etc.) because
            # the bridge needs them for correct weight mapping.
            filter_cmd = (
                "CONV_ARGS=(); SKIP=0; "
                'for a in "${MODEL_ARGS[@]}"; do '
                '  if [ "$SKIP" -gt 0 ]; then SKIP=$((SKIP-1)); continue; fi; '
                '  case "$a" in '
                "    --tensor-model-parallel-size|--pipeline-model-parallel-size"
                "|--expert-model-parallel-size|--expert-tensor-parallel-size"
                "|--context-parallel-size|--transformer-pipeline-model-parallel-size"
                "|--mtp-num-layers|--virtual-pipeline-model-parallel-size"
                "|--decoder-last-pipeline-num-layers"
                ") SKIP=1; continue ;; "
                "  esac; "
                '  CONV_ARGS+=("$a"); '
                "done"
            )
            # The miles convert_hf_to_torch_dist.py auto-inflates PP to
            # world_size when PP=1 and world_size>1.  For EP>1 conversions
            # that need PP=1 we must neutralise that logic.  Patch the
            # script in-place so the condition also requires
            # SKIP_PP_AUTOINFLATE to be unset.
            patch_cmd = (
                f"sed -i 's/if args.pipeline_model_parallel_size == 1 and world_size > 1:/"
                f"if args.pipeline_model_parallel_size == 1 and world_size > 1 "
                f'and not os.environ.get("SKIP_PP_AUTOINFLATE"):/\' '
                f"{convert_script}"
            )
            cmd = (
                f"source {MILES_ROOT}/{miles.miles_model_script} && {filter_cmd} && "
                f"{patch_cmd} && "
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                '"${CONV_ARGS[@]}" '
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )
        elif miles.miles_model_script:
            cmd = (
                f"source {MILES_ROOT}/{miles.miles_model_script} && "
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"${{MODEL_ARGS[@]}} {' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )
        else:
            cmd = (
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )

        env = {**os.environ, **miles.environment}
        if num_nodes > 1:
            env["SKIP_RELEASE_RENAME"] = "1"
        if ep > 1:
            env["SKIP_PP_AUTOINFLATE"] = "1"

        print(
            f"Conversion layout: nodes={num_nodes}, nproc_per_node={nproc_per_node}, "
            f"node_rank={node_rank}"
        )
        print(f"Running: bash -c {cmd!r}")
        subprocess.run(["bash", "-c", cmd], check=True, env=env)
        checkpoints_volume.commit()

        if node_rank == 0:
            print(f"Saved Megatron torch_dist checkpoint to {save_path}")

    _multi_node = miles.total_nodes > 1

    @app.function(
        image=image,
        gpu=gpu_spec,
        memory=miles.memory,
        cloud=miles.cloud,
        region=miles.region,
        volumes=all_volumes,
        secrets=[
            *(
                []
                if miles.wandb is None
                else [Secret.from_name(miles.wandb.modal_wandb_secret_name)]
            ),
        ],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True} if _multi_node else {},
        ephemeral_disk=miles.ephemeral_disk,
        serialized=True,
        name="train",
    )
    @clustered(miles.total_nodes, rdma=_multi_node)
    async def train(modal_app_id: str = "", modal_app_url: str = ""):
        modal_app_id = modal_app_id or os.environ.get("MODAL_APP_ID", "")
        modal_app_url = modal_app_url or modal_app_dashboard_url(modal_app_id)

        await asyncio.gather(
            hf_cache_volume.reload.aio(),
            data_volume.reload.aio(),
            checkpoints_volume.reload.aio(),
        )

        cluster = ModalRayCluster()
        cluster.discover_cluster(miles.total_nodes)

        os.environ["MILES_HOST_IP"] = cluster.node_ip
        os.environ["SGLANG_HOST_IP"] = cluster.node_ip
        os.environ["HOST_IP"] = cluster.node_ip

        prep_id = hashlib.sha1(training_run_id.encode("utf-8")).hexdigest()[:16]
        prep_marker = os.path.join(
            checkpoints_mount_path, f".training_gym_prepared_{prep_id}"
        )
        prep_error = f"{prep_marker}.error"

        async def _prepare_shared_inputs() -> None:
            if model:
                cache_dir = (
                    HF_CACHE_PATH
                    / "hub"
                    / ("models--" + model.model_name.replace("/", "--"))
                )
                snapshots_dir = cache_dir / "snapshots"
                has_snapshot = snapshots_dir.is_dir() and any(snapshots_dir.iterdir())
                model_path = getattr(model, "model_path", None)
                has_model_path = True
                if model_path:
                    model_path_obj = Path(model_path)
                    has_model_path = model_path_obj.exists() and (
                        not model_path_obj.is_dir() or any(model_path_obj.iterdir())
                    )
                if not has_snapshot or not has_model_path:
                    print(f"Downloading model {model.model_name}...")
                    model.download()
                if hasattr(model, "prepare_runtime_cache"):
                    model.prepare_runtime_cache()

            miles.download_model()
            miles.post_process_model()
            await hf_cache_volume.commit.aio()
            await checkpoints_volume.commit.aio()

            prompt_data, eval_paths = MilesConfig._resolve_data_paths(dataset)
            needs_prepare = not os.path.exists(prompt_data)
            if dataset.always_prepare and os.path.exists(prompt_data):
                data_dir = os.path.dirname(prompt_data)
                print(f"always_prepare=True - removing {data_dir}")
                shutil.rmtree(data_dir, ignore_errors=True)
                needs_prepare = True
            if needs_prepare:
                print(f"Preparing dataset ({prompt_data})...")
                dataset.prepare(prompt_data, eval_paths)
                await data_volume.commit.aio()
            dataset.validate_prepared(prompt_data)
            for ep in (eval_paths or {}).values():
                if os.path.exists(ep):
                    dataset.validate_prepared(ep)

        if cluster.is_head:
            try:
                await _prepare_shared_inputs()
            except BaseException as exc:
                os.makedirs(os.path.dirname(prep_error), exist_ok=True)
                with open(prep_error, "w") as f:
                    f.write(repr(exc))
                await checkpoints_volume.commit.aio()
                raise
            os.makedirs(os.path.dirname(prep_marker), exist_ok=True)
            with open(prep_marker, "w") as f:
                f.write(str(time.time()))
            await checkpoints_volume.commit.aio()
        else:
            deadline = time.time() + 4 * 60 * 60
            while True:
                await asyncio.gather(
                    hf_cache_volume.reload.aio(),
                    data_volume.reload.aio(),
                    checkpoints_volume.reload.aio(),
                )
                if os.path.exists(prep_marker):
                    break
                if os.path.exists(prep_error):
                    with open(prep_error) as f:
                        raise RuntimeError(f"Head preparation failed: {f.read()}")
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for head preparation marker")
                await asyncio.sleep(5)

        cluster.start_ray()

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        created_at = int(time.time())
        print(f"Training run id: {training_run_id}")
        config_summary = {
            "model": {"model_name": model.model_name} if model else {},
            "recipe": {
                "gpu_type": miles.gpu_type,
                "actor_num_nodes": miles.actor_num_nodes,
                "actor_num_gpus_per_node": miles.actor_num_gpus_per_node,
            },
            "wandb": (
                {"project": miles.wandb.project, "group": miles.wandb.group}
                if miles.wandb
                else {}
            ),
            "dataset": {
                "hf_repo": getattr(dataset, "hf_repo", ""),
                "name": type(dataset).__name__,
            },
            "lr": miles.lr,
            "global_batch_size": miles.global_batch_size,
        }
        run_record = TrainingRun(
            training_run_id=training_run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url,
            framework=Framework.MILES,
            config=config_summary,
            created_at=created_at,
            started_at=created_at,
        )
        await run_record.save_async()
        print(f"TrainingRun recorded: {training_run_id}")

        try:
            prepare_miles_config(miles, model, tempfile.mkdtemp())

            if wandb_key := os.environ.get("WANDB_API_KEY", ""):
                if miles.wandb is not None:
                    miles.wandb.key = wandb_key

            recipe_default_save_root = str(CHECKPOINTS_PATH).rstrip("/")
            mounted_save_root = checkpoints_mount_path
            configured_save_root = (
                str(miles.save).rstrip("/") if miles.save else mounted_save_root
            )
            base_save_root = (
                mounted_save_root
                if configured_save_root == recipe_default_save_root
                else configured_save_root
            )
            save_root = (
                f"{mounted_save_root}/{training_run_id}"
                if base_save_root == mounted_save_root
                else configured_save_root
            )
            os.makedirs(save_root, exist_ok=True)

            original_save = miles.save
            original_load = miles.load
            miles.save = save_root
            if _has_torch_dist_checkpoint(save_root):
                print(
                    f"Detected existing checkpoint in {save_root}; "
                    "will resume training from last saved iteration."
                )
                miles.load = save_root
            try:
                cmd = build_train_cmd(miles, MILES_ROOT, model=model, dataset=dataset)
            finally:
                miles.save = original_save
                miles.load = original_load

            runtime_env = {
                "env_vars": {
                    "no_proxy": f"127.0.0.1,{cluster.head_addr}",
                    "MASTER_ADDR": cluster.head_addr,
                    **miles.environment,
                }
            }

            mode = "async" if miles.async_mode else "sync"
            print(
                f"Training {app_name} - {miles.total_nodes} node(s) x {gpu_spec} ({mode})"
            )
            print(f"Command: {cmd}, runtime_env: {runtime_env}")

            async with cluster.forward_dashboard() as tunnel:
                print(f"Ray dashboard: {tunnel.url}")
                await cluster.submit_and_tail(cmd, runtime_env=runtime_env)

            result_kwargs = {
                "app_name": app_name,
                "framework": Framework.MILES,
                "training_run_id": training_run_id,
                "checkpoint_dir": save_root,
                "model_config": model,
                "checkpoints_volume_name": checkpoints_volume_name,
                "checkpoints_mount_path": checkpoints_mount_path,
            }
            accepted_fields = set(inspect.signature(TrainResult).parameters)
            result = TrainResult(
                **{k: v for k, v in result_kwargs.items() if k in accepted_fields}
            )
            await result.save_async()
            run_record.status = TrainingRunStatus.COMPLETED
            await checkpoints_volume.commit.aio()
            print(f"TrainResult saved: {training_run_id}")
            return result._to_dict()
        except KeyboardInterrupt:
            run_record.status = TrainingRunStatus.STOPPED
            raise
        except BaseException:
            run_record.status = TrainingRunStatus.FAILED
            raise
        finally:
            finished_at = int(time.time())
            run_record.ended_at = finished_at
            if run_record.completed_at is None:
                run_record.completed_at = finished_at
            run_record.duration_seconds = max(0, finished_at - run_record.started_at)
            await run_record.save_async()

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
