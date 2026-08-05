# ── Checkpoint ───────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import time

import modal
from modal import App, Volume
from modal.exception import NotFoundError

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.train_result import TrainResult


class CheckpointType(Enum):
    hf = "hf"
    megatron = "megatron"


@dataclass
class Checkpoint:
    """A single discovered checkpoint on the local filesystem."""

    checkpoint_type: CheckpointType
    name: str
    path: str
    timestamp: float
    training_run_id: str = ""
    app_name: str = ""
    checkpoints_volume_name: str = ""
    checkpoints_mount_path: str = ""


def _checkpoint_sort_key(name: str) -> tuple[int, bool, str]:
    stem = name.removeprefix("iter_").removesuffix("_hf")
    return (int(stem) if stem.isdigit() else -1, name.endswith("_hf"), name)


def list_checkpoints(training_run_id: str) -> list[Checkpoint]:
    result = TrainResult.from_training_run_id(training_run_id)
    if result.framework in {
        Framework.SLIME,
        Framework.SLIME.value,
        Framework.MILES,
        Framework.MILES.value,
    }:
        return _list_checkpoints(result)
    raise TrainingGymConfigError(f"Unsupported framework: {result.framework}")


def to_volume_path(
    checkpoint_dir: str, checkpoints_mount_path: str = "/checkpoints"
) -> str:
    """Convert an in-container checkpoint path to a Volume-relative one.

    Modal Volume APIs and ``modal endpoint create --custom-volume-path`` address
    entries relative to the Volume root, not the container mount point, so
    ``/checkpoints/run-1/iter_10`` has to become ``run-1/iter_10``.
    """
    checkpoint_dir_norm = os.path.normpath(checkpoint_dir)
    checkpoints_mount_path_norm = os.path.normpath(checkpoints_mount_path)

    if os.path.isabs(checkpoint_dir_norm):
        if not (
            checkpoint_dir_norm == checkpoints_mount_path_norm
            or checkpoint_dir_norm.startswith(checkpoints_mount_path_norm + os.sep)
        ):
            raise TrainingGymConfigError(
                f"Checkpoint path {checkpoint_dir!r} is not under the checkpoints "
                f"mount {checkpoints_mount_path!r}, so it has no Volume-relative "
                "path. Pass a path inside the mount, or pass the matching "
                "checkpoints_mount_path."
            )
        rel = os.path.relpath(checkpoint_dir_norm, checkpoints_mount_path_norm)
        return "" if rel == "." else rel

    return checkpoint_dir_norm.lstrip("/")


def _list_checkpoints(train_result: "TrainResult") -> list[Checkpoint]:
    checkpoint_dir = train_result.checkpoint_dir.rstrip("/")
    if not checkpoint_dir:
        return []

    def _entry_name(entry: object) -> str:
        return getattr(entry, "path", "").rstrip("/").rsplit("/", 1)[-1]

    def _checkpoint_type(name: str) -> CheckpointType:
        return CheckpointType.hf if name.endswith("_hf") else CheckpointType.megatron

    checkpoints_volume_name = (
        train_result.checkpoints_volume_name or f"{train_result.app_name}-checkpoints"
    )
    checkpoints_mount_path = train_result.checkpoints_mount_path or "/checkpoints"
    prefix = "iter_"
    try:
        rel = to_volume_path(checkpoint_dir, checkpoints_mount_path)
    except TrainingGymConfigError:
        return []
    volume = Volume.from_name(checkpoints_volume_name, create_if_missing=True)

    try:
        entries = {
            _entry_name(entry): entry
            for entry in volume.iterdir(rel or "/", recursive=False)
        }
    except (FileNotFoundError, NotFoundError):
        return []

    def _is_dir_entry(entry: object) -> bool:
        is_dir_fn = getattr(entry, "is_dir", None)
        if callable(is_dir_fn):
            return bool(is_dir_fn())
        entry_type = getattr(entry, "type", None)
        if entry_type is None:
            return False
        entry_type_name = getattr(entry_type, "name", "")
        if isinstance(entry_type_name, str):
            return entry_type_name.upper() == "DIRECTORY"
        return False

    checkpoints: list[Checkpoint] = []
    for entry in sorted(
        (entry for entry in entries.values() if _is_dir_entry(entry)),
        key=lambda entry: _checkpoint_sort_key(_entry_name(entry)),
    ):
        name = _entry_name(entry)
        if not name.startswith(prefix):
            continue

        checkpoints.append(
            Checkpoint(
                checkpoint_type=_checkpoint_type(name),
                name=name,
                path=os.path.join(checkpoint_dir, name),
                timestamp=float(getattr(entry, "mtime", 0.0)),
                training_run_id=train_result.training_run_id,
                app_name=train_result.app_name,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
            )
        )
    return checkpoints


def _conversion_gpu_spec(checkpoint: Checkpoint) -> str:
    if checkpoint.training_run_id:
        try:
            training_run = TrainingRun.from_id(checkpoint.training_run_id)
        except (KeyError, FileNotFoundError, NotFoundError):
            training_run = None
        if training_run is not None:
            recipe_config = training_run.config.get("recipe", {})
            gpu_type = recipe_config.get("gpu_type")
            n_gpu = recipe_config.get("actor_num_gpus_per_node")
            if gpu_type and n_gpu:
                n_gpu = int(n_gpu)
                return f"{gpu_type}:{n_gpu}" if n_gpu > 1 else str(gpu_type)

    raise TrainingGymConfigError(
        "Could not infer a GPU spec for checkpoint conversion from training run "
        f"{checkpoint.training_run_id!r}. Pass gpu=... explicitly, e.g. "
        'convert_checkpoint_to_hf(checkpoint, model, gpu="H100:8").'
    )


def convert_checkpoint_to_hf(
    checkpoint: Checkpoint,
    model: ModelConfig,
    *,
    gpu: str | None = None,
) -> Checkpoint:
    """Convert a Megatron/torch_dist checkpoint to HuggingFace format on Modal.

    Serving runtimes (and ``modal endpoint create --custom-volume-path``) need an
    HF-format directory with a ``config.json``; slime writes torch_dist
    checkpoints unless HF export is enabled. This runs slime's converter on a GPU
    worker and writes ``<checkpoint>_hf`` next to the input on the same
    checkpoints Volume.

    ``gpu`` defaults to the training run's actor GPU spec.
    """
    if checkpoint.checkpoint_type is CheckpointType.hf:
        return checkpoint

    checkpoints_volume_name = checkpoint.checkpoints_volume_name
    if not checkpoints_volume_name:
        raise TrainingGymConfigError(
            "Cannot convert checkpoint without checkpoints volume metadata."
        )
    checkpoints_mount_path = checkpoint.checkpoints_mount_path or "/checkpoints"

    model_ref = model.model_name or model.model_path
    if not model_ref:
        raise TrainingGymConfigError(
            "Cannot convert a megatron checkpoint without model_name or model_path."
        )

    from modal_training_gym.common import hf_secrets
    from modal_training_gym.frameworks.slime.launcher import _build_slime_base_image

    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        checkpoints_volume_name, create_if_missing=True
    )
    image = _build_slime_base_image().add_local_python_source(
        "modal_training_gym", copy=True
    )
    conversion_app = App("training-gym-checkpoint-convert")

    @conversion_app.function(
        image=image,
        gpu=gpu or _conversion_gpu_spec(checkpoint),
        volumes={
            "/root/.cache/huggingface": hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=4 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="convert_megatron_to_hf",
    )
    def convert_megatron_to_hf(input_dir: str, output_dir: str, model_ref: str) -> str:
        import importlib.util
        import shlex
        import subprocess

        from huggingface_hub import snapshot_download

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        if os.path.isabs(model_ref) and os.path.isdir(model_ref):
            hf_path = model_ref
        else:
            hf_path = snapshot_download(model_ref, local_files_only=True)

        spec = importlib.util.find_spec(
            "modal_training_gym.frameworks.slime.modal_helpers.convert_torch_dist_to_hf"
        )
        convert_script = spec.origin if spec is not None else None
        if not convert_script:
            raise RuntimeError(
                "modal_training_gym.frameworks.slime.modal_helpers."
                "convert_torch_dist_to_hf not found"
            )
        cmd = (
            f"python {convert_script} "
            f"--input-dir {shlex.quote(input_dir)} "
            f"--output-dir {shlex.quote(output_dir)} "
            f"--origin-hf-dir {shlex.quote(hf_path)} "
            f"--force"
        )
        print(f"Converting checkpoint for serving: {cmd}")
        subprocess.run(["bash", "-c", cmd], check=True)
        checkpoints_volume.commit()
        return output_dir

    output_path = f"{checkpoint.path}_hf"
    with modal.enable_output(), conversion_app.run():
        output_path = convert_megatron_to_hf.remote(
            input_dir=checkpoint.path,
            output_dir=output_path,
            model_ref=model_ref,
        )

    return Checkpoint(
        checkpoint_type=CheckpointType.hf,
        name=os.path.basename(output_path),
        path=output_path,
        timestamp=time.time(),
        training_run_id=checkpoint.training_run_id,
        app_name=checkpoint.app_name,
        checkpoints_volume_name=checkpoints_volume_name,
        checkpoints_mount_path=checkpoints_mount_path,
    )
