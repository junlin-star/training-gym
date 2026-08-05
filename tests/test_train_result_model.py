from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from modal_training_gym.common.checkpoint import (
    Checkpoint,
    CheckpointType,
    _list_checkpoints,
    convert_checkpoint_to_hf,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.train_result import TrainResult

MOUNT = "/checkpoints"


def _train_result(**overrides) -> TrainResult:
    kwargs = {
        "app_name": "gym-app",
        "framework": Framework.SLIME,
        "training_run_id": "run-1",
        "checkpoint_dir": f"{MOUNT}/run-1",
        "model_config": ModelConfig(model_name="Qwen/Qwen3-4B"),
    }
    kwargs.update(overrides)
    return TrainResult(**kwargs)


def _checkpoint(name: str, *, training_run_id: str = "run-1") -> Checkpoint:
    return Checkpoint(
        checkpoint_type=(
            CheckpointType.hf if name.endswith("_hf") else CheckpointType.megatron
        ),
        name=name,
        path=f"{MOUNT}/run-1/{name}",
        timestamp=0.0,
        training_run_id=training_run_id,
        app_name="gym-app",
        checkpoints_volume_name="gym-app-checkpoints",
        checkpoints_mount_path=MOUNT,
    )


@pytest.fixture
def listed_checkpoints(monkeypatch: pytest.MonkeyPatch):
    def _install(names: list[str]) -> None:
        monkeypatch.setattr(
            "modal_training_gym.common.checkpoint.list_checkpoints",
            lambda _run_id: [_checkpoint(name) for name in names],
        )

    return _install


@pytest.fixture
def recorded_conversions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str | None]]:
    calls: list[tuple[str, str | None]] = []

    def _convert(checkpoint: Checkpoint, _model, *, gpu: str | None = None):
        calls.append((checkpoint.name, gpu))
        return _checkpoint(f"{checkpoint.name}_hf")

    monkeypatch.setattr(
        "modal_training_gym.common.checkpoint.convert_checkpoint_to_hf", _convert
    )
    return calls


def test_hf_model_converts_latest_megatron_checkpoint(
    listed_checkpoints, recorded_conversions
) -> None:
    listed_checkpoints(["iter_10", "iter_9"])

    model = _train_result(
        checkpoints_volume_name="shared-checkpoints",
        checkpoints_mount_path=MOUNT,
    ).hf_model()

    assert recorded_conversions == [("iter_10", None)]
    assert model.model_path == f"{MOUNT}/run-1/iter_10_hf"
    assert getattr(model, "checkpoints_volume_name") == "shared-checkpoints"
    assert getattr(model, "checkpoints_mount_path") == MOUNT


def test_hf_model_reuses_existing_hf_checkpoint(
    listed_checkpoints, recorded_conversions
) -> None:
    listed_checkpoints(["iter_10_hf", "iter_10"])

    model = _train_result().hf_model()

    assert recorded_conversions == []
    assert model.model_path == f"{MOUNT}/run-1/iter_10_hf"


def test_hf_model_rejects_missing_checkpoints(
    listed_checkpoints, recorded_conversions
) -> None:
    listed_checkpoints([])

    with pytest.raises(TrainingGymConfigError, match="No checkpoints found"):
        _train_result().hf_model()
    assert recorded_conversions == []


def test_hf_model_forwards_explicit_gpu(
    listed_checkpoints, recorded_conversions
) -> None:
    listed_checkpoints(["iter_10"])

    _train_result().hf_model(gpu="H100:8")

    assert recorded_conversions == [("iter_10", "H100:8")]


def test_list_checkpoints_handles_legacy_mount_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "modal_training_gym.common.checkpoint.Volume.from_name",
        lambda *_, **__: pytest.fail("should not open a mismatched Volume path"),
    )

    result = _train_result(
        checkpoint_dir="/legacy-checkpoints/run-1",
        checkpoints_mount_path=None,
    )

    assert _list_checkpoints(result) == []


def _stub_convert_modal(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _FakeVolume:
        @staticmethod
        def from_name(*_args: Any, **_kwargs: Any) -> object:
            return object()

    class _FakeApp:
        def __init__(self, _name: str) -> None:
            return None

        def function(self, **kwargs: Any):
            captured["gpu"] = kwargs.get("gpu")

            def _decorate(fn):
                class _Bound:
                    @staticmethod
                    def remote(**remote_kwargs: Any) -> str:
                        return remote_kwargs["output_dir"]

                return _Bound()

            return _decorate

        def run(self):
            return nullcontext()

    class _FakeImage:
        def add_local_python_source(self, *_args: Any, **_kwargs: Any) -> _FakeImage:
            return self

    monkeypatch.setattr("modal_training_gym.common.checkpoint.Volume", _FakeVolume)
    monkeypatch.setattr("modal_training_gym.common.checkpoint.App", _FakeApp)
    monkeypatch.setattr(
        "modal_training_gym.common.checkpoint.modal.enable_output",
        lambda: nullcontext(),
    )
    monkeypatch.setattr(
        "modal_training_gym.common.hf_secrets",
        lambda: [],
    )
    monkeypatch.setattr(
        "modal_training_gym.frameworks.slime.launcher._build_slime_base_image",
        lambda: _FakeImage(),
    )
    return captured


def test_convert_checkpoint_to_hf_infers_gpu_from_training_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _stub_convert_modal(monkeypatch)
    monkeypatch.setattr(
        "modal_training_gym.common.run.TrainingRun.from_id",
        lambda _run_id: SimpleNamespace(
            config={"recipe": {"gpu_type": "H100", "actor_num_gpus_per_node": 8}}
        ),
    )

    converted = convert_checkpoint_to_hf(
        _checkpoint("iter_10"),
        ModelConfig(model_name="Qwen/Qwen3-4B"),
    )

    assert captured["gpu"] == "H100:8"
    assert converted.checkpoint_type is CheckpointType.hf
    assert converted.path == f"{MOUNT}/run-1/iter_10_hf"


def test_convert_checkpoint_to_hf_errors_actionably_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_convert_modal(monkeypatch)
    monkeypatch.setattr(
        "modal_training_gym.common.run.TrainingRun.from_id",
        lambda _run_id: SimpleNamespace(config={"recipe": {}}),
    )

    with pytest.raises(TrainingGymConfigError) as excinfo:
        convert_checkpoint_to_hf(
            _checkpoint("iter_10"),
            ModelConfig(model_name="Qwen/Qwen3-4B"),
        )

    assert 'gpu="H100:8"' in str(excinfo.value)
