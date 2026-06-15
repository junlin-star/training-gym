"""Save-path regression tests for the metadata layer.

Two bugs reached `main` because nothing exercised a run all the way to
``save()`` without a full GPU train: a ``Framework`` enum that wasn't
JSON-serializable, and ``Volume.reload()`` crashing when the volume isn't
attached (local driver / a function that doesn't mount it). These tests drive
the real ``save()`` chain against an in-memory volume — no Modal, no GPU — so
that path is covered in CI.
"""

from __future__ import annotations

import io
import json
import os

import pytest

from modal_training_gym.common import run as run_mod
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore


class _FakeVolume:
    """In-memory stand-in for a Modal Volume that is *not* attached.

    ``reload()`` raises like a real unattached/local volume; reads and writes
    operate on an in-memory dict. A correct metadata layer must still complete a
    ``save()`` against this — reload is only a freshness hint.
    """

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def reload(self) -> None:
        raise RuntimeError("reload() can only be called from within a running function")

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]

    def remove_file(self, path: str) -> None:
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]

    def batch_upload(self):
        files = self.files

        class _Batch:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def put_file(self, fileobj: io.BytesIO, path: str) -> None:
                files[path] = fileobj.read()

        return _Batch()


@pytest.fixture
def fake_volume(monkeypatch) -> _FakeVolume:
    vol = _FakeVolume()
    monkeypatch.setattr(metadata, "_metadata_volume", lambda: vol)
    return vol


def test_training_run_save_survives_unmounted_volume(fake_volume):
    """The local-driver path that crashed: TrainingRun.save() must complete even
    when reload() raises, and persist a valid JSON payload."""
    run_mod.TrainingRun(
        training_run_id="t1", framework=Framework.SLIME, config={}
    ).save()

    blob = fake_volume.files[f"{MetadataStore.TRAINING_RUNS.value}/t1.json"]
    assert json.loads(blob)["framework"] == "slime"


def test_train_result_save_survives_unmounted_volume(fake_volume):
    """TrainResult.save() is the path that hit both the reload crash and the
    non-serializable Framework enum (asdict keeps the enum)."""
    TrainResult(app_name="a", framework=Framework.MILES, training_run_id="t2").save()

    blob = fake_volume.files[f"{MetadataStore.TRAIN_RESULTS.value}/t2.json"]
    assert json.loads(blob)["framework"] == "miles"


@pytest.mark.parametrize("fw", list(Framework))
def test_train_result_payload_is_json_serializable(fw):
    """Every Framework must round-trip through plain json.dumps — guards against
    regressing the (str, Enum) base back to a bare Enum."""
    payload = TrainResult(app_name="a", framework=fw, training_run_id="t")._to_dict()
    assert json.loads(json.dumps(payload))["framework"] == fw.value


@pytest.mark.skipif(
    os.environ.get("RUN_MODAL_TESTS") != "1",
    reason="hits Modal (no GPU); opt in with RUN_MODAL_TESTS=1",
)
def test_remote_save_from_unmounted_container():
    """The faithful remote counterpart: run save() inside a real Modal container
    that does *not* mount the metadata volume — the exact context where the
    original training run crashed with `volume … not attached`. The fake-volume
    tests simulate that; this proves it against real Modal Volume semantics
    (reload unavailable, but the client-side write still lands).
    """
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("modal>=1.4.0", "pydantic")
        .add_local_python_source("modal_training_gym")
    )
    app = modal.App("training-gym-metadata-save-probe")

    # NB: deliberately no volumes= — this is the unmounted case.
    @app.function(image=image, serialized=True)
    def _save_probe() -> str:
        from modal_training_gym.common.framework import Framework
        from modal_training_gym.common.run import TrainingRun
        from modal_training_gym.common.train_result import TrainResult

        rid = "ci-remote-save-probe"  # fixed id → overwrites, no junk accrual
        TrainingRun(training_run_id=rid, framework=Framework.SLIME, config={}).save()
        TrainResult(app_name=rid, framework=Framework.SLIME, training_run_id=rid).save()
        return "ok"

    with modal.enable_output():
        with app.run():
            assert _save_probe.remote() == "ok"
