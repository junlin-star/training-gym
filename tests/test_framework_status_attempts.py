from __future__ import annotations

import copy
from types import SimpleNamespace

import modal_training_gym.common.run as run_mod
from modal_training_gym.common.framework import Framework
from modal_training_gym.common import status_reporter
from modal_training_gym.common.run import (
    FrameworkStatusUpdate,
    TrainingRun,
    TrainingRunStatus,
)
from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.frameworks.slime.reporting import _run_context


def _run(*, active_attempt_id: str = "", status=TrainingRunStatus.RUNNING):
    metadata = {"active_attempt_id": active_attempt_id} if active_attempt_id else {}
    return TrainingRun(
        training_run_id="run-a",
        framework=Framework.SLIME,
        config={},
        status=status,
        metadata=metadata,
    )


def _update(*, attempt_id: str | None):
    return FrameworkStatusUpdate(
        training_run_id="run-a",
        phase=SlimeStatus.OPTIMIZER_STEP.value,
        attempt_id=attempt_id,
        progress_current=4,
    )


def test_matching_attempt_can_update_progress():
    run = _run(active_attempt_id="attempt-002")

    applied = run.apply_framework_status(_update(attempt_id="attempt-002"))

    assert applied == SlimeStatus.OPTIMIZER_STEP
    assert run.metadata["framework_progress"]["current"] == 4


def test_stale_or_unversioned_attempt_cannot_update_progress():
    run = _run(active_attempt_id="attempt-002")
    before = copy.deepcopy(run.model_dump())

    assert (
        run.framework_status_rejection(_update(attempt_id="attempt-001"))
        == "attempt_mismatch"
    )
    assert run.apply_framework_status(_update(attempt_id="attempt-001")) is None
    assert run.model_dump() == before
    assert run.apply_framework_status(_update(attempt_id=None)) is None
    assert run.model_dump() == before


def test_terminal_run_rejects_even_matching_attempt_status():
    run = _run(
        active_attempt_id="attempt-002",
        status=TrainingRunStatus.COMPLETED,
    )
    before = copy.deepcopy(run.model_dump())

    assert (
        run.framework_status_rejection(_update(attempt_id="attempt-002"))
        == "terminal_run"
    )
    assert run.apply_framework_status(_update(attempt_id="attempt-002")) is None
    assert run.model_dump() == before


def test_step_timing_side_effect_is_attempt_namespaced(monkeypatch):
    timings = {}
    monkeypatch.setattr(run_mod, "_step_times_dict", lambda: timings)
    run = _run(active_attempt_id="attempt-002")
    update = _update(attempt_id="attempt-002")
    update.step_event = "start"

    assert run.apply_framework_status(update) == SlimeStatus.OPTIMIZER_STEP
    assert set(timings) == {"run-a:attempt-002:4:start"}


def test_pre_attempt_orchestration_status_remains_compatible():
    run = _run()

    assert run.framework_status_rejection(_update(attempt_id=None)) is None
    applied = run.apply_framework_status(_update(attempt_id=None))

    assert applied == SlimeStatus.OPTIMIZER_STEP


def test_unsupported_phase_has_distinct_rejection_reason():
    run = _run()
    update = FrameworkStatusUpdate(
        training_run_id="run-a",
        phase="not-a-slime-phase",
    )

    assert run.framework_status_rejection(update) == "unsupported_phase"
    assert run.apply_framework_status(update) is None


def test_common_reporter_attaches_ambient_attempt_id(monkeypatch):
    captured = []
    monkeypatch.setenv("TRAINING_GYM_ATTEMPT_ID", "attempt-002")
    monkeypatch.setattr(status_reporter, "_ensure_worker", lambda: None)
    monkeypatch.setattr(
        status_reporter,
        "_QUEUE",
        SimpleNamespace(put_nowait=captured.append),
    )

    status_reporter.enqueue_framework_status(
        "run-a",
        SlimeStatus.OPTIMIZER_STEP.value,
        url="https://dashboard.invalid/api/framework-status",
    )

    assert captured[0]["attempt_id"] == "attempt-002"


def test_slime_worker_context_attaches_attempt_id(monkeypatch):
    monkeypatch.setenv("TRAINING_GYM_ATTEMPT_ID", "attempt-002")

    context = _run_context(SimpleNamespace(training_run_id="run-a"))

    assert context["attempt_id"] == "attempt-002"
