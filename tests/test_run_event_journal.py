"""The append-only run journal must dominate every cache write race."""

from __future__ import annotations

import asyncio
import copy
import random

import pytest

import modal_training_gym.common.launcher_helpers as launcher_helpers
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.launcher_helpers import (
    init_training_run_record,
    record_setup_failure,
)
from modal_training_gym.common.run import (
    FrameworkStatusUpdate,
    TrainingRun,
    TrainingRunStatus,
    materialize_training_run_summaries,
)
from modal_training_gym.common.run_events import (
    RunEventConflict,
    build_training_run_event,
    materialize_training_run_payload,
)
from modal_training_gym.common.run_reconciler import _parse_running_run
from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items_healed,
)


def _snapshot(
    attempt_count: int,
    attempt_id: str,
    *,
    status: str = "running",
    error_message: str | None = None,
    current: int | None = None,
) -> dict:
    metadata: dict = {
        "attempt_count": attempt_count,
        "active_attempt_id": attempt_id,
        "attempts": [
            {
                "attempt": attempt_count,
                "attempt_id": attempt_id,
                "status": status,
            }
        ],
    }
    if current is not None:
        metadata["framework_progress"] = {
            "phase": SlimeStatus.OPTIMIZER_STEP.value,
            "current": current,
        }
    return {
        "training_run_id": "run-a",
        "framework": Framework.SLIME.value,
        "config": {},
        "status": status,
        "metadata": metadata,
        "error_message": error_message,
        "updated_at": 0,
    }


def _event(
    snapshot: dict,
    kind: str,
    timestamp: int,
) -> dict:
    return build_training_run_event(
        snapshot,
        kind=kind,
        observed_at_ns=timestamp,
    )


def test_delayed_same_attempt_status_cannot_regress_terminal() -> None:
    running = _snapshot(1, "attempt-001", current=99)
    failed = _snapshot(
        1,
        "attempt-001",
        status="failed",
        error_message="Ray worker lost",
    )
    events = [
        _event(running, "started", 1),
        _event(failed, "terminal", 2),
    ]

    materialized = materialize_training_run_payload("run-a", running, events)

    assert materialized["status"] == "failed"
    assert materialized["error_message"] == "Ray worker lost"
    assert "framework_progress" not in materialized["metadata"]


def test_old_finalizer_and_status_cannot_regress_new_attempt() -> None:
    first = _snapshot(1, "attempt-001")
    second = _snapshot(2, "attempt-002", current=7)
    old_failed = _snapshot(
        1,
        "attempt-001",
        status="failed",
        error_message="old failure",
    )
    events = [
        _event(first, "started", 1),
        _event(second, "started", 2),
        _event(old_failed, "terminal", 4),
    ]

    materialized = materialize_training_run_payload("run-a", second, events)

    assert materialized["status"] == "running"
    assert materialized["metadata"]["active_attempt_id"] == "attempt-002"
    assert materialized["metadata"]["framework_progress"]["current"] == 7


def test_materialization_is_independent_of_listing_order() -> None:
    running = _snapshot(1, "attempt-001")
    events = [
        _event(running, "started", 1),
        _event(_snapshot(1, "attempt-001", current=2), "snapshot", 2),
        _event(_snapshot(1, "attempt-001", current=3), "snapshot", 3),
    ]
    expected = materialize_training_run_payload("run-a", None, events)

    for seed in range(20):
        shuffled = copy.deepcopy(events)
        random.Random(seed).shuffle(shuffled)
        assert materialize_training_run_payload("run-a", None, shuffled) == expected


def test_duplicate_identical_event_is_idempotent_but_tampering_fails() -> None:
    running = _snapshot(1, "attempt-001")
    started = _event(running, "started", 1)

    assert (
        materialize_training_run_payload(
            "run-a",
            None,
            [started, copy.deepcopy(started)],
        )["status"]
        == "running"
    )

    tampered = copy.deepcopy(started)
    tampered["payload"]["run_snapshot"]["config"] = {"different": True}
    with pytest.raises(RunEventConflict, match="content hash"):
        materialize_training_run_payload("run-a", None, [started, tampered])


def test_attempt_count_fork_and_conflicting_terminals_fail_closed() -> None:
    first = _snapshot(1, "attempt-001")
    sibling = _snapshot(1, "attempt-sibling")
    with pytest.raises(RunEventConflict, match="forked"):
        materialize_training_run_payload(
            "run-a",
            None,
            [
                _event(first, "started", 1),
                _event(sibling, "started", 2),
            ],
        )

    failed_a = _snapshot(
        1,
        "attempt-001",
        status="failed",
        error_message="cause a",
    )
    failed_b = _snapshot(
        1,
        "attempt-001",
        status="failed",
        error_message="cause b",
    )
    with pytest.raises(RunEventConflict, match="conflicting terminals"):
        materialize_training_run_payload(
            "run-a",
            None,
            [
                _event(first, "started", 1),
                _event(failed_a, "terminal", 2),
                _event(failed_b, "terminal", 3),
            ],
        )


def test_missing_start_and_attempt_count_gap_fail_closed() -> None:
    first = _snapshot(1, "attempt-001")
    third = _snapshot(3, "attempt-003")
    with pytest.raises(RunEventConflict, match="0 start events"):
        materialize_training_run_payload(
            "run-a",
            first,
            [_event(first, "snapshot", 1)],
        )
    with pytest.raises(RunEventConflict, match="not contiguous"):
        materialize_training_run_payload(
            "run-a",
            None,
            [
                _event(first, "started", 1),
                _event(third, "started", 2),
            ],
        )


def test_real_save_chain_materializes_terminal_after_stale_cache_write(
    fake_volume,
) -> None:
    run = TrainingRun(
        training_run_id="run-a",
        framework=Framework.SLIME,
        config={"recipe": {"attempt_mode": "committed"}},
        metadata={
            "attempt_count": 1,
            "active_attempt_id": "attempt-001",
            "attempts": [
                {
                    "attempt": 1,
                    "attempt_id": "attempt-001",
                    "status": "running",
                }
            ],
        },
    )
    run.save(event_kind="started")
    stale = run.model_copy(deep=True)

    run.status = TrainingRunStatus.FAILED
    run.error_message = "primary failure"
    run.metadata["attempts"][0]["status"] = "failed"
    run.save()

    update = FrameworkStatusUpdate(
        training_run_id="run-a",
        phase=SlimeStatus.OPTIMIZER_STEP.value,
        attempt_id="attempt-001",
        progress_current=99,
    )
    assert stale.apply_framework_status(update) == SlimeStatus.OPTIMIZER_STEP
    stale.save_cache()

    materialized = TrainingRun.from_id("run-a")
    assert materialized.status == TrainingRunStatus.FAILED
    assert materialized.error_message == "primary failure"

    # The delayed presentation writer really did regress both mutable caches. The
    # dashboard list path must nevertheless project the terminal journal.
    cached = vol_get_summary_items_healed(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert cached[0]["status"] == TrainingRunStatus.RUNNING.value
    assert _parse_running_run(cached[0]) is None
    summaries = asyncio.run(materialize_training_run_summaries(cached))
    assert summaries[0]["status"] == TrainingRunStatus.FAILED.value
    assert summaries[0]["error_message"] == "primary failure"


def test_d1a_legacy_event_journal_keeps_completed_attempt_after_stale_cache(
    fake_volume,
) -> None:
    attempt_id = "d1a-attempt-001"
    metadata = {
        "attempt_mode": "legacy",
        "event_journal_enabled": True,
        "event_journal_contract": "d1a_legacy_single_attempt_v1",
        "attempt_count": 1,
        "active_attempt_id": attempt_id,
        "attempts": [
            {
                "attempt": 1,
                "attempt_id": attempt_id,
                "status": "running",
            }
        ],
    }
    run = TrainingRun(
        training_run_id="run-a",
        framework=Framework.SLIME,
        config={"recipe": {"attempt_mode": "legacy"}},
        metadata=metadata,
    )
    run.save(event_kind="started")
    stale = run.model_copy(deep=True)

    run.status = TrainingRunStatus.COMPLETED
    run.metadata["attempts"][0]["status"] = "completed"
    run.save()
    stale.save_cache()

    materialized = TrainingRun.from_id("run-a")
    assert materialized.status == TrainingRunStatus.COMPLETED
    assert materialized.metadata["attempt_count"] == 1
    assert materialized.metadata["active_attempt_id"] == attempt_id
    assert materialized.metadata["attempts"] == [
        {
            "attempt": 1,
            "attempt_id": attempt_id,
            "status": "completed",
        }
    ]


def test_framework_progress_uses_cache_without_growing_journal(fake_volume) -> None:
    run = TrainingRun(
        training_run_id="run-a",
        framework=Framework.SLIME,
        config={"recipe": {"attempt_mode": "committed"}},
        metadata={
            "attempt_count": 1,
            "active_attempt_id": "attempt-001",
            "attempts": [
                {
                    "attempt": 1,
                    "attempt_id": "attempt-001",
                    "status": "running",
                }
            ],
        },
    )
    run.save(event_kind="started")
    journal_paths = {
        path for path in fake_volume.files if path.startswith("training-run-events/")
    }

    update = FrameworkStatusUpdate(
        training_run_id="run-a",
        phase=SlimeStatus.OPTIMIZER_STEP.value,
        attempt_id="attempt-001",
        progress_current=9,
    )
    assert run.apply_framework_status(update) == SlimeStatus.OPTIMIZER_STEP
    run.save_cache()

    assert {
        path for path in fake_volume.files if path.startswith("training-run-events/")
    } == journal_paths
    materialized = TrainingRun.from_id("run-a")
    assert materialized.metadata["framework_progress"]["current"] == 9


def test_token_write_failure_after_attempt_start_is_terminalized(
    fake_volume,
    monkeypatch,
) -> None:
    pre_attempt = TrainingRun(
        training_run_id="run-a",
        framework=Framework.SLIME,
        config={"recipe": {"attempt_mode": "committed"}},
        created_at=1,
        started_at=1,
    )
    pre_attempt.save_cache()

    async def _failed_token_write(*_args, **_kwargs):
        raise RuntimeError("token persistence failed")

    monkeypatch.setattr(launcher_helpers, "vol_put", _failed_token_write)
    with pytest.raises(RuntimeError, match="token persistence failed"):
        asyncio.run(
            init_training_run_record(
                training_run_id="run-a",
                modal_app_id="app-1",
                modal_app_url="https://modal.test/app-1",
                framework=Framework.SLIME,
                initializing_status=SlimeStatus.INITIALIZING,
                config_summary={"recipe": {"attempt_mode": "committed"}},
                wandb_cfg=None,
                wandb_entity="",
                framework_status_token="token",
            )
        )

    primary = asyncio.run(
        record_setup_failure(
            "run-a",
            RuntimeError("token persistence failed"),
        )
    )
    materialized = TrainingRun.from_id("run-a")

    assert primary == "RuntimeError: token persistence failed"
    assert materialized.status == TrainingRunStatus.FAILED
    assert materialized.error_message == primary
    assert materialized.metadata["last_attempt_status"] == "failed"
