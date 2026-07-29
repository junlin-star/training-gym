from __future__ import annotations

import time

import pytest

from modal_training_gym.cli.cleanup import cleanup
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus


def test_cleanup_removes_committed_run_journal_and_prevents_resurrection(
    fake_volume,
) -> None:
    old = int(time.time()) - 10 * 86400
    run = TrainingRun(
        training_run_id="old-failed-run",
        framework=Framework.SLIME,
        config={"recipe": {"attempt_mode": "committed"}},
        created_at=old,
        started_at=old,
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
    run.status = TrainingRunStatus.FAILED
    run.error_message = "worker lost"
    run.metadata["attempts"][0]["status"] = "failed"
    run.save(event_kind="terminal")
    assert any(path.startswith("training-run-events/") for path in fake_volume.files)

    cleanup(older_than_days=7)

    assert not any(
        path.startswith("training-run-events/") for path in fake_volume.files
    )
    with pytest.raises(KeyError):
        TrainingRun.from_id("old-failed-run")
