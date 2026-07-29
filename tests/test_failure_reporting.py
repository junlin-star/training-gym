"""Failure provenance must preserve the first/root cause across Modal retries."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

import modal_training_gym.common.launcher_helpers as launcher_helpers
from modal_training_gym.frameworks.slime.launcher import (
    _setup_rank_owns_logical_run_failure,
)
from modal_training_gym.common.launcher_helpers import (
    build_terminal_run_record,
    capture_and_record_ray_failure_diagnostic,
    mark_run_failed,
    record_attempt_failure,
    record_last_committed_boundary_snapshot,
    record_ray_failure_diagnostic,
)
from modal_training_gym.common.run import (
    mark_training_attempt_started,
    record_training_attempt_cluster_identity,
)


def _run(*, metadata=None, error_message=None):
    return SimpleNamespace(metadata=metadata, error_message=error_message)


@pytest.mark.parametrize(
    ("is_head", "owns_failure"),
    [(True, True), (False, False), (None, False)],
)
def test_only_positively_discovered_head_owns_setup_failure(
    is_head,
    owns_failure,
):
    assert _setup_rank_owns_logical_run_failure(is_head) is owns_failure


def test_first_attempt_failure_becomes_primary():
    run = _run(metadata={"attempt_count": 1, "active_attempt_id": "attempt-001"})

    primary = record_attempt_failure(
        run,
        "Ray worker 10.100.0.3 missed five heartbeats",
        recorded_at=100,
    )

    expected = {
        "attempt_id": "attempt-001",
        "attempt_count": 1,
        "message": "Ray worker 10.100.0.3 missed five heartbeats",
        "recorded_at": 100,
    }
    assert primary == expected["message"]
    assert run.error_message == expected["message"]
    assert run.metadata["primary_failure"] == expected
    assert run.metadata["attempt_failures"] == [expected]


def test_retry_failure_does_not_replace_primary_failure():
    run = _run(metadata={"attempt_count": 1, "active_attempt_id": "attempt-001"})
    record_attempt_failure(run, "original Ray node loss", recorded_at=100)

    run.metadata["attempt_count"] = 2
    run.metadata["active_attempt_id"] = "attempt-002"
    primary = record_attempt_failure(
        run,
        "resume_guard marker already exists",
        recorded_at=200,
    )

    assert primary == "original Ray node loss"
    assert run.error_message == "original Ray node loss"
    assert run.metadata["primary_failure"]["message"] == "original Ray node loss"
    assert run.metadata["attempt_failures"] == [
        {
            "attempt_id": "attempt-001",
            "attempt_count": 1,
            "message": "original Ray node loss",
            "recorded_at": 100,
        },
        {
            "attempt_id": "attempt-002",
            "attempt_count": 2,
            "message": "resume_guard marker already exists",
            "recorded_at": 200,
        },
    ]


def test_mark_failed_returns_primary_and_retains_retry_failure():
    run = _run(
        metadata={"attempt_count": 1, "active_attempt_id": "attempt-001"},
    )
    run.status = None
    record_attempt_failure(run, "original Ray node loss", recorded_at=100)
    run.metadata["attempt_count"] = 2
    run.metadata["active_attempt_id"] = "attempt-002"
    run.metadata["attempts"] = [
        {
            "attempt": 1,
            "attempt_id": "attempt-001",
            "status": "failed",
            "ended_at": 100,
        },
        {
            "attempt": 2,
            "attempt_id": "attempt-002",
            "status": "running",
            "started_at": 200,
        },
    ]

    primary = mark_run_failed(run, RuntimeError("secondary resume setup failure"))

    assert primary == "original Ray node loss"
    assert run.error_message == "original Ray node loss"
    assert run.metadata["primary_failure"]["message"] == "original Ray node loss"
    assert run.metadata["attempt_failures"][-1]["attempt_id"] == "attempt-002"
    assert (
        run.metadata["attempt_failures"][-1]["message"]
        == "RuntimeError: secondary resume setup failure"
    )
    assert run.metadata["attempts"][-1]["status"] == "failed"


def test_repeated_report_for_same_attempt_updates_in_place():
    run = _run(metadata={"attempt_count": 3})
    record_attempt_failure(run, "wrapper error", recorded_at=100)

    primary = record_attempt_failure(run, "specific Ray error", recorded_at=101)

    assert primary == "wrapper error"
    assert run.error_message == "wrapper error"
    assert run.metadata["primary_failure"]["message"] == "wrapper error"
    assert run.metadata["attempt_failures"] == [
        {
            "attempt_count": 3,
            "message": "specific Ray error",
            "recorded_at": 101,
        }
    ]


def test_existing_top_level_error_is_migrated_without_misattribution():
    run = _run(
        metadata={"attempt_count": 2, "active_attempt_id": "attempt-002"},
        error_message="failure from legacy attempt 1",
    )

    primary = record_attempt_failure(
        run,
        "secondary retry failure",
        recorded_at=200,
    )

    assert primary == "failure from legacy attempt 1"
    assert run.metadata["primary_failure"] == {
        "message": "failure from legacy attempt 1",
        "recorded_at": 200,
    }
    assert run.metadata["attempt_failures"][0]["attempt_id"] == "attempt-002"


def test_retry_start_migrates_legacy_error_before_clearing_terminal_field():
    run = SimpleNamespace(
        metadata={"attempt_count": 1, "last_attempt_ended_at": 123},
        error_message="legacy node-loss root cause",
        status="failed",
        ended_at=123,
        completed_at=123,
        duration_seconds=23,
    )

    attempt_count = mark_training_attempt_started(run, started_at=200)
    primary = record_attempt_failure(
        run,
        "secondary retry wrapper error",
        recorded_at=201,
    )

    assert attempt_count == 2
    assert primary == "legacy node-loss root cause"
    assert run.metadata["primary_failure"] == {
        "message": "legacy node-loss root cause",
        "recorded_at": 123,
    }
    assert run.error_message == "legacy node-loss root cause"


def test_terminal_refetch_preserves_attempt_failures_and_newer_progress(
    monkeypatch,
):
    latest = SimpleNamespace(
        metadata={"framework_progress": {"phase": "training", "current": 12}},
        status=None,
        error_message=None,
        ended_at=None,
        completed_at=None,
        duration_seconds=None,
        started_at=10,
    )
    local = SimpleNamespace(
        metadata={
            "attempt_count": 2,
            "active_attempt_id": "attempt-002",
            "attempts": [{"attempt_id": "attempt-002", "status": "failed"}],
            "attempt_failures": [{"message": "raylet lost"}],
            "primary_failure": {"message": "raylet lost"},
            "ray_failure_diagnostics": [{"snapshot": {"nodes": []}}],
            "last_committed_boundary": {
                "trained_through_rollout_id": 8,
                "pending_generated_rollout_id": 9,
            },
            # This is intentionally stale and must not replace the re-fetched
            # framework status.
            "framework_progress": {"phase": "initializing"},
        },
        status="failed",
        error_message="raylet lost",
    )

    async def _from_id(_run_id, *, is_async):
        assert is_async is True
        return latest

    monkeypatch.setattr(
        launcher_helpers,
        "TrainingRun",
        SimpleNamespace(from_id=_from_id),
    )

    merged = asyncio.run(build_terminal_run_record(local, "run-a"))

    assert merged.metadata["framework_progress"] == {
        "phase": "training",
        "current": 12,
    }
    assert merged.metadata["attempt_failures"] == [{"message": "raylet lost"}]
    assert merged.metadata["primary_failure"] == {"message": "raylet lost"}
    assert merged.metadata["ray_failure_diagnostics"] == [{"snapshot": {"nodes": []}}]
    assert merged.metadata["last_committed_boundary"] == {
        "trained_through_rollout_id": 8,
        "pending_generated_rollout_id": 9,
    }
    assert merged.metadata["attempts"][0]["status"] == "failed"
    assert merged.error_message == "raylet lost"


def test_terminal_refetch_rejects_stale_attempt_finalizer(monkeypatch):
    latest = SimpleNamespace(
        metadata={
            "active_attempt_id": "attempt-002",
            "framework_progress": {"phase": "optimizer_step", "current": 12},
        },
        status="running",
        error_message=None,
        ended_at=None,
    )
    local = SimpleNamespace(
        metadata={
            "active_attempt_id": "attempt-001",
            "primary_failure": {"message": "old attempt failed"},
        },
        status="failed",
        error_message="old attempt failed",
    )

    async def _from_id(_run_id, *, is_async):
        assert is_async is True
        return latest

    monkeypatch.setattr(
        launcher_helpers,
        "TrainingRun",
        SimpleNamespace(from_id=_from_id),
    )

    merged = asyncio.run(build_terminal_run_record(local, "run-a"))

    assert merged is latest
    assert merged.status == "running"
    assert merged.error_message is None
    assert merged.ended_at is None
    assert merged.metadata["active_attempt_id"] == "attempt-002"


def test_attempt_cluster_identity_is_write_once_and_preserves_prior_attempts():
    prior_attempt = {
        "attempt": 1,
        "attempt_id": "attempt-001",
        "status": "failed",
        "modal_cluster": {"cluster_id": "old-cluster"},
    }
    run = _run(
        metadata={
            "attempt_count": 2,
            "active_attempt_id": "attempt-002",
            "attempts": [
                prior_attempt,
                {
                    "attempt": 2,
                    "attempt_id": "attempt-002",
                    "status": "running",
                },
            ],
        }
    )
    identity = {
        "schema_version": 1,
        "cluster_id": "new-cluster",
        "node_count": 2,
        "rank_ordered_container_ipv4_ips": ["10.0.0.1", "10.0.0.2"],
        "head_addr": "10.0.0.1",
    }

    record_training_attempt_cluster_identity(run, identity)
    saved_metadata = copy.deepcopy(run.metadata)
    record_training_attempt_cluster_identity(run, copy.deepcopy(identity))

    assert run.metadata == saved_metadata
    assert run.metadata["attempts"][0] == prior_attempt
    assert run.metadata["attempts"][1]["modal_cluster"] == identity

    identity["rank_ordered_container_ipv4_ips"].append("mutated")
    assert run.metadata["attempts"][1]["modal_cluster"][
        "rank_ordered_container_ipv4_ips"
    ] == ["10.0.0.1", "10.0.0.2"]

    conflicting_identity = {
        **saved_metadata["attempts"][1]["modal_cluster"],
        "cluster_id": "conflicting-cluster",
    }
    with pytest.raises(RuntimeError, match="immutable"):
        record_training_attempt_cluster_identity(run, conflicting_identity)
    assert run.metadata == saved_metadata


def test_fallback_ray_diagnostic_records_post_start_failure_context():
    run = _run(
        metadata={"attempt_count": 2, "active_attempt_id": "attempt-002"},
    )

    recorded = capture_and_record_ray_failure_diagnostic(
        run,
        lambda: {"nodes": [{"NodeID": "worker", "Alive": False}]},
        attempt_id="attempt-002",
        attempt_count=2,
        ray_job_id="ray-job-7",
        ray_job_status="LAUNCHER_EXCEPTION",
        failure_stage="ray_job_submission_or_streaming",
    )

    assert recorded is True
    assert run.metadata["ray_failure_diagnostics"] == [
        {
            "attempt_id": "attempt-002",
            "attempt_count": 2,
            "ray_job_id": "ray-job-7",
            "ray_job_status": "LAUNCHER_EXCEPTION",
            "failure_stage": "ray_job_submission_or_streaming",
            "snapshot": {"nodes": [{"NodeID": "worker", "Alive": False}]},
        }
    ]


def test_fallback_does_not_duplicate_normal_failed_result_snapshot():
    run = _run(
        metadata={"attempt_count": 1, "active_attempt_id": "attempt-001"},
    )
    assert record_ray_failure_diagnostic(
        run,
        {"ray_status": "worker lost"},
        attempt_id="attempt-001",
        attempt_count=1,
        ray_job_id="ray-job-1",
        ray_job_status="FAILED",
        failure_stage="ray_job_terminal_result",
    )
    capture_calls = 0

    def _capture():
        nonlocal capture_calls
        capture_calls += 1
        return {"ray_status": "duplicate"}

    recorded = capture_and_record_ray_failure_diagnostic(
        run,
        _capture,
        attempt_id="attempt-001",
        attempt_count=1,
        ray_job_id="ray-job-1",
        ray_job_status="LAUNCHER_EXCEPTION",
        failure_stage="ray_job_terminal_result",
    )

    assert recorded is False
    assert capture_calls == 0
    assert len(run.metadata["ray_failure_diagnostics"]) == 1
    assert run.metadata["ray_failure_diagnostics"][0]["snapshot"] == {
        "ray_status": "worker lost"
    }


def test_fallback_diagnostic_capture_error_does_not_mask_root_failure(capsys):
    run = _run(
        metadata={"attempt_count": 1, "active_attempt_id": "attempt-001"},
    )

    def _capture():
        raise RuntimeError("diagnostic endpoint unavailable")

    recorded = capture_and_record_ray_failure_diagnostic(
        run,
        _capture,
        attempt_id="attempt-001",
        attempt_count=1,
        ray_job_status="LAUNCHER_EXCEPTION",
        failure_stage="ray_dashboard_setup",
    )

    assert recorded is False
    assert "ray_failure_diagnostics" not in run.metadata
    assert "diagnostic endpoint unavailable" in capsys.readouterr().out


def test_ray_failure_diagnostics_are_bounded_to_eight_attempts():
    run = _run(
        metadata={
            "attempt_count": 9,
            "active_attempt_id": "attempt-009",
            "ray_failure_diagnostics": [
                {
                    "attempt_id": f"attempt-{index:03d}",
                    "snapshot": {"index": index},
                }
                for index in range(1, 9)
            ],
        },
    )

    assert record_ray_failure_diagnostic(
        run,
        {"index": 9},
        attempt_id="attempt-009",
        attempt_count=9,
        ray_job_status="LAUNCHER_EXCEPTION",
        failure_stage="post_ray_start_setup",
    )

    diagnostics = run.metadata["ray_failure_diagnostics"]
    assert len(diagnostics) == 8
    assert diagnostics[0]["attempt_id"] == "attempt-002"
    assert diagnostics[-1]["attempt_id"] == "attempt-009"


def test_last_committed_boundary_separates_trained_and_generated_progress():
    run = _run(
        metadata={"active_attempt_id": "attempt-002"},
    )

    snapshot = record_last_committed_boundary_snapshot(
        run,
        {
            "attempt_id": "attempt-001",
            "rollout_id": 8,
            "checkpoint_iteration": 8,
            "scientific_commit_id": "boundary-sha256:abc",
            "parent_commit_id": "boundary-sha256:parent",
            "boundary_sha256": "manifest-digest",
            "pending_rollout": {"rollout_id": 9, "path": "pending/9.pkl"},
            "terminal": False,
            # This large field must not leak into the bounded terminal summary.
            "inventory": [{"path": f"file-{index}"} for index in range(100)],
        },
        captured_at=123,
    )

    assert snapshot == {
        "schema_version": 1,
        "captured_at": 123,
        "metadata_only": True,
        "found": True,
        "active_attempt_id": "attempt-002",
        "committed_attempt_id": "attempt-001",
        "scientific_commit_id": "boundary-sha256:abc",
        "parent_commit_id": "boundary-sha256:parent",
        "trained_through_rollout_id": 8,
        "checkpoint_iteration": 8,
        "pending_generated_rollout_id": 9,
        "terminal": False,
        "boundary_sha256": "manifest-digest",
    }
    assert run.metadata["last_committed_boundary"] == snapshot


def test_last_committed_boundary_explicitly_records_no_durable_progress():
    run = _run(metadata={"active_attempt_id": "attempt-001"})

    snapshot = record_last_committed_boundary_snapshot(
        run,
        None,
        captured_at=321,
    )

    assert snapshot["found"] is False
    assert snapshot["active_attempt_id"] == "attempt-001"
    assert snapshot["trained_through_rollout_id"] is None
    assert snapshot["pending_generated_rollout_id"] is None
    assert snapshot["scientific_commit_id"] is None
