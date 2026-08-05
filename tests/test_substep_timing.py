from __future__ import annotations

from queue import Queue

from modal_training_gym.common.step_timing import phase_totals, rollout_lanes
from modal_training_gym.frameworks.slime import reporting
from modal_training_gym.frameworks.slime.timing import recording_lane


def test_phase_totals():
    totals = phase_totals([[0.0, 1.0], [2.0, 4.0]])
    assert totals == {
        "total_duration_s": 3.0,
        "wall_span_s": 4.0,
        "count": 2,
        "start_offset_s": 0.0,
    }


def test_phase_totals_empty():
    totals = phase_totals([])
    assert totals["count"] == 0
    assert totals["total_duration_s"] == 0.0


def test_rollout_lanes():
    records = [
        {
            "role": "driver",
            "lane_start_unix_s": 1.0,
            "phases": {"generate_rollouts": [[0.0, 1.0]]},
        }
    ]
    lanes = rollout_lanes(records)
    assert "driver" in lanes["roles"]
    assert lanes["roles"]["driver"]["totals"]["generate_rollouts"]["count"] == 1


def test_role_recorder():
    with recording_lane("rollout", 0) as rec:
        with rec.phase("custom_reward"):
            pass
    assert "custom_reward" in rec.phases
    assert len(rec.phases["custom_reward"]) == 1


def test_full_queue_drops_snapshot_and_its_pending_payload(monkeypatch):
    """A full timing queue must drop the snapshot, then recover on the next publish."""
    monkeypatch.setattr(reporting, "_TIMING_WORKER_STARTED", True)
    monkeypatch.setattr(reporting, "_TIMING_QUEUE", Queue(maxsize=1))
    monkeypatch.setattr(reporting, "_timing_url", lambda: "http://test/timing")
    monkeypatch.setattr(reporting, "_env_training_run_id", lambda: "run-1")

    # Fill the queue so the next put fails.
    reporting._TIMING_QUEUE.put_nowait("dummy")

    # First publish: queue full -> the pending entry should be removed, not left poisoned.
    reporting._enqueue_timing({"rollout_id": 0, "role": "rollout"})
    assert reporting._TIMING_PENDING == {}

    # Make room, then publish again; the lane should enqueue normally.
    reporting._TIMING_QUEUE.get()
    reporting._enqueue_timing({"rollout_id": 0, "role": "rollout"})
    assert reporting._TIMING_QUEUE.qsize() == 1
    assert (0, "rollout") in reporting._TIMING_PENDING


def test_legacy_record_with_stray_dropped_key_loads():
    from modal_training_gym.common.step_timing import RoleTimingRecord

    record = RoleTimingRecord(
        training_run_id="run-1",
        rollout_id=0,
        role="driver",
        phases={},
        dropped={"x": 1},  # type: ignore[call-arg]
    )
    assert record.phases == {}
