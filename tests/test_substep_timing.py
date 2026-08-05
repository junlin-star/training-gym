from __future__ import annotations

import time
from queue import Queue


from modal_training_gym.common.step_timing import phase_totals
from modal_training_gym.frameworks.slime import reporting
from modal_training_gym.frameworks.slime.timing import (
    recording_lane,
)


def test_overflow_keeps_true_total_and_count(monkeypatch):
    monkeypatch.setattr(
        "modal_training_gym.frameworks.slime.timing.MAX_INTERVALS_PER_PHASE", 4
    )

    with recording_lane("rollout", 0) as rec:
        for _ in range(10):
            with rec.phase("reward"):
                time.sleep(0.001)

    assert len(rec.phases["reward"]) == 4
    overflow = rec.overflow["reward"]
    assert overflow[0] == 6
    # total of all ten 1ms sleeps (allow margin)
    assert overflow[1] >= 0.006
    # latest end is from the last (10th) measurement
    assert overflow[2] > rec.phases["reward"][-1][1]

    totals = phase_totals(rec.phases["reward"], overflow)
    assert totals["count"] == 10
    assert totals["total_duration_s"] >= 0.01
    # wall span reaches the last measurement
    assert (
        abs(totals["wall_span_s"] - (overflow[2] - rec.phases["reward"][0][0])) < 1e-9
    )


def test_phase_totals_without_overflow():
    intervals = [[0.0, 1.0], [2.0, 4.0]]
    totals = phase_totals(intervals)
    assert totals == {
        "total_duration_s": 3.0,
        "wall_span_s": 4.0,
        "count": 2,
        "start_offset_s": 0.0,
    }


def test_phase_totals_with_overflow():
    intervals = [[0.0, 1.0], [2.0, 3.0]]
    totals = phase_totals(intervals, (30, 6.0, 4.0))
    assert totals["total_duration_s"] == 8.0
    assert totals["wall_span_s"] == 4.0
    assert totals["count"] == 32
    assert totals["start_offset_s"] == 0.0


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
    assert record.overflow == {}
