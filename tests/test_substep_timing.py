"""Recorder aggregates, snapshot posting, and the legacy read path."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from modal_training_gym.common import status_reporter, step_timing
from modal_training_gym.common.step_timing import (
    PhaseTiming,
    RoleTimingRecord,
    legacy_run_to_records,
    probe_substep_timing,
    rollout_lanes,
)
from modal_training_gym.common.timing_recorder import (
    MAX_DRAWN_INVOCATIONS,
    recording_lane,
    time_phase,
)


@pytest.fixture
def timing_env(monkeypatch):
    """Point the recorder at a dashboard so it publishes, without posting."""
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "http://test/")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "run-1")
    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "auto")
    posted: list[dict] = []
    monkeypatch.setattr(status_reporter, "post_item", posted.append)
    return posted


def test_repeated_phase_aggregates_every_run(timing_env):
    with recording_lane("rollout", 0) as lane:
        for _ in range(10):
            with lane.phase("reward"):
                time.sleep(0.002)

    reward = PhaseTiming(**lane.phases["reward"])
    assert reward.count == 10
    assert reward.total_duration_s >= 0.02
    assert reward.longest_duration_s >= reward.average_duration_s
    assert reward.total_duration_s == pytest.approx(
        reward.average_duration_s * 10, rel=1e-9
    )
    assert 0.0 <= reward.first_start_s < reward.last_end_s


def test_each_run_is_kept_until_there_are_too_many_to_draw(timing_env):
    """The timeline draws runs, so alternating phases must not look nested."""
    with recording_lane("actor", 0) as lane:
        for _ in range(2):
            with lane.phase("forward_backward"):
                time.sleep(0.001)
            with lane.phase("optimizer_step"):
                time.sleep(0.001)

    forward, optimizer = (
        lane.invocations["forward_backward"],
        lane.invocations["optimizer_step"],
    )
    assert len(forward) == len(optimizer) == 2
    assert forward[0][1] <= optimizer[0][0] <= forward[1][0] <= optimizer[1][0]

    with recording_lane("rollout", 0) as sampled:
        for _ in range(MAX_DRAWN_INVOCATIONS + 1):
            with sampled.phase("reward"):
                pass

    # Emptied rather than truncated: a prefix would draw as if the phase had
    # stopped early. The aggregate still counts every run.
    assert sampled.invocations["reward"] == []
    assert sampled.phases["reward"]["count"] == MAX_DRAWN_INVOCATIONS + 1


def test_concurrent_phase_spans_less_than_it_sums(timing_env):
    """Rewards are scored in parallel: time spent exceeds wall-clock span."""
    with recording_lane("rollout", 0) as lane:
        threads = [
            threading.Thread(target=lambda: _sleep_in_phase(lane, 0.05))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    reward = PhaseTiming(**lane.phases["reward"])
    span = reward.last_end_s - reward.first_start_s
    assert reward.count == 8
    assert reward.total_duration_s >= 0.4
    assert span < reward.total_duration_s


def _sleep_in_phase(lane, seconds: float) -> None:
    with lane.phase("reward"):
        time.sleep(seconds)


def test_a_reward_scored_on_the_frameworks_loop_thread_lands_on_the_lane(timing_env):
    """The reward lane is non-empty only because the loop inherits the context.

    slime and miles score samples with ``asyncio.run_coroutine_threadsafe`` on a
    background loop thread that was started before the lane existed; the phase
    finds the lane because that call copies the *submitting* thread's context.
    """
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    async def score() -> None:
        with time_phase("reward"):
            await asyncio.sleep(0.001)

    try:
        with recording_lane("rollout", 0) as lane:
            scored = [asyncio.run_coroutine_threadsafe(score(), loop) for _ in range(4)]
            for future in scored:
                future.result(timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()

    assert lane.phases["reward"]["count"] == 4


def test_time_phase_records_on_the_active_lane_and_is_a_noop_without_one(timing_env):
    with recording_lane("actor", 3) as lane:
        with time_phase("forward_backward"):
            pass
    assert lane.phases["forward_backward"]["count"] == 1

    with time_phase("forward_backward"):  # no lane: must not raise
        pass


def test_published_record_validates_and_reduces_to_lanes(timing_env):
    with recording_lane("driver", 2) as lane:
        with lane.phase("train_models"):
            time.sleep(0.001)

    item = timing_env[-1]
    record = RoleTimingRecord(
        **{key: value for key, value in item.items() if not key.startswith("_")}
    )
    assert record.rollout_id == 2 and record.role.value == "driver"
    assert record.phases["train_models"].count == 1

    lanes = rollout_lanes([record.model_dump(mode="json")])
    assert set(lanes["roles"]) == {"driver"}
    assert lanes["roles"]["driver"]["phases"]["train_models"]["count"] == 1


def test_timing_off_publishes_nothing(timing_env, monkeypatch):
    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "off")
    with recording_lane("driver", 0) as lane:
        with lane.phase("train_models"):
            pass
    assert timing_env == []


def test_lane_exit_publishes_even_inside_the_rate_limit(timing_env, monkeypatch):
    monkeypatch.setattr(
        "modal_training_gym.common.timing_recorder.MIN_PUBLISH_INTERVAL_S", 3600.0
    )
    with recording_lane("driver", 0) as lane:
        for _ in range(5):
            with lane.phase("train_models"):
                pass
    lane._poster.join(timeout=5)
    # The four measurements after the first are rate-limited away, and lane exit
    # forces the final state out even though the limit still holds.
    assert timing_env[-1]["phases"]["train_models"]["count"] == 5


def test_snapshots_taken_while_one_is_in_flight_are_superseded(timing_env, monkeypatch):
    """A snapshot replaces the lane's stored record, so the ones measured while
    the dashboard was slow are skipped rather than posted in turn."""
    monkeypatch.setattr(
        "modal_training_gym.common.timing_recorder.MIN_PUBLISH_INTERVAL_S", 0.0
    )
    in_flight, finish_post = threading.Event(), threading.Event()

    def slow_post(item: dict) -> None:
        timing_env.append(item)
        in_flight.set()
        finish_post.wait(5)

    monkeypatch.setattr(status_reporter, "post_item", slow_post)

    with recording_lane("rollout", 0) as lane:
        with lane.phase("reward"):
            pass
        assert in_flight.wait(5)
        for _ in range(5):
            with lane.phase("reward"):
                pass
        finish_post.set()
    lane._poster.join(timeout=5)

    counts = [item["phases"]["reward"]["count"] for item in timing_env]
    assert counts[0] == 1 and counts[-1] == 6
    assert not [count for count in counts if 1 < count < 6]


def test_a_lane_whose_ranks_never_form_writes_itself(timing_env):
    """One process measuring an actor alone is the one that publishes it."""
    with recording_lane("actor", 0, publish_gate=lambda: None) as lane:
        with lane.phase("forward_backward"):
            pass
    assert [item["role"] for item in timing_env] == ["actor"]


# ---------- capability probe ----------


def test_probe_requires_a_dashboard_url_when_required():
    with pytest.raises(RuntimeError, match="substep_timing='require'"):
        probe_substep_timing("", mode="require")


def test_probe_is_off_or_silent_without_a_dashboard():
    assert probe_substep_timing("http://test/", mode="off") is False
    assert probe_substep_timing("", mode="auto") is False


def test_a_step_adds_its_substeps_and_leaves_out_the_checkpoint_and_eval(
    monkeypatch,
):
    def one_phase(name, start, duration):
        return {
            name: {
                "count": 1,
                "total_duration_s": duration,
                "longest_duration_s": duration,
                "first_start_s": start,
                "last_end_s": start + duration,
            }
        }

    monkeypatch.setattr(
        step_timing,
        "load_run",
        lambda _: {
            0: [
                {
                    "role": "driver",
                    "lane_start_unix_s": 1000.0,
                    "phases": {
                        # A 60s eval before the step, and a 5s checkpoint in the
                        # middle of its 15s of work.
                        **one_phase("evaluate_rollouts", 0.0, 60.0),
                        **one_phase("generate_rollouts", 60.0, 8.0),
                        **one_phase("checkpoint_save", 68.0, 5.0),
                        **one_phase("weight_sync", 73.0, 7.0),
                    },
                },
                {
                    "role": "rollout",
                    "lane_start_unix_s": 1060.0,
                    # The driver's generate_rollouts was the wait for this.
                    "phases": one_phase("generate_samples", 0.0, 8.0),
                },
            ]
        },
    )

    # Rollout 0 is step 1, the way a stored baseline numbers it.
    step_times, substep_times = step_timing.measured_run_times("run-1")
    assert step_times == {"1": {"duration_s": 15}}
    assert substep_times["1"]["generate_samples (rollout)"]["duration_s"] == 8.0
    assert substep_times["1"]["evaluate_rollouts"] == {
        "start": 1000.0,
        "duration_s": 60.0,
    }


# ---------- legacy read path ----------


def test_legacy_run_converts_to_zero_based_driver_lanes():
    records = legacy_run_to_records(
        {
            "1": {
                "generate_rollouts": {"start": 100.0, "duration_s": 2.0},
                "optimizer_step": {"start": 102.0, "duration_s": 3.0},
                "weight_sync": {"start": None, "duration_s": None},
            }
        }
    )
    assert len(records) == 1
    record = records[0]
    assert record["rollout_id"] == 0
    assert record["role"] == "driver"
    assert record["lane_start_unix_s"] == 100.0
    # Legacy "optimizer_step" bracketed the whole train call.
    assert set(record["phases"]) == {"generate_rollouts", "train_models"}
    assert record["phases"]["train_models"] == {
        "count": 1,
        "total_duration_s": 3.0,
        "longest_duration_s": 3.0,
        "first_start_s": 2.0,
        "last_end_s": 5.0,
    }
    # Same shape as a measured record, so the frontend has one renderer.
    RoleTimingRecord(training_run_id="run-1", **record)


def test_legacy_run_without_substep_times_yields_nothing():
    assert legacy_run_to_records(None) == []
