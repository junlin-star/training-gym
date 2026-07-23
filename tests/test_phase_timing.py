import asyncio
import inspect
import sys
import threading
from queue import Queue
from types import ModuleType, SimpleNamespace

import pytest

from modal_training_gym.common.async_timing_types import (
    AsyncTimingEvent,
    AsyncTimingEventType,
)
from modal_training_gym.common.status import SlimeStatus, resolve_framework_status
from modal_training_gym.common.timing_types import TrainingSubstep
from modal_training_gym.frameworks.slime import phase_reporting, reporting
from modal_training_gym.utils import metadata


class _Optimizer:
    def step(self):
        return "updated"

    def zero_grad(self):
        return "cleared"


class _TimingStore(dict[object, AsyncTimingEvent]):
    def put(
        self,
        key: object,
        value: AsyncTimingEvent,
        *,
        skip_if_exists: bool = False,
    ) -> bool:
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def _timing_event(
    *,
    rollout_id: int = 0,
    phase: str = "training",
    event_type: AsyncTimingEventType = "phase_start",
    timestamp: float = 1.0,
    training_attempt: int = 1,
    occurrence_id: int | None = None,
    role: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> AsyncTimingEvent:
    return {
        "training_run_id": "run-1",
        "training_attempt": training_attempt,
        "rollout_id": rollout_id,
        "phase": phase,
        "event_type": event_type,
        "timestamp": timestamp,
        "monotonic_timestamp": timestamp,
        "occurrence_id": occurrence_id,
        "role": role,
        "rank": rank,
        "world_size": world_size,
        "timeline_lane": None,
        "parent_phase": None,
        "display_name": None,
    }


@pytest.fixture(autouse=True)
def _reset_timing_delivery_counters(monkeypatch):
    monkeypatch.setattr(reporting, "_TIMING_FAILED_EVENTS", 0)


def test_training_substeps_only_reuse_existing_framework_statuses():
    framework_status_substeps = {
        TrainingSubstep.POLICY_LOG_PROBS,
        TrainingSubstep.OPTIMIZER_STEP,
    }
    for substep in TrainingSubstep:
        resolved = resolve_framework_status(substep.value, "slime")
        if substep in framework_status_substeps:
            assert resolved is not None
        else:
            assert resolved is None


def test_record_async_phase_interval_finishes_when_wrapped_call_raises(monkeypatch):
    reports = []
    args = SimpleNamespace(async_mode=True)
    wall_times = iter((10.0, 12.0))
    monotonic_times = iter((100.0, 101.5))
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(phase_reporting.time, "time", lambda: next(wall_times))
    monkeypatch.setattr(
        phase_reporting.time, "monotonic", lambda: next(monotonic_times)
    )

    with pytest.raises(RuntimeError, match="failed"):
        with phase_reporting.record_async_phase_interval(
            "data_preprocess",
            args,
            2,
            timeline_lane="training",
            parent_phase="training",
            display_name="Load & transfer training batch",
        ):
            raise RuntimeError("failed")

    assert [report[0][3] for report in reports] == ["phase_start", "phase_finish"]
    assert [report[1]["timestamp"] for report in reports] == [10.0, 12.0]
    assert [report[1]["monotonic_timestamp"] for report in reports] == [100.0, 101.5]
    for _, report in reports:
        assert report["timeline_lane"] == "training"
        assert report["parent_phase"] == "training"
        assert report["display_name"] == "Load & transfer training batch"


def test_async_rollout_logging_does_not_duplicate_driver_timing(monkeypatch):
    reports = []
    args = SimpleNamespace(num_rollout=3, async_mode=True)
    monkeypatch.setattr(
        phase_reporting,
        "report_phase",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(
        phase_reporting, "report_rollout_samples", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)

    phase_reporting.log_rollout_data(1, args, [{}], {}, 2.5)

    assert reports == []


def test_custom_reward_timing_wraps_only_the_user_callable(monkeypatch):
    reports = []
    sample = SimpleNamespace(index=17)
    wall_times = iter((10.0, 12.0))
    monotonic_times = iter((100.0, 102.5))

    async def reward_function(args, reward_sample, *, scale):
        assert reward_sample is sample
        return scale

    monkeypatch.setattr(
        phase_reporting,
        "time",
        SimpleNamespace(
            time=lambda: next(wall_times),
            monotonic=lambda: next(monotonic_times),
        ),
    )
    monkeypatch.setattr(phase_reporting, "_resolve_hook", lambda path: reward_function)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    token = phase_reporting._CURRENT_ROLLOUT_ID.set(3)
    try:
        result = asyncio.run(
            phase_reporting.custom_reward_with_timing(
                SimpleNamespace(training_gym_custom_reward_function_path="reward"),
                sample,
                scale=2,
            )
        )
    finally:
        phase_reporting._CURRENT_ROLLOUT_ID.reset(token)

    assert result == 2
    assert [args[3] for args, _ in reports] == ["phase_start", "phase_finish"]
    assert all(args[:3] == ("custom_reward", args[1], 3) for args, _ in reports)
    assert all(kwargs["occurrence_id"] == 17 for _, kwargs in reports)
    assert all(kwargs["timeline_lane"] == "reward" for _, kwargs in reports)
    assert [kwargs["timestamp"] for _, kwargs in reports] == [10.0, 12.0]
    assert [kwargs["monotonic_timestamp"] for _, kwargs in reports] == [
        100.0,
        102.5,
    ]


def test_custom_reward_timing_finishes_and_preserves_user_exception(monkeypatch):
    reports = []

    async def reward_function(args, sample):
        raise RuntimeError("reward failed")

    monkeypatch.setattr(phase_reporting, "_resolve_hook", lambda path: reward_function)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    token = phase_reporting._CURRENT_ROLLOUT_ID.set(1)
    try:
        with pytest.raises(RuntimeError, match="reward failed"):
            asyncio.run(
                phase_reporting.custom_reward_with_timing(
                    SimpleNamespace(training_gym_custom_reward_function_path="reward"),
                    SimpleNamespace(index=4),
                )
            )
    finally:
        phase_reporting._CURRENT_ROLLOUT_ID.reset(token)

    assert [args[3] for args, _ in reports] == ["phase_start", "phase_finish"]


def test_custom_reward_timing_records_group_reward_calls(monkeypatch):
    reports = []
    samples = [SimpleNamespace(index=8), SimpleNamespace(index=9)]

    async def reward_function(args, reward_samples):
        assert reward_samples is samples
        return [1, 0]

    monkeypatch.setattr(phase_reporting, "_resolve_hook", lambda path: reward_function)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    token = phase_reporting._CURRENT_ROLLOUT_ID.set(2)
    try:
        result = asyncio.run(
            phase_reporting.custom_reward_with_timing(
                SimpleNamespace(training_gym_custom_reward_function_path="reward"),
                samples,
            )
        )
    finally:
        phase_reporting._CURRENT_ROLLOUT_ID.reset(token)

    assert result == [1, 0]
    assert [args[3] for args, _ in reports] == ["phase_start", "phase_finish"]
    assert all(kwargs["occurrence_id"] == 8 for _, kwargs in reports)


def test_custom_reward_without_sample_index_is_not_recorded(monkeypatch):
    reports = []

    async def reward_function(args, sample):
        return 1

    monkeypatch.setattr(phase_reporting, "_resolve_hook", lambda path: reward_function)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    token = phase_reporting._CURRENT_ROLLOUT_ID.set(1)
    try:
        result = asyncio.run(
            phase_reporting.custom_reward_with_timing(
                SimpleNamespace(training_gym_custom_reward_function_path="reward"),
                SimpleNamespace(),
            )
        )
    finally:
        phase_reporting._CURRENT_ROLLOUT_ID.reset(token)

    assert result == 1
    assert reports == []


def test_rollout_timing_context_flushes_only_after_final_training_rollout(
    monkeypatch,
):
    contexts = []
    flushes = []

    def rollout_function(args, rollout_id, data_source, evaluation=False):
        contexts.append(phase_reporting._CURRENT_ROLLOUT_ID.get())
        return rollout_id

    monkeypatch.setattr(phase_reporting, "_resolve_hook", lambda path: rollout_function)
    monkeypatch.setattr(
        phase_reporting,
        "flush_async_timing_events",
        lambda: flushes.append(True),
    )
    args = SimpleNamespace(
        num_rollout=2,
        training_gym_rollout_function_path="rollout",
    )

    assert phase_reporting.rollout_with_timing_context(args, 0, object()) == 0
    assert phase_reporting.rollout_with_timing_context(args, 1, object()) == 1
    assert (
        phase_reporting.rollout_with_timing_context(
            args,
            1,
            object(),
            evaluation=True,
        )
        == 1
    )

    assert contexts == [0, 1, None]
    assert flushes == [True]
    assert phase_reporting._CURRENT_ROLLOUT_ID.get() is None


def test_rollout_timing_context_flushes_and_resets_context_on_error(monkeypatch):
    flushes = []

    def rollout_function(args, rollout_id, data_source, evaluation=False):
        assert phase_reporting._CURRENT_ROLLOUT_ID.get() == rollout_id
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(phase_reporting, "_resolve_hook", lambda path: rollout_function)
    monkeypatch.setattr(
        phase_reporting,
        "flush_async_timing_events",
        lambda: flushes.append(True),
    )

    with pytest.raises(RuntimeError, match="rollout failed"):
        phase_reporting.rollout_with_timing_context(
            SimpleNamespace(training_gym_rollout_function_path="rollout"),
            0,
            object(),
        )

    assert flushes == [True]
    assert phase_reporting._CURRENT_ROLLOUT_ID.get() is None


def test_async_train_hook_reports_inner_model_and_optimizer_intervals(monkeypatch):
    reports = []
    wall_times = iter((100.0, 104.0, 104.25, 105.5))
    monotonic_times = iter((10.0, 14.0, 14.25, 15.5))
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(phase_reporting.time, "time", lambda: next(wall_times))
    monkeypatch.setattr(
        phase_reporting.time, "monotonic", lambda: next(monotonic_times)
    )

    args = SimpleNamespace(async_mode=True)
    optimizer = _Optimizer()
    phase_reporting.before_train_step_hook(args, 2, 0, object(), optimizer, object())

    assert optimizer.step() == "updated"
    assert reports == [
        (
            (TrainingSubstep.FORWARD_BACKWARD, args, 2, "phase_start"),
            {
                "occurrence_id": 0,
                "timestamp": 100.0,
                "monotonic_timestamp": 10.0,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Forward / backward",
            },
        ),
        (
            (TrainingSubstep.FORWARD_BACKWARD, args, 2, "phase_finish"),
            {
                "occurrence_id": 0,
                "timestamp": 104.0,
                "monotonic_timestamp": 14.0,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Forward / backward",
            },
        ),
        (
            (TrainingSubstep.OPTIMIZER_STEP, args, 2, "phase_start"),
            {
                "occurrence_id": 0,
                "timestamp": 104.25,
                "monotonic_timestamp": 14.25,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Optimizer step",
            },
        ),
        (
            (TrainingSubstep.OPTIMIZER_STEP, args, 2, "phase_finish"),
            {
                "occurrence_id": 0,
                "timestamp": 105.5,
                "monotonic_timestamp": 15.5,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Optimizer step",
            },
        ),
    ]


def test_async_train_hook_closes_forward_backward_when_optimizer_is_skipped(
    monkeypatch,
):
    reports = []
    wall_times = iter((100.0, 104.0))
    monotonic_times = iter((10.0, 14.0))
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(phase_reporting.time, "time", lambda: next(wall_times))
    monkeypatch.setattr(
        phase_reporting.time, "monotonic", lambda: next(monotonic_times)
    )

    args = SimpleNamespace(async_mode=True)
    optimizer = _Optimizer()
    phase_reporting.before_train_step_hook(args, 2, 0, object(), optimizer, object())

    assert optimizer.zero_grad() == "cleared"
    assert [report[0][3] for report in reports] == ["phase_start", "phase_finish"]
    assert all(report[0][0] is TrainingSubstep.FORWARD_BACKWARD for report in reports)
    assert reports[1][1]["timestamp"] == 104.0


def test_async_train_hook_profiles_log_prob_forwards_without_synchronizing(
    monkeypatch,
):
    reports = []
    wall_times = iter((90.0, 92.0, 95.0, 100.0))
    slime = ModuleType("slime")
    slime.__path__ = []
    slime_utils = ModuleType("slime.utils")
    slime_utils.__path__ = []
    slime_timer = ModuleType("slime.utils.timer")
    timer_snapshots = iter(
        (
            {"ref_log_probs": 8.0},
            {"ref_log_probs": 9.5, "teacher_log_probs": 3.0},
            {
                "ref_log_probs": 9.5,
                "teacher_log_probs": 5.0,
                "log_probs": 12.0,
            },
            {
                "ref_log_probs": 9.5,
                "teacher_log_probs": 5.0,
                "log_probs": 16.837,
            },
        )
    )
    slime_timer.Timer = lambda: SimpleNamespace(log_dict=lambda: next(timer_snapshots))
    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.utils", slime_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.timer", slime_timer)
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(phase_reporting.time, "time", lambda: next(wall_times))
    monkeypatch.setattr(phase_reporting, "_LOG_PROB_STARTS", {})

    args = SimpleNamespace(async_mode=True)
    phase_reporting.before_log_prob_hook(args, object(), "ref_")
    phase_reporting.before_log_prob_hook(args, object(), "teacher_")
    phase_reporting.before_log_prob_hook(args, object(), "")
    phase_reporting.before_train_step_hook(args, 2, 0, object(), _Optimizer(), object())

    assert [args[0] for args, _ in reports] == [
        TrainingSubstep.REFERENCE_LOG_PROBS,
        TrainingSubstep.REFERENCE_LOG_PROBS,
        TrainingSubstep.TEACHER_LOG_PROBS,
        TrainingSubstep.TEACHER_LOG_PROBS,
        TrainingSubstep.POLICY_LOG_PROBS,
        TrainingSubstep.POLICY_LOG_PROBS,
        TrainingSubstep.FORWARD_BACKWARD,
    ]
    assert [kwargs["timestamp"] for _, kwargs in reports] == pytest.approx(
        [90.0, 91.5, 92.0, 94.0, 95.0, 99.837, 100.0]
    )
    for _, kwargs in reports[:6]:
        assert kwargs["occurrence_id"] == 0
        assert kwargs["timeline_lane"] == "training"
        assert kwargs["parent_phase"] == SlimeStatus.TRAINING.value
    assert [kwargs["display_name"] for _, kwargs in reports[:6:2]] == [
        "Reference log probabilities",
        "Teacher log probabilities",
        "Policy log probabilities",
    ]


def test_critic_forward_is_not_recorded_as_policy_log_probs(monkeypatch):
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(phase_reporting, "_LOG_PROB_STARTS", {})

    phase_reporting.before_log_prob_hook(
        SimpleNamespace(async_mode=True, training_gym_role="critic"),
        object(),
        "",
    )

    assert phase_reporting._LOG_PROB_STARTS == {}


def test_sync_train_hook_keeps_existing_optimizer_report(monkeypatch):
    reports = []
    monkeypatch.setattr(
        phase_reporting,
        "report_phase",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    args = SimpleNamespace(async_mode=False, num_rollout=4)
    optimizer = _Optimizer()

    phase_reporting.before_train_step_hook(
        args=args,
        rollout_id=1,
        step_id=2,
        model=object(),
        optimizer=optimizer,
        opt_param_scheduler=object(),
    )

    assert reports[0][0] == (SlimeStatus.OPTIMIZER_STEP, args)
    assert not hasattr(optimizer, phase_reporting._OPTIMIZER_TIMING_INSTALLED)


def test_sync_step_boundaries_keep_direct_delivery(monkeypatch):
    reports = []
    monkeypatch.setattr(
        phase_reporting,
        "_post_framework_status",
        lambda payload, timeout: reports.append((payload, timeout)),
    )
    monkeypatch.setattr(phase_reporting.time, "time", lambda: 123.0)

    phase_reporting.report_step_event(
        SlimeStatus.TRAINING,
        SimpleNamespace(num_rollout=3, async_mode=False),
        rollout_id=1,
        step_event="finish",
    )

    assert reports == [
        (
            {
                "training_run_id": "",
                "app_name": "",
                "modal_app_id": "",
                "phase": SlimeStatus.TRAINING.value,
                "progress_current": 2,
                "progress_total": 3,
                "progress_unit": "step",
                "rollout_id": 1,
                "event_ts": 123.0,
                "step_event": "finish",
            },
            phase_reporting._STEP_EVENT_TIMEOUT_SECONDS,
        )
    ]


def test_async_step_boundaries_are_enqueued(monkeypatch):
    reports = []
    monkeypatch.setattr(phase_reporting, "_enqueue_async_timing_event", reports.append)

    phase_reporting.report_async_timing_event(
        TrainingSubstep.FORWARD_BACKWARD,
        SimpleNamespace(
            num_rollout=3,
            async_mode=True,
            training_gym_role="actor",
            rank=3,
            world_size=8,
        ),
        rollout_id=1,
        event_type="phase_start",
        occurrence_id=2,
        timestamp=123.0,
        timeline_lane="training",
        parent_phase="training",
        display_name="Actor update",
    )

    assert reports[0]["event_type"] == "phase_start"
    assert reports[0]["occurrence_id"] == 2
    assert reports[0]["timestamp"] == 123.0
    assert reports[0]["role"] == "actor"
    assert reports[0]["rank"] == 3
    assert reports[0]["world_size"] == 8
    assert reports[0]["timeline_lane"] == "training"
    assert reports[0]["parent_phase"] == "training"
    assert reports[0]["display_name"] == "Actor update"


def test_training_attempt_is_only_added_to_async_timing_events(monkeypatch):
    reports = []
    monkeypatch.setenv("TRAINING_GYM_TRAINING_ATTEMPT", "2")
    monkeypatch.setattr(phase_reporting, "_enqueue_async_timing_event", reports.append)

    phase_reporting.report_async_timing_event(
        SlimeStatus.TRAINING,
        SimpleNamespace(num_rollout=1, async_mode=True),
        rollout_id=0,
        event_type="phase_start",
        timestamp=123.0,
    )

    assert reports[0]["training_attempt"] == 2
    assert "training_attempt" not in reporting._run_context(SimpleNamespace())


def test_timing_events_use_deterministic_idempotency_keys(monkeypatch):
    store = _TimingStore()
    monkeypatch.setattr(metadata, "_step_times_dict", lambda: store)

    first = _timing_event(training_attempt=2)
    second = _timing_event(
        training_attempt=2,
        event_type="phase_finish",
        timestamp=2.0,
    )
    reporting._enqueue_async_timing_event(first)
    reporting._enqueue_async_timing_event({**first, "timestamp": 1.5})
    reporting._enqueue_async_timing_event(second)
    assert reporting.flush_async_timing_events()

    assert len(store) == 2
    assert (
        store[("run-1", "timing_event", 2, "", 0, "training", "phase_start", -1, -1)][
            "timestamp"
        ]
        == 1.0
    )
    assert (
        store[("run-1", "timing_event", 2, "", 0, "training", "phase_finish", -1, -1)][
            "timestamp"
        ]
        == 2.0
    )


def test_distributed_timing_events_use_unique_rank_keys(monkeypatch):
    store = _TimingStore()
    monkeypatch.setattr(metadata, "_step_times_dict", lambda: store)

    for rank in (0, 1):
        for event_type, timestamp in (
            ("phase_start", 10.0 + rank),
            ("phase_finish", 12.0 + rank),
        ):
            reporting._enqueue_async_timing_event(
                _timing_event(
                    training_attempt=2,
                    role="actor",
                    rank=rank,
                    world_size=2,
                    phase="forward_backward",
                    event_type=event_type,
                    occurrence_id=0,
                    timestamp=timestamp,
                )
            )

    assert reporting.flush_async_timing_events()
    assert set(store) == {
        (
            "run-1",
            "timing_event",
            2,
            "actor",
            0,
            "forward_backward",
            "phase_start",
            0,
            0,
        ),
        (
            "run-1",
            "timing_event",
            2,
            "actor",
            0,
            "forward_backward",
            "phase_finish",
            0,
            0,
        ),
        (
            "run-1",
            "timing_event",
            2,
            "actor",
            0,
            "forward_backward",
            "phase_start",
            0,
            1,
        ),
        (
            "run-1",
            "timing_event",
            2,
            "actor",
            0,
            "forward_backward",
            "phase_finish",
            0,
            1,
        ),
    }
    assert all(event["world_size"] == 2 for event in store.values())


def test_timing_event_retry_reuses_the_same_key(monkeypatch):
    class Store:
        def __init__(self):
            self.keys = []

        def put(self, key, value, *, skip_if_exists):
            self.keys.append(key)
            if len(self.keys) < 5:
                raise OSError("temporary")
            return True

    store = Store()
    monkeypatch.setattr(metadata, "_step_times_dict", lambda: store)
    monkeypatch.setattr(reporting, "_TIMING_RETRY_DELAY_SECONDS", 0)

    reporting._enqueue_async_timing_event(_timing_event())
    assert reporting.flush_async_timing_events()

    assert len(store.keys) == 5
    assert len(set(store.keys)) == 1


def test_timing_delivery_failure_is_nonfatal(monkeypatch, capsys):
    class Store:
        def put(self, key, value, *, skip_if_exists):
            raise OSError("unavailable")

    monkeypatch.setattr(metadata, "_step_times_dict", Store)
    monkeypatch.setattr(reporting, "_TIMING_RETRY_DELAY_SECONDS", 0)
    failed_before = reporting._TIMING_FAILED_EVENTS

    reporting._enqueue_async_timing_event(_timing_event())

    assert not reporting.flush_async_timing_events()
    assert reporting._TIMING_FAILED_EVENTS == failed_before + 1
    assert (
        "Failed to write async timing event after 5 attempts" in capsys.readouterr().out
    )


def test_completed_step_is_queued_after_its_timing_event(monkeypatch):
    writes = []
    posts = []

    class Store:
        def put(self, key, value, *, skip_if_exists):
            writes.append((key, value))
            return True

    monkeypatch.setattr(metadata, "_step_times_dict", Store)
    monkeypatch.setattr(reporting, "_enqueue_async_timing_notification", posts.append)

    reporting._enqueue_async_timing_event(
        _timing_event(event_type="rollout_finish", timestamp=2.0)
    )

    assert reporting.flush_async_timing_events()
    assert len(writes) == 1
    assert posts == [
        {
            "training_run_id": "run-1",
            "training_attempt": 1,
            "completed_rollout_id": 0,
        }
    ]
    assert (
        inspect.signature(reporting.flush_async_timing_events)
        .parameters["timeout_seconds"]
        .default
        > reporting._ASYNC_STEP_TIMES_TIMEOUT_SECONDS
    )


def test_completed_step_http_does_not_block_timing_flush(monkeypatch):
    store = _TimingStore()
    report_queue = Queue()
    monkeypatch.setattr(metadata, "_step_times_dict", lambda: store)
    monkeypatch.setattr(reporting, "_REPORT_QUEUE", report_queue)
    monkeypatch.setattr(reporting, "_ensure_worker", lambda: None)
    monkeypatch.setattr(
        reporting, "_async_step_times_url", lambda: "https://dashboard/step-times"
    )

    payload = _timing_event(event_type="rollout_finish")
    reporting._enqueue_async_timing_event(payload)

    assert reporting.flush_async_timing_events()
    assert report_queue.get_nowait() == {
        "_url": "https://dashboard/step-times",
        "_timeout": reporting._ASYNC_STEP_TIMES_TIMEOUT_SECONDS,
        "training_run_id": "run-1",
        "training_attempt": 1,
        "completed_rollout_id": 0,
    }


def test_timing_worker_survives_unexpected_delivery_error(monkeypatch):
    store = _TimingStore()
    monkeypatch.setattr(metadata, "_step_times_dict", lambda: store)
    monkeypatch.setattr(
        reporting,
        "_enqueue_async_timing_notification",
        lambda payload: (_ for _ in ()).throw(RuntimeError("broken delivery")),
    )

    reporting._enqueue_async_timing_event(_timing_event(event_type="rollout_finish"))
    assert not reporting.flush_async_timing_events()

    reporting._enqueue_async_timing_event(_timing_event(rollout_id=1))
    assert not reporting.flush_async_timing_events()
    assert len(store) == 2


def test_training_roles_use_distinct_timing_keys(monkeypatch):
    store = _TimingStore()
    monkeypatch.setattr(metadata, "_step_times_dict", lambda: store)
    reporting._enqueue_async_timing_event(
        _timing_event(phase="forward_backward", occurrence_id=0, role="actor")
    )
    reporting._enqueue_async_timing_event(
        _timing_event(phase="forward_backward", occurrence_id=0, role="critic")
    )
    assert reporting.flush_async_timing_events()

    assert len(store) == 2


def test_timing_worker_start_failure_is_nonfatal(monkeypatch, capsys):
    def fail_to_start():
        raise RuntimeError("unavailable")

    monkeypatch.setattr(reporting, "_TIMING_WORKER_STARTED", False)
    monkeypatch.setattr(reporting, "_ensure_timing_worker", fail_to_start)

    reporting._enqueue_async_timing_event(_timing_event())

    assert not reporting.flush_async_timing_events()
    assert "Failed to queue async timing event" in capsys.readouterr().out


def test_timing_delivery_does_not_block_the_caller(monkeypatch):
    write_started = threading.Event()
    allow_write = threading.Event()

    class Store:
        def put(self, key, value, *, skip_if_exists):
            write_started.set()
            allow_write.wait()
            return True

    monkeypatch.setattr(metadata, "_step_times_dict", Store)

    reporting._enqueue_async_timing_event(_timing_event())

    assert write_started.wait(1)
    assert not reporting.flush_async_timing_events(0.01)
    allow_write.set()
    assert reporting.flush_async_timing_events()


def test_optimizer_timing_wrapper_is_reused_across_updates(monkeypatch):
    reports = []
    wall_times = iter((100.0, 101.0, 101.25, 102.0, 103.0, 104.0, 104.25, 105.0))
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(phase_reporting.time, "time", lambda: next(wall_times))

    args = SimpleNamespace(async_mode=True)
    optimizer = _Optimizer()
    phase_reporting.before_train_step_hook(args, 1, 0, object(), optimizer, object())
    wrapped_step = optimizer.step
    optimizer.step()
    phase_reporting.before_train_step_hook(args, 1, 1, object(), optimizer, object())

    assert optimizer.step == wrapped_step
    optimizer.step()
    assert [kwargs.get("occurrence_id") for _, kwargs in reports] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert [args[0][0] for args in reports] == [
        TrainingSubstep.FORWARD_BACKWARD,
        TrainingSubstep.FORWARD_BACKWARD,
        TrainingSubstep.OPTIMIZER_STEP,
        TrainingSubstep.OPTIMIZER_STEP,
        TrainingSubstep.FORWARD_BACKWARD,
        TrainingSubstep.FORWARD_BACKWARD,
        TrainingSubstep.OPTIMIZER_STEP,
        TrainingSubstep.OPTIMIZER_STEP,
    ]


def test_non_primary_rank_records_update_timing(monkeypatch):
    reports = []
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        phase_reporting,
        "report_async_timing_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )

    optimizer = _Optimizer()
    phase_reporting.before_train_step_hook(
        SimpleNamespace(async_mode=True, rank=1, world_size=2),
        0,
        0,
        object(),
        optimizer,
        object(),
    )

    assert hasattr(optimizer, phase_reporting._OPTIMIZER_TIMING_INSTALLED)
    assert reports[0][0][0] == TrainingSubstep.FORWARD_BACKWARD
