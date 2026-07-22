import inspect
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from modal_training_gym.common.status import SlimeStatus, resolve_framework_status
from modal_training_gym.common.step_timing import TrainingSubstep
from modal_training_gym.frameworks.slime import phase_reporting, reporting


class _Optimizer:
    def step(self):
        return "updated"


def test_existing_training_substeps_remain_framework_statuses():
    legacy_status_substeps = {
        TrainingSubstep.POLICY_LOG_PROBS,
        TrainingSubstep.REFERENCE_LOG_PROBS,
        TrainingSubstep.TEACHER_LOG_PROBS,
        TrainingSubstep.VALUE_INFERENCE,
        TrainingSubstep.FORWARD_BACKWARD,
        TrainingSubstep.OPTIMIZER_STEP,
    }
    for substep in TrainingSubstep:
        resolved = resolve_framework_status(substep.value, "slime")
        if substep in legacy_status_substeps:
            assert resolved is not None
        else:
            assert resolved is None


def test_record_step_interval_finishes_when_the_wrapped_call_raises(monkeypatch):
    reports = []
    args = SimpleNamespace(async_mode=True)
    monkeypatch.setattr(
        phase_reporting,
        "report_step_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="failed"):
        with phase_reporting.record_step_interval(
            TrainingSubstep.DATA_PREPROCESS,
            args,
            2,
            timeline_lane="training",
            parent_phase="training",
            display_name="Load & transfer training batch",
        ):
            raise RuntimeError("failed")

    assert [report[0][3] for report in reports] == ["phase_start", "phase_finish"]
    assert reports[0][1] == reports[1][1]


def test_disabled_step_interval_does_not_report(monkeypatch):
    reports = []
    monkeypatch.setattr(
        phase_reporting,
        "report_step_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )

    with phase_reporting.record_step_interval(
        TrainingSubstep.DATA_PREPROCESS,
        SimpleNamespace(async_mode=True),
        2,
        enabled=False,
    ):
        pass

    assert reports == []


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


def test_async_train_hook_reports_inner_model_and_optimizer_intervals(monkeypatch):
    reports = []
    wall_times = iter((100.0, 104.0, 105.5))
    monotonic_times = iter((10.0, 14.0, 15.5))
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        phase_reporting,
        "report_step_event",
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
                "step_id": 0,
                "event_ts": 100.0,
                "event_monotonic": 10.0,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Forward / backward",
            },
        ),
        (
            (TrainingSubstep.FORWARD_BACKWARD, args, 2, "phase_finish"),
            {
                "step_id": 0,
                "event_ts": 104.0,
                "event_monotonic": 14.0,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Forward / backward",
            },
        ),
        (
            (TrainingSubstep.OPTIMIZER_STEP, args, 2, "phase_start"),
            {
                "step_id": 0,
                "event_ts": 104.0,
                "event_monotonic": 14.0,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Optimizer step",
            },
        ),
        (
            (TrainingSubstep.OPTIMIZER_STEP, args, 2, "phase_finish"),
            {
                "step_id": 0,
                "event_ts": 105.5,
                "event_monotonic": 15.5,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Optimizer step",
            },
        ),
    ]


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
        "report_step_event",
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
    assert [kwargs["event_ts"] for _, kwargs in reports] == pytest.approx(
        [90.0, 91.5, 92.0, 94.0, 95.0, 99.837, 100.0]
    )
    for _, kwargs in reports[:6]:
        assert kwargs["step_id"] == 0
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

    phase_reporting.before_train_step_hook(args, 1, 2, object(), optimizer, object())

    assert reports[0][0] == (SlimeStatus.OPTIMIZER_STEP, args)
    assert not hasattr(optimizer, phase_reporting._OPTIMIZER_TIMING_INSTALLED)


def test_sync_step_boundaries_keep_direct_delivery(monkeypatch):
    reports = []
    monkeypatch.setattr(
        phase_reporting,
        "_post_framework_status",
        lambda payload, timeout: reports.append((payload, timeout)),
    )

    phase_reporting.report_step_event(
        SlimeStatus.TRAINING,
        SimpleNamespace(num_rollout=3, async_mode=False),
        rollout_id=1,
        step_event="finish",
        event_ts=123.0,
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

    phase_reporting.report_step_event(
        TrainingSubstep.FORWARD_BACKWARD,
        SimpleNamespace(
            num_rollout=3,
            async_mode=True,
            training_gym_role="actor",
            rank=3,
            world_size=8,
        ),
        rollout_id=1,
        step_event="phase_start",
        step_id=2,
        event_ts=123.0,
        timeline_lane="training",
        parent_phase="training",
        display_name="Actor update",
    )

    assert reports[0]["step_event"] == "phase_start"
    assert reports[0]["step_id"] == 2
    assert reports[0]["event_ts"] == 123.0
    assert reports[0]["training_role"] == "actor"
    assert reports[0]["training_rank"] == 3
    assert reports[0]["training_world_size"] == 8
    assert reports[0]["timeline_lane"] == "training"
    assert reports[0]["parent_phase"] == "training"
    assert reports[0]["display_name"] == "Actor update"


def test_training_attempt_is_only_added_to_async_timing_events(monkeypatch):
    reports = []
    monkeypatch.setenv("TRAINING_GYM_TRAINING_ATTEMPT", "2")
    monkeypatch.setattr(phase_reporting, "_enqueue_async_timing_event", reports.append)

    phase_reporting.report_step_event(
        SlimeStatus.TRAINING,
        SimpleNamespace(num_rollout=1, async_mode=True),
        rollout_id=0,
        step_event="phase_start",
        event_ts=123.0,
    )

    assert reports[0]["training_attempt"] == "2"
    assert "training_attempt" not in reporting._run_context(SimpleNamespace())


def test_timing_events_use_deterministic_idempotency_keys(monkeypatch):
    store = {}
    monkeypatch.setattr(reporting, "_step_times_dict", lambda: store)

    first = {
        "training_run_id": "run-1",
        "training_attempt": 2,
        "rollout_id": 0,
        "phase": "training",
        "step_event": "phase_start",
        "event_ts": 1.0,
    }
    second = {**first, "step_event": "phase_finish", "event_ts": 2.0}
    reporting._enqueue_async_timing_event(first)
    reporting._enqueue_async_timing_event({**first, "event_ts": 1.5})
    reporting._enqueue_async_timing_event(second)
    assert reporting.flush_async_timing_events()

    assert len(store) == 2
    assert (
        store[("run-1", "timing_event", 2, "", 0, "training", "phase_start", -1)][
            "event_ts"
        ]
        == 1.5
    )
    assert (
        store[("run-1", "timing_event", 2, "", 0, "training", "phase_finish", -1)][
            "event_ts"
        ]
        == 2.0
    )


def test_distributed_timing_events_are_batched_per_rank(monkeypatch):
    store = {}
    monkeypatch.setattr(reporting, "_step_times_dict", lambda: store)
    monkeypatch.setattr(reporting, "_RANK_TIMING_EVENTS", {})

    for rank in (0, 1):
        for step_event, event_ts in (
            ("phase_start", 10.0 + rank),
            ("phase_finish", 12.0 + rank),
        ):
            reporting._enqueue_async_timing_event(
                {
                    "training_run_id": "run-1",
                    "training_attempt": 2,
                    "training_role": "actor",
                    "training_rank": rank,
                    "training_world_size": 2,
                    "rollout_id": 0,
                    "progress_current": 1,
                    "phase": "train_model",
                    "step_event": step_event,
                    "step_id": 0,
                    "event_ts": event_ts,
                    "event_monotonic": event_ts,
                }
            )

    assert store == {}
    assert reporting.flush_async_timing_events()
    assert set(store) == {
        ("run-1", "rank_timing", 2, "actor", 0, 0),
        ("run-1", "rank_timing", 2, "actor", 0, 1),
    }
    assert all(len(batch["events"]) == 2 for batch in store.values())
    assert store[("run-1", "rank_timing", 2, "actor", 0, 1)]["training_world_size"] == 2


def test_timing_event_retry_reuses_the_same_key(monkeypatch):
    class Store:
        def __init__(self):
            self.keys = []

        def __setitem__(self, key, value):
            self.keys.append(key)
            if len(self.keys) < 5:
                raise OSError("temporary")

    store = Store()
    monkeypatch.setattr(reporting, "_step_times_dict", lambda: store)
    monkeypatch.setattr(reporting, "_TIMING_RETRY_DELAY_SECONDS", 0)

    reporting._enqueue_async_timing_event(
        {
            "training_run_id": "run-1",
            "rollout_id": 0,
            "phase": "training",
            "step_event": "phase_start",
        }
    )
    assert reporting.flush_async_timing_events()

    assert len(store.keys) == 5
    assert len(set(store.keys)) == 1


def test_timing_delivery_failure_is_nonfatal(monkeypatch, capsys):
    class Store:
        def __setitem__(self, key, value):
            raise OSError("unavailable")

    monkeypatch.setattr(reporting, "_step_times_dict", Store)
    monkeypatch.setattr(reporting, "_TIMING_RETRY_DELAY_SECONDS", 0)
    failed_before = reporting._TIMING_FAILED_EVENTS

    reporting._enqueue_async_timing_event(
        {
            "training_run_id": "run-1",
            "rollout_id": 0,
            "phase": "training",
            "step_event": "phase_start",
        }
    )

    assert not reporting.flush_async_timing_events()
    assert reporting._TIMING_FAILED_EVENTS == failed_before + 1
    assert (
        "Failed to write async timing event after 5 attempts" in capsys.readouterr().out
    )


def test_completed_step_is_saved_after_its_timing_event(monkeypatch):
    writes = []
    posts = []

    class Store:
        def __setitem__(self, key, value):
            writes.append((key, value))

    monkeypatch.setattr(reporting, "_step_times_dict", Store)
    monkeypatch.setattr(
        reporting, "_async_step_times_url", lambda: "https://dashboard/step-times"
    )
    monkeypatch.setattr(reporting, "_post", posts.append)

    reporting._enqueue_async_timing_event(
        {
            "training_run_id": "run-1",
            "training_attempt": 1,
            "rollout_id": 0,
            "progress_current": 1,
            "phase": "training",
            "step_event": "finish",
            "event_ts": 2.0,
        }
    )

    assert reporting.flush_async_timing_events()
    assert len(writes) == 1
    assert posts == [
        {
            "_url": "https://dashboard/step-times",
            "_timeout": reporting._ASYNC_STEP_TIMES_TIMEOUT_SECONDS,
            "training_run_id": "run-1",
            "training_attempt": 1,
            "rollout_id": 0,
            "progress_current": 1,
            "phase": "training",
            "step_event": "finish",
            "event_ts": 2.0,
        }
    ]
    assert (
        inspect.signature(reporting.flush_async_timing_events)
        .parameters["timeout_seconds"]
        .default
        > reporting._ASYNC_STEP_TIMES_TIMEOUT_SECONDS
    )


def test_training_roles_use_distinct_timing_keys(monkeypatch):
    store = {}
    monkeypatch.setattr(reporting, "_step_times_dict", lambda: store)
    event = {
        "training_run_id": "run-1",
        "rollout_id": 0,
        "phase": "train_model",
        "step_event": "phase_start",
        "step_id": 0,
    }

    reporting._enqueue_async_timing_event({**event, "training_role": "actor"})
    reporting._enqueue_async_timing_event({**event, "training_role": "critic"})
    assert reporting.flush_async_timing_events()

    assert len(store) == 2


def test_timing_event_without_complete_identity_is_nonfatal(monkeypatch):
    monkeypatch.setattr(
        reporting,
        "_step_times_dict",
        lambda: pytest.fail("store should not be opened without a run ID"),
    )

    reporting._enqueue_async_timing_event({})


def test_timing_worker_start_failure_is_nonfatal(monkeypatch, capsys):
    def fail_to_start():
        raise RuntimeError("unavailable")

    monkeypatch.setattr(reporting, "_TIMING_WORKER_STARTED", False)
    monkeypatch.setattr(reporting, "_TIMING_DROPPED_EVENTS", 0)
    monkeypatch.setattr(reporting, "_TIMING_REPORTED_DROPS", 0)
    monkeypatch.setattr(reporting, "_ensure_timing_worker", fail_to_start)

    reporting._enqueue_async_timing_event(
        {
            "training_run_id": "run-1",
            "rollout_id": 0,
            "phase": "training",
            "step_event": "phase_start",
        }
    )

    assert not reporting.flush_async_timing_events()
    assert reporting.flush_async_timing_events()
    assert "Failed to queue async timing event" in capsys.readouterr().out


def test_timing_delivery_does_not_block_the_caller(monkeypatch):
    write_started = threading.Event()
    allow_write = threading.Event()

    class Store:
        def __setitem__(self, key, value):
            write_started.set()
            allow_write.wait()

    monkeypatch.setattr(reporting, "_step_times_dict", Store)

    reporting._enqueue_async_timing_event(
        {
            "training_run_id": "run-1",
            "rollout_id": 0,
            "phase": "training",
            "step_event": "phase_start",
        }
    )

    assert write_started.wait(1)
    assert not reporting.flush_async_timing_events(0.01)
    allow_write.set()
    assert reporting.flush_async_timing_events()


def test_optimizer_timing_wrapper_is_reused_across_updates(monkeypatch):
    reports = []
    wall_times = iter((100.0, 101.0, 102.0, 103.0, 104.0, 105.0))
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        phase_reporting,
        "report_step_event",
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
    assert [kwargs.get("step_id") for _, kwargs in reports] == [
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
        "report_step_event",
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
