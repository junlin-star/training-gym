import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.frameworks.slime import phase_reporting, reporting


class _Optimizer:
    def step(self):
        return "updated"


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
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        phase_reporting,
        "report_step_event",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(phase_reporting.time, "time", lambda: next(wall_times))

    args = SimpleNamespace(async_mode=True)
    optimizer = _Optimizer()
    phase_reporting.before_train_step_hook(args, 2, 3, object(), optimizer, object())

    assert optimizer.step() == "updated"
    assert reports == [
        (
            (SlimeStatus.TRAIN_MODEL, args, 2, "phase_start"),
            {"step_id": 3, "event_ts": 100.0},
        ),
        (
            (SlimeStatus.TRAIN_MODEL, args, 2, "phase_finish"),
            {"step_id": 3, "event_ts": 104.0},
        ),
        (
            (SlimeStatus.OPTIMIZER_STEP, args, 2, "phase_start"),
            {"step_id": 3, "event_ts": 104.0},
        ),
        (
            (SlimeStatus.OPTIMIZER_STEP, args, 2, "phase_finish"),
            {"step_id": 3, "event_ts": 105.5},
        ),
    ]


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
        SlimeStatus.TRAIN_MODEL,
        SimpleNamespace(
            num_rollout=3,
            async_mode=True,
            training_gym_role="actor",
        ),
        rollout_id=1,
        step_event="phase_start",
        step_id=2,
        event_ts=123.0,
    )

    assert reports[0]["step_event"] == "phase_start"
    assert reports[0]["step_id"] == 2
    assert reports[0]["event_ts"] == 123.0
    assert reports[0]["training_role"] == "actor"


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
    assert [kwargs["step_id"] for _, kwargs in reports] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [args[0][0] for args in reports] == [
        SlimeStatus.TRAIN_MODEL,
        SlimeStatus.TRAIN_MODEL,
        SlimeStatus.OPTIMIZER_STEP,
        SlimeStatus.OPTIMIZER_STEP,
        SlimeStatus.TRAIN_MODEL,
        SlimeStatus.TRAIN_MODEL,
        SlimeStatus.OPTIMIZER_STEP,
        SlimeStatus.OPTIMIZER_STEP,
    ]


def test_non_primary_rank_does_not_report_update_timing(monkeypatch):
    dist = ModuleType("torch.distributed")
    setattr(dist, "is_available", lambda: True)
    setattr(dist, "is_initialized", lambda: True)
    setattr(dist, "get_rank", lambda: 1)
    torch = ModuleType("torch")
    setattr(torch, "distributed", dist)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setattr(phase_reporting, "_call_hook", lambda *args, **kwargs: None)

    optimizer = _Optimizer()
    phase_reporting.before_train_step_hook(
        SimpleNamespace(async_mode=True), 0, 0, object(), optimizer, object()
    )

    assert not hasattr(optimizer, phase_reporting._OPTIMIZER_TIMING_INSTALLED)
