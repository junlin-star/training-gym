from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.common.step_timing import (
    advance_synchronous_timing_watermark,
    aggregate_step_times,
    aggregate_training_time_intervals,
    record_step_time_event,
    restore_checkpoint_step_times,
)
from modal_training_gym.frameworks.slime import phase_reporting

EVAL_BEFORE = SlimeStatus.EVAL_ROLLOUT_LOGGING.value
EVAL_AFTER = f"{EVAL_BEFORE}_end"
ROLLOUT_LOGGING = SlimeStatus.ROLLOUT_LOGGING.value
OFFLOAD_ROLLOUT = SlimeStatus.OFFLOAD_ROLLOUT.value
COMPUTE_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
OFFLOAD_TRAIN = SlimeStatus.OFFLOAD_TRAIN.value
WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value

SUBSTEP_ORDER = [
    EVAL_BEFORE,
    ROLLOUT_LOGGING,
    OFFLOAD_ROLLOUT,
    COMPUTE_LOG_PROBS,
    OPTIMIZER_STEP,
    CHECKPOINT_SAVE,
    OFFLOAD_TRAIN,
    WEIGHT_SYNC,
    EVAL_AFTER,
]
OPTIONAL_SUBSTEPS = {
    EVAL_BEFORE,
    OFFLOAD_ROLLOUT,
    CHECKPOINT_SAVE,
    OFFLOAD_TRAIN,
    EVAL_AFTER,
}

RUN_ID = "run-hooks"
STEP = 1

EVENT_REPORTS = {
    "step_start": (ROLLOUT_LOGGING, "start"),
    "step_complete": (WEIGHT_SYNC, "finish"),
    "substep_window_start": (ROLLOUT_LOGGING, "substep_start"),
    "substep_finish": (WEIGHT_SYNC, "substep_finish"),
    "eval_begin": (EVAL_BEFORE, "eval_begin"),
    "eval_end": (EVAL_BEFORE, "eval_end"),
}

STEP_SCHEDULE = [
    (0.0, "substep_window_start"),
    (0.5, "eval_begin"),
    (2.0, "step_start"),
    (4.0, "offload_rollout"),
    (6.0, "compute_log_probs"),
    (10.0, "optimizer_step"),
    (12.0, "checkpoint_save"),
    (13.0, "offload_train"),
    (15.0, "weight_sync"),
    (16.0, "eval_end"),
    (18.0, "substep_finish"),
    (18.0, "step_complete"),
]

EXPECTED_DURATIONS = {
    EVAL_BEFORE: 1.5,
    ROLLOUT_LOGGING: 2.0,
    OFFLOAD_ROLLOUT: 2.0,
    COMPUTE_LOG_PROBS: 4.0,
    OPTIMIZER_STEP: 2.0,
    CHECKPOINT_SAVE: 1.0,
    OFFLOAD_TRAIN: 2.0,
    WEIGHT_SYNC: 1.0,
    EVAL_AFTER: 2.0,
}


def test_checkpoint_restore_uses_zero_based_rollout_id():
    archived_step_times = {
        str(step): {"start": step, "end": step + 1, "duration_s": 1}
        for step in range(1, 4)
    }
    run = TrainingRun(
        training_run_id=RUN_ID,
        framework=Framework.SLIME,
        config={},
        metadata={
            "training_step_timing_attempts": {"1": {"step_times": archived_step_times}}
        },
    )

    restore_checkpoint_step_times(run, 2, checkpoint_rollout_id=1)

    assert run.step_times == {
        "1": archived_step_times["1"],
        "2": archived_step_times["2"],
    }
    assert run.substep_times == {}
    assert advance_synchronous_timing_watermark(run, 2) == 2

    run.step_times = {}
    assert (
        advance_synchronous_timing_watermark(
            run,
            3,
            completed_through_step=2,
        )
        == 2
    )
    assert run.step_times == {}


def test_training_interval_preserves_active_duration():
    events = [
        {
            "training_attempt": 1,
            "progress_current": 1,
            "phase": "custom_reward_function",
            "step_event": boundary,
            "step_id": 0,
            "event_ts": timestamp,
            "training_role": "driver",
            "active_duration_s": 4.0,
        }
        for boundary, timestamp in (("phase_start", 10.0), ("phase_finish", 15.0))
    ]

    intervals = aggregate_training_time_intervals(
        events, training_attempt=1, first_step=1, num_steps=1
    )

    assert intervals["1"]["custom_reward_function"]["duration_s"] == 4.0


def test_substep_finish_records_timing_and_triggers_reconciliation(monkeypatch):
    timing_events = []
    completed_step_statuses = []
    monkeypatch.delenv("TRAINING_GYM_ASYNC_MODE", raising=False)
    monkeypatch.setattr(phase_reporting, "_enqueue_timing_event", timing_events.append)
    monkeypatch.setattr(
        phase_reporting,
        "_enqueue_completed_step_status",
        completed_step_statuses.append,
    )

    phase_reporting.report_step_event(
        "train_model", rollout_id=0, step_event="phase_start"
    )
    phase_reporting.report_step_event(
        "weight_sync", rollout_id=0, step_event="substep_finish"
    )

    assert [event["step_event"] for event in timing_events] == [
        "phase_start",
        "substep_finish",
    ]
    assert [event["step_event"] for event in completed_step_statuses] == [
        "substep_finish"
    ]
    assert completed_step_statuses[0]["completed_step"] == 1


def test_train_model_status_keeps_legacy_optimizer_phase(monkeypatch):
    timing_events = []
    framework_statuses = []
    monkeypatch.setattr(phase_reporting, "_enqueue_timing_event", timing_events.append)
    monkeypatch.setattr(phase_reporting, "_enqueue", framework_statuses.append)

    phase_reporting.report_phase(
        phase_reporting.SlimeStatus.TRAIN_MODEL,
        rollout_id=0,
        progress_current=1,
        progress_total=2,
    )

    assert [event["phase"] for event in timing_events] == ["train_model"]
    assert [status["phase"] for status in framework_statuses] == [
        "optimizer_step",
        "train_model",
    ]
    assert framework_statuses[0]["legacy_optimizer_status"] is True


def build_step_times_dict(
    schedule: list[tuple[float, str]],
    *,
    step: int = STEP,
    offset: float = 0.0,
    into: dict[str, float] | None = None,
) -> dict[str, float]:
    """Feed a (timestamp, event) schedule through the same
    record_step_time_event the dashboard's /api/framework-status handler uses."""
    step_times: dict[str, float] = {} if into is None else into
    for ts, event in schedule:
        phase, step_event = EVENT_REPORTS.get(event, (event, ""))
        record_step_time_event(step_times, RUN_ID, step, phase, step_event, ts + offset)
    return step_times


def aggregate_durations(
    schedule: list[tuple[float, str]],
) -> dict[str, float | None]:
    _, substep_times = aggregate_step_times(
        build_step_times_dict(schedule),
        RUN_ID,
        STEP,
        SUBSTEP_ORDER,
        OPTIONAL_SUBSTEPS,
    )
    return {
        substep: entry["duration_s"]
        for substep, entry in substep_times[str(STEP)].items()
    }


def test_substep_times_aggregation():
    step_times, substep_times = aggregate_step_times(
        build_step_times_dict(STEP_SCHEDULE),
        RUN_ID,
        STEP,
        SUBSTEP_ORDER,
        OPTIONAL_SUBSTEPS,
    )

    assert step_times[str(STEP)] == {"start": 2.0, "end": 18.0, "duration_s": 16.0}

    durations = {
        substep: entry["duration_s"]
        for substep, entry in substep_times[str(STEP)].items()
    }
    assert durations == EXPECTED_DURATIONS

    assert substep_times[str(STEP)][EVAL_BEFORE]["start"] == 0.5
    assert substep_times[str(STEP)][ROLLOUT_LOGGING]["start"] == 2.0


def test_replayed_step_replaces_stale_substep_times():
    step_times_dict = build_step_times_dict(STEP_SCHEDULE)
    retry_schedule = [
        (0.0, "substep_window_start"),
        (0.5, "eval_begin"),
        (2.0, "step_start"),
        (4.0, "offload_rollout"),
        (6.0, "compute_log_probs"),
        (10.0, "optimizer_step"),
        (11.0, "optimizer_step"),
        (13.0, "offload_train"),
        (15.0, "weight_sync"),
        (16.0, "eval_end"),
        (18.0, "substep_finish"),
        (18.0, "step_complete"),
    ]
    build_step_times_dict(retry_schedule, offset=100.0, into=step_times_dict)

    step_times, substep_times = aggregate_step_times(
        step_times_dict,
        RUN_ID,
        STEP,
        SUBSTEP_ORDER,
        OPTIONAL_SUBSTEPS,
    )

    assert step_times[str(STEP)] == {
        "start": 102,
        "end": 118,
        "duration_s": 16,
    }
    assert substep_times[str(STEP)] == {
        EVAL_BEFORE: {"start": 100.5, "duration_s": 1.5},
        ROLLOUT_LOGGING: {"start": 102.0, "duration_s": 2.0},
        OFFLOAD_ROLLOUT: {"start": 104.0, "duration_s": 2.0},
        COMPUTE_LOG_PROBS: {"start": 106.0, "duration_s": 4.0},
        OPTIMIZER_STEP: {"start": 110.0, "duration_s": 3.0},
        OFFLOAD_TRAIN: {"start": 113.0, "duration_s": 2.0},
        WEIGHT_SYNC: {"start": 115.0, "duration_s": 1.0},
        EVAL_AFTER: {"start": 116.0, "duration_s": 2.0},
    }


def test_fractional_events_preserve_integer_step_format():
    step_times, substep_times = aggregate_step_times(
        {
            f"{RUN_ID}:1:start": 2.875,
            f"{RUN_ID}:1:finish": 18.999,
            f"{RUN_ID}:1:substep:{ROLLOUT_LOGGING}": 2.875,
            f"{RUN_ID}:1:substep:{OFFLOAD_ROLLOUT}": 4.125,
        },
        RUN_ID,
        1,
        SUBSTEP_ORDER,
        OPTIONAL_SUBSTEPS,
    )

    assert step_times["1"] == {"start": 2, "end": 18, "duration_s": 16}
    assert all(type(value) is int for value in step_times["1"].values())
    assert substep_times["1"][ROLLOUT_LOGGING] == {
        "start": 2.875,
        "duration_s": 1.25,
    }


def test_in_loop_generate_stamp_splits_eval_before_from_generation():
    schedule = [
        (0.0, "substep_window_start"),
        (0.0, "eval_begin"),
        (2.0, "step_start"),
        (2.0, ROLLOUT_LOGGING),
        (4.0, "offload_rollout"),
        (6.0, "compute_log_probs"),
        (10.0, "optimizer_step"),
        (12.0, "checkpoint_save"),
        (13.0, "offload_train"),
        (15.0, "weight_sync"),
        (16.0, "eval_end"),
        (18.0, "substep_finish"),
        (18.0, "step_complete"),
    ]
    durations = aggregate_durations(schedule)
    assert durations[EVAL_BEFORE] == 2.0
    assert durations[ROLLOUT_LOGGING] == 2.0


def test_substep_times_with_missing_substeps():
    schedule_missing_offload_rollout = [
        (0.0, "substep_window_start"),
        (0.5, "eval_begin"),
        (2.0, "step_start"),
        (6.0, "compute_log_probs"),
        (10.0, "optimizer_step"),
        (12.0, "checkpoint_save"),
        (13.0, "offload_train"),
        (15.0, "weight_sync"),
        (16.0, "eval_end"),
        (18.0, "substep_finish"),
        (18.0, "step_complete"),
    ]
    assert aggregate_durations(schedule_missing_offload_rollout) == {
        EVAL_BEFORE: 1.5,
        ROLLOUT_LOGGING: 4.0,
        COMPUTE_LOG_PROBS: 4.0,
        OPTIMIZER_STEP: 2.0,
        CHECKPOINT_SAVE: 1.0,
        OFFLOAD_TRAIN: 2.0,
        WEIGHT_SYNC: 1.0,
        EVAL_AFTER: 2.0,
    }

    schedule_missing_both_offloads = [
        (0.0, "substep_window_start"),
        (2.0, "step_start"),
        (6.0, "compute_log_probs"),
        (10.0, "optimizer_step"),
        (15.0, "weight_sync"),
        (18.0, "substep_finish"),
        (18.0, "step_complete"),
    ]
    assert aggregate_durations(schedule_missing_both_offloads) == {
        ROLLOUT_LOGGING: 4.0,
        COMPUTE_LOG_PROBS: 4.0,
        OPTIMIZER_STEP: 5.0,
        WEIGHT_SYNC: 3.0,
    }

    schedule_missing_substep_finish = [
        (0.0, "substep_window_start"),
        (0.5, "eval_begin"),
        (2.0, "step_start"),
        (4.0, "offload_rollout"),
        (6.0, "compute_log_probs"),
        (10.0, "optimizer_step"),
        (12.0, "checkpoint_save"),
        (13.0, "offload_train"),
        (15.0, "weight_sync"),
        (16.0, "eval_end"),
        (18.0, "step_complete"),
    ]
    assert aggregate_durations(schedule_missing_substep_finish) == EXPECTED_DURATIONS
    schedule_missing_all_finish_events = schedule_missing_substep_finish[:-1]
    assert aggregate_durations(schedule_missing_all_finish_events) == {
        **EXPECTED_DURATIONS,
        EVAL_AFTER: None,
    }


def test_substep_times_with_missing_optional_substeps():
    schedule_missing_eval_end = [
        (0.0, "substep_window_start"),
        (0.5, "eval_begin"),
        (2.0, "step_start"),
        (4.0, "offload_rollout"),
        (6.0, "compute_log_probs"),
        (10.0, "optimizer_step"),
        (12.0, "checkpoint_save"),
        (13.0, "offload_train"),
        (15.0, "weight_sync"),
        (18.0, "substep_finish"),
        (18.0, "step_complete"),
    ]
    assert aggregate_durations(schedule_missing_eval_end) == {
        EVAL_BEFORE: 1.5,
        ROLLOUT_LOGGING: 2.0,
        OFFLOAD_ROLLOUT: 2.0,
        COMPUTE_LOG_PROBS: 4.0,
        OPTIMIZER_STEP: 2.0,
        CHECKPOINT_SAVE: 1.0,
        OFFLOAD_TRAIN: 2.0,
        WEIGHT_SYNC: 3.0,
    }

    schedule_missing_all_optional = [
        (0.0, "substep_window_start"),
        (2.0, "step_start"),
        (4.0, "offload_rollout"),
        (6.0, "compute_log_probs"),
        (10.0, "optimizer_step"),
        (13.0, "offload_train"),
        (15.0, "weight_sync"),
        (18.0, "substep_finish"),
        (18.0, "step_complete"),
    ]
    assert aggregate_durations(schedule_missing_all_optional) == {
        ROLLOUT_LOGGING: 2.0,
        OFFLOAD_ROLLOUT: 2.0,
        COMPUTE_LOG_PROBS: 4.0,
        OPTIMIZER_STEP: 3.0,
        OFFLOAD_TRAIN: 2.0,
        WEIGHT_SYNC: 3.0,
    }


def test_substep_times_clamped_to_window():
    schedule = [
        (0.5, "eval_begin"),
        (2.0, "substep_window_start"),
        (1.0, "step_start"),
        (4.0, "offload_rollout"),
        (6.0, "compute_log_probs"),
        (10.0, "optimizer_step"),
        (12.0, "checkpoint_save"),
        (13.0, "offload_train"),
        (15.0, "weight_sync"),
        (19.0, "eval_end"),
        (18.0, "substep_finish"),
        (20.0, "step_complete"),
    ]
    step_times, substep_times = aggregate_step_times(
        build_step_times_dict(schedule),
        RUN_ID,
        STEP,
        SUBSTEP_ORDER,
        OPTIONAL_SUBSTEPS,
    )

    assert step_times[str(STEP)] == {"start": 1.0, "end": 20.0, "duration_s": 19.0}

    entries = substep_times[str(STEP)]
    assert entries[EVAL_BEFORE]["start"] == 0.5
    assert entries[ROLLOUT_LOGGING]["start"] == 2.0
    assert entries[EVAL_AFTER]["start"] == 18.0

    durations = {substep: entry["duration_s"] for substep, entry in entries.items()}
    assert durations == {
        **EXPECTED_DURATIONS,
        WEIGHT_SYNC: 3.0,
        EVAL_AFTER: 0.0,
    }


def test_substep_times_multiple_steps():
    step_times_dict: dict[str, float] = {}
    build_step_times_dict(STEP_SCHEDULE, step=1, into=step_times_dict)
    build_step_times_dict(STEP_SCHEDULE, step=2, offset=100.0, into=step_times_dict)

    step_times, substep_times = aggregate_step_times(
        step_times_dict,
        RUN_ID,
        3,
        SUBSTEP_ORDER,
        OPTIONAL_SUBSTEPS,
    )

    assert step_times["1"] == {"start": 2.0, "end": 18.0, "duration_s": 16.0}
    assert step_times["2"] == {"start": 102.0, "end": 118.0, "duration_s": 16.0}
    assert step_times["3"] == {"start": None, "end": None, "duration_s": None}

    for step in ("1", "2"):
        durations = {
            substep: entry["duration_s"]
            for substep, entry in substep_times[step].items()
        }
        assert durations == EXPECTED_DURATIONS
    assert substep_times["3"] == {}


def test_substep_times_adverse_timings():
    schedule = [
        (0.0, "substep_window_start"),
        (2.0, "step_start"),
        (4.0, "offload_rollout"),
        (4.0, "compute_log_probs"),
        (5.0, "offload_rollout"),
        (10.0, "optimizer_step"),
        (9.0, "offload_train"),
        (15.0, "weight_sync"),
        (18.0, "substep_finish"),
        (17.0, "substep_finish"),
        (18.0, "step_complete"),
    ]
    assert aggregate_durations(schedule) == {
        ROLLOUT_LOGGING: 2.0,
        OFFLOAD_ROLLOUT: 0.0,
        COMPUTE_LOG_PROBS: 5.0,
        OFFLOAD_TRAIN: 1.0,
        OPTIMIZER_STEP: 5.0,
        WEIGHT_SYNC: 3.0,
    }
