from modal_training_gym.common.async_timing_aggregation import (
    StepTimingSnapshot,
    aggregate_async_step_times,
    apply_step_timing_snapshot,
    reconcile_attempt_step_times,
    reconcile_completed_step_times,
)
from modal_training_gym.common.async_timing_types import (
    AsyncTimingEvent,
    AsyncTimingEventType,
)
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.common.timing_types import (
    StepTimes,
    SubstepTimes,
    TimingLane,
    TrainingSubstep,
)

EVAL_AFTER = f"{SlimeStatus.EVAL_ROLLOUT_LOGGING.value}_end"
ROLLOUT_LOGGING = SlimeStatus.ROLLOUT_LOGGING.value
OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value

RUN_ID = "run-hooks"


def _async_event(
    rollout_id: int,
    phase: str,
    event_type: AsyncTimingEventType,
    timestamp: float,
    *,
    training_attempt: int = 1,
    occurrence_id: int | None = None,
    role: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    monotonic_timestamp: float | None = None,
    timeline_lane: TimingLane | None = None,
    parent_phase: str | None = None,
    display_name: str | None = None,
) -> AsyncTimingEvent:
    return {
        "training_run_id": RUN_ID,
        "training_attempt": training_attempt,
        "rollout_id": rollout_id,
        "phase": phase,
        "event_type": event_type,
        "timestamp": timestamp,
        "monotonic_timestamp": (
            timestamp if monotonic_timestamp is None else monotonic_timestamp
        ),
        "occurrence_id": occurrence_id,
        "role": role,
        "rank": rank,
        "world_size": world_size,
        "timeline_lane": timeline_lane,
        "parent_phase": parent_phase,
        "display_name": display_name,
    }


def _snapshot(
    training_attempt: int,
    step_times: StepTimes,
    substep_times: SubstepTimes,
) -> StepTimingSnapshot:
    return {
        "displayed_training_attempt": training_attempt,
        "archived_timing_attempts": {},
        "timing_event_counts": {},
        "step_times": step_times,
        "substep_times": substep_times,
    }


def _run_with_timings(last_step: int) -> TrainingRun:
    return TrainingRun(
        training_run_id=RUN_ID,
        framework=Framework.SLIME,
        config={},
        step_times={
            str(step): {"start": step * 10, "end": step * 10 + 5, "duration_s": 5}
            for step in range(1, last_step + 1)
        },
        substep_times={
            str(step): {ROLLOUT_LOGGING: {"start": step * 10.0, "duration_s": 5.0}}
            for step in range(1, last_step + 1)
        },
    )


def test_training_run_reads_main_timings_and_stores_intervals_directly():
    main_payload = _run_with_timings(1).model_dump(mode="json")
    run = TrainingRun.model_validate(main_payload)
    assert run.substep_times == main_payload["substep_times"]

    run.substep_times = {
        "1": {
            TrainingSubstep.FORWARD_BACKWARD.value: {
                "start": 10.25,
                "duration_s": 2.5,
                "intervals": [
                    {
                        "step_id": 0,
                        "training_role": "actor",
                        "slowest_rank": 3,
                        "reported_rank_count": 8,
                        "training_world_size": 8,
                        "start": 10.25,
                        "duration_s": 2.5,
                    }
                ],
            }
        }
    }

    stored = run.model_dump(mode="json")
    assert stored["substep_times"]["1"]["forward_backward"]["intervals"][0] == {
        "start": 10.25,
        "duration_s": 2.5,
        "step_id": 0,
        "training_role": "actor",
        "slowest_rank": 3,
        "reported_rank_count": 8,
        "training_world_size": 8,
    }
    restored = TrainingRun.model_validate(stored)
    assert restored.substep_times == run.substep_times


def test_fresh_retry_displays_only_new_attempt_and_archives_previous_attempt():
    run = _run_with_timings(15)
    new_step_times = {
        str(step): {
            "start": 200 if step == 15 else None,
            "end": 205 if step == 15 else None,
            "duration_s": 5 if step == 15 else None,
        }
        for step in range(1, 16)
    }
    new_substep_times = {
        str(step): (
            {ROLLOUT_LOGGING: {"start": 200.0, "duration_s": 5.0}} if step == 15 else {}
        )
        for step in range(1, 16)
    }

    reconcile_attempt_step_times(
        run,
        new_step_times,
        new_substep_times,
        training_attempt=2,
        resume_from_iteration=None,
    )

    assert run.step_times == {"15": {"start": 200, "end": 205, "duration_s": 5}}
    assert set(run.substep_times or {}) == {"15"}
    attempts = (run.metadata or {})["archived_timing_attempts"]
    assert set(attempts["1"]["step_times"]) == {str(step) for step in range(1, 16)}
    assert set(attempts["1"]) == {"step_times", "substep_times"}
    assert set(attempts) == {"1"}
    assert (run.metadata or {})["displayed_training_attempt"] == 2


def test_main_era_timings_are_archived_as_the_previous_attempt():
    run = _run_with_timings(3)

    reconcile_attempt_step_times(
        run,
        {"1": {"start": 40, "end": 45, "duration_s": 5}},
        {"1": {ROLLOUT_LOGGING: {"start": 40.0, "duration_s": 5.0}}},
        training_attempt=4,
        resume_from_iteration=None,
    )

    metadata = run.metadata or {}
    assert metadata["displayed_training_attempt"] == 4
    assert set(metadata["archived_timing_attempts"]) == {"3"}


def test_reconciling_each_step_does_not_overwrite_the_previous_attempt():
    run = _run_with_timings(2)
    previous_attempt = dict(run.step_times or {})

    reconcile_completed_step_times(
        run,
        {"1": {"start": 200, "end": 205, "duration_s": 5}},
        {"1": {ROLLOUT_LOGGING: {"start": 200.0, "duration_s": 5.0}}},
        training_attempt=2,
        resume_from_iteration=None,
    )
    reconcile_completed_step_times(
        run,
        {"2": {"start": 210, "end": 215, "duration_s": 5}},
        {"2": {ROLLOUT_LOGGING: {"start": 210.0, "duration_s": 5.0}}},
        training_attempt=2,
        resume_from_iteration=None,
    )

    attempts = (run.metadata or {})["archived_timing_attempts"]
    assert attempts["1"]["step_times"] == previous_attempt
    assert set(attempts) == {"1"}
    assert set(run.step_times or {}) == {"1", "2"}
    assert (run.metadata or {})["displayed_training_attempt"] == 2


def test_reconciling_completed_steps_builds_the_live_first_attempt():
    run = TrainingRun(training_run_id=RUN_ID, framework=Framework.SLIME, config={})

    for step in (1, 2):
        reconcile_completed_step_times(
            run,
            {
                str(step): {
                    "start": step * 10,
                    "end": step * 10 + 5,
                    "duration_s": 5,
                }
            },
            {
                str(step): {
                    ROLLOUT_LOGGING: {
                        "start": step * 10.0,
                        "duration_s": 5.0,
                    }
                }
            },
            training_attempt=1,
            resume_from_iteration=None,
        )

    assert set(run.step_times or {}) == {"1", "2"}
    assert set(run.substep_times or {}) == {"1", "2"}
    assert (run.metadata or {})["displayed_training_attempt"] == 1


def test_previous_attempt_snapshot_fills_missing_canonical_timing():
    run = _run_with_timings(2)
    run.step_times.pop("1")
    run.substep_times.pop("1")
    snapshot = _snapshot(
        1,
        {"1": {"start": 10, "end": 15, "duration_s": 5}},
        {"1": {ROLLOUT_LOGGING: {"start": 10.0, "duration_s": 5.0}}},
    )

    apply_step_timing_snapshot(
        run,
        snapshot,
        training_attempt=2,
    )

    assert set(run.step_times or {}) == {"1", "2"}
    assert set(run.substep_times or {}) == {"1", "2"}
    assert (run.metadata or {})["displayed_training_attempt"] == 1


def test_current_attempt_snapshot_replaces_canonical_timing():
    run = _run_with_timings(2)
    snapshot = _snapshot(
        2,
        {"1": {"start": 200, "end": 205, "duration_s": 5}},
        {"1": {ROLLOUT_LOGGING: {"start": 200.0, "duration_s": 5.0}}},
    )

    apply_step_timing_snapshot(
        run,
        snapshot,
        training_attempt=2,
    )

    assert run.step_times == snapshot["step_times"]
    assert run.substep_times == snapshot["substep_times"]
    assert (run.metadata or {})["displayed_training_attempt"] == 2


def test_previous_attempt_snapshot_does_not_overwrite_canonical_timing():
    run = _run_with_timings(1)
    snapshot = _snapshot(
        1,
        {"1": {"start": 10, "end": None, "duration_s": None}},
        {"1": {ROLLOUT_LOGGING: {"start": 10.0, "duration_s": None}}},
    )

    apply_step_timing_snapshot(
        run,
        snapshot,
        training_attempt=2,
    )

    assert run.step_times == {"1": {"start": 10, "end": 15, "duration_s": 5}}
    assert run.substep_times == {
        "1": {ROLLOUT_LOGGING: {"start": 10.0, "duration_s": 5.0}}
    }


def test_checkpoint_retry_keeps_only_steps_before_new_attempt_boundary():
    run = _run_with_timings(17)
    run.substep_times["14"][ROLLOUT_LOGGING]["intervals"] = [
        {"step_id": 0, "start": 140.0, "duration_s": 5.0}
    ]
    new_step_times = {
        "15": {"start": 200, "end": 205, "duration_s": 5},
        "16": {"start": 210, "end": 215, "duration_s": 5},
    }
    new_substep_times = {
        "15": {ROLLOUT_LOGGING: {"start": 200.0, "duration_s": 5.0}},
        "16": {ROLLOUT_LOGGING: {"start": 210.0, "duration_s": 5.0}},
    }

    reconcile_attempt_step_times(
        run,
        new_step_times,
        new_substep_times,
        training_attempt=2,
        resume_from_iteration=13,
    )

    assert set(run.step_times or {}) == {str(step) for step in range(1, 17)}
    assert run.step_times["14"]["start"] == 140
    assert run.step_times["15"]["start"] == 200
    assert "17" not in run.step_times
    assert run.substep_times["14"][ROLLOUT_LOGGING]["intervals"] == [
        {"step_id": 0, "start": 140.0, "duration_s": 5.0}
    ]
    attempts = (run.metadata or {})["archived_timing_attempts"]
    assert set(attempts["1"]["step_times"]) == {str(step) for step in range(1, 18)}
    assert set(attempts) == {"1"}
    assert (run.metadata or {})["displayed_training_attempt"] == 2
    restored = TrainingRun.model_validate(run.model_dump(mode="json"))
    assert restored.substep_times["14"][ROLLOUT_LOGGING]["intervals"] == [
        {"step_id": 0, "start": 140.0, "duration_s": 5.0}
    ]


def test_completed_checkpoint_steps_keep_the_old_prefix_and_one_owner():
    run = _run_with_timings(17)

    for step in (15, 16):
        reconcile_completed_step_times(
            run,
            {
                str(step): {
                    "start": step * 20,
                    "end": step * 20 + 5,
                    "duration_s": 5,
                }
            },
            {
                str(step): {
                    ROLLOUT_LOGGING: {
                        "start": step * 20.0,
                        "duration_s": 5.0,
                    }
                }
            },
            training_attempt=2,
            resume_from_iteration=13,
        )

    assert set(run.step_times or {}) == {str(step) for step in range(1, 17)}
    assert run.step_times["14"]["start"] == 140
    assert run.step_times["15"]["start"] == 300
    assert run.step_times["16"]["start"] == 320
    assert (run.metadata or {})["displayed_training_attempt"] == 2
    assert set((run.metadata or {})["archived_timing_attempts"]) == {"1"}


def test_checkpoint_boundary_does_not_depend_on_first_reported_step():
    run = _run_with_timings(17)

    reconcile_completed_step_times(
        run,
        {"16": {"start": 320, "end": 325, "duration_s": 5}},
        {"16": {ROLLOUT_LOGGING: {"start": 320.0, "duration_s": 5.0}}},
        training_attempt=2,
        resume_from_iteration=13,
    )

    assert set(run.step_times or {}) == {str(step) for step in range(1, 15)} | {"16"}
    assert "15" not in (run.step_times or {})


def test_retry_without_new_timing_keeps_existing_display():
    run = _run_with_timings(3)

    reconcile_attempt_step_times(
        run,
        {"1": {"start": None, "end": None, "duration_s": None}},
        {"1": {}},
        training_attempt=2,
        resume_from_iteration=None,
    )

    assert set(run.step_times or {}) == {"1", "2", "3"}
    assert run.metadata == {"displayed_training_attempt": 1}


def test_retry_without_timing_is_not_mislabeled_by_a_later_attempt():
    run = _run_with_timings(1)

    reconcile_attempt_step_times(
        run,
        {"1": {"start": None, "end": None, "duration_s": None}},
        {"1": {}},
        training_attempt=2,
        resume_from_iteration=None,
    )
    reconcile_attempt_step_times(
        run,
        {"1": {"start": 300, "end": 305, "duration_s": 5}},
        {"1": {ROLLOUT_LOGGING: {"start": 300.0, "duration_s": 5.0}}},
        training_attempt=3,
        resume_from_iteration=None,
    )

    attempts = (run.metadata or {})["archived_timing_attempts"]
    assert set(attempts) == {"1"}
    assert attempts["1"]["step_times"]["1"]["start"] == 10
    assert (run.metadata or {})["displayed_training_attempt"] == 3


def test_later_retry_preserves_every_attempt_snapshot():
    run = _run_with_timings(2)
    run.metadata = {
        "displayed_training_attempt": 2,
        "archived_timing_attempts": {
            "1": {
                "step_times": {"1": {"start": 10, "end": 15, "duration_s": 5}},
                "substep_times": {
                    "1": {ROLLOUT_LOGGING: {"start": 10.0, "duration_s": 5.0}}
                },
            }
        },
    }

    reconcile_attempt_step_times(
        run,
        {"1": {"start": 300, "end": 305, "duration_s": 5}},
        {"1": {ROLLOUT_LOGGING: {"start": 300.0, "duration_s": 5.0}}},
        training_attempt=3,
        resume_from_iteration=None,
    )

    attempts = (run.metadata or {})["archived_timing_attempts"]
    assert set(attempts) == {"1", "2"}
    assert set(attempts["2"]["step_times"]) == {"1", "2"}
    assert run.step_times == {"1": {"start": 300, "end": 305, "duration_s": 5}}
    assert (run.metadata or {})["displayed_training_attempt"] == 3


def test_aggregate_async_step_times_returns_detailed_updates():
    _, substep_times = aggregate_async_step_times(
        [
            _async_event(
                0,
                TrainingSubstep.FORWARD_BACKWARD.value,
                "phase_start",
                10.25,
                occurrence_id=2,
                role="actor",
                rank=0,
                world_size=1,
            ),
            _async_event(
                0,
                TrainingSubstep.FORWARD_BACKWARD.value,
                "phase_finish",
                12.75,
                occurrence_id=2,
                role="actor",
                rank=0,
                world_size=1,
            ),
        ],
        1,
        training_attempt=1,
    )

    assert substep_times["1"][TrainingSubstep.FORWARD_BACKWARD.value] == {
        "start": 10.25,
        "duration_s": 2.5,
        "intervals": [
            {
                "step_id": 2,
                "training_role": "actor",
                "slowest_rank": 0,
                "reported_rank_count": 1,
                "training_world_size": 1,
                "start": 10.25,
                "duration_s": 2.5,
            }
        ],
    }


def test_aggregate_async_step_times_can_select_current_attempt():
    events = [
        _async_event(0, ROLLOUT_LOGGING, "phase_start", 10.0, training_attempt=1),
        _async_event(0, ROLLOUT_LOGGING, "phase_finish", 15.0, training_attempt=1),
        _async_event(14, ROLLOUT_LOGGING, "phase_start", 200.0, training_attempt=2),
        _async_event(14, ROLLOUT_LOGGING, "phase_finish", 205.0, training_attempt=2),
    ]

    _, substep_times = aggregate_async_step_times(
        events,
        15,
        training_attempt=2,
    )

    assert substep_times["1"] == {}
    assert substep_times["15"][ROLLOUT_LOGGING]["start"] == 200.0
    assert substep_times["15"][ROLLOUT_LOGGING]["duration_s"] == 5.0


def test_aggregate_async_step_times_keeps_described_custom_phases():
    events = [
        _async_event(
            0,
            "data_packing",
            event_type,
            timestamp,
            timeline_lane="training",
            parent_phase="training",
            display_name="Data packing",
        )
        for event_type, timestamp in (
            ("phase_start", 10.0),
            ("phase_finish", 15.0),
        )
    ]
    for role, start, finish in (("actor", 11.0, 13.0), ("critic", 11.5, 14.5)):
        events.extend(
            [
                _async_event(
                    0,
                    "policy_evaluation",
                    event_type,
                    timestamp,
                    occurrence_id=0,
                    role=role,
                    timeline_lane="training",
                    parent_phase="data_packing",
                    display_name="Policy evaluation",
                )
                for event_type, timestamp in (
                    ("phase_start", start),
                    ("phase_finish", finish),
                )
            ]
        )

    _, substep_times = aggregate_async_step_times(events, 1, training_attempt=1)

    assert substep_times["1"]["data_packing"] == {
        "start": 10.0,
        "duration_s": 5.0,
        "intervals": [
            {
                "start": 10.0,
                "duration_s": 5.0,
                "timeline_lane": "training",
                "parent_phase": "training",
                "display_name": "Data packing",
            }
        ],
    }
    assert substep_times["1"]["policy_evaluation"] == {
        "start": 11.0,
        "duration_s": 3.5,
        "intervals": [
            {
                "step_id": 0,
                "start": 11.0,
                "duration_s": 2.0,
                "training_role": "actor",
                "timeline_lane": "training",
                "parent_phase": "data_packing",
                "display_name": "Policy evaluation",
            },
            {
                "step_id": 0,
                "start": 11.5,
                "duration_s": 3.0,
                "training_role": "critic",
                "timeline_lane": "training",
                "parent_phase": "data_packing",
                "display_name": "Policy evaluation",
            },
        ],
    }


def test_aggregate_async_step_times_does_not_infer_missing_finishes():
    events = [
        _async_event(
            0,
            TrainingSubstep.FORWARD_BACKWARD.value,
            "phase_start",
            10.0,
            occurrence_id=0,
            role="actor",
        ),
        _async_event(
            0,
            OPTIMIZER_STEP,
            "phase_start",
            12.0,
            occurrence_id=0,
            role="actor",
        ),
        _async_event(
            0,
            OPTIMIZER_STEP,
            "phase_finish",
            13.0,
            occurrence_id=0,
            role="actor",
        ),
    ]

    _, substep_times = aggregate_async_step_times(
        events,
        1,
        training_attempt=1,
    )

    assert substep_times["1"][TrainingSubstep.FORWARD_BACKWARD.value] == {
        "start": 10.0,
        "duration_s": None,
    }
    assert "intervals" not in substep_times["1"][TrainingSubstep.FORWARD_BACKWARD.value]
    assert substep_times["1"][OPTIMIZER_STEP]["duration_s"] == 1.0


def test_aggregate_async_step_times_includes_incomplete_update_in_phase_start():
    events = [
        _async_event(
            0,
            TrainingSubstep.FORWARD_BACKWARD.value,
            "phase_start",
            8.0,
            occurrence_id=0,
            role="actor",
        ),
        _async_event(
            0,
            TrainingSubstep.FORWARD_BACKWARD.value,
            "phase_start",
            10.0,
            occurrence_id=1,
            role="actor",
        ),
        _async_event(
            0,
            TrainingSubstep.FORWARD_BACKWARD.value,
            "phase_finish",
            12.0,
            occurrence_id=1,
            role="actor",
        ),
    ]

    _, substep_times = aggregate_async_step_times(
        events,
        1,
        training_attempt=1,
    )

    assert substep_times["1"][TrainingSubstep.FORWARD_BACKWARD.value] == {
        "start": 10.0,
        "duration_s": 2.0,
        "intervals": [
            {
                "step_id": 1,
                "training_role": "actor",
                "start": 10.0,
                "duration_s": 2.0,
            }
        ],
    }


def test_aggregate_async_step_times_keeps_training_roles_separate():
    events = []
    for role, start, finish in (("actor", 10.0, 12.0), ("critic", 9.0, 13.0)):
        for event_type, timestamp in (
            ("phase_start", start),
            ("phase_finish", finish),
        ):
            events.append(
                _async_event(
                    0,
                    TrainingSubstep.FORWARD_BACKWARD.value,
                    event_type,
                    timestamp,
                    occurrence_id=0,
                    role=role,
                )
            )

    _, substep_times = aggregate_async_step_times(
        events,
        1,
        training_attempt=1,
    )
    assert substep_times["1"][TrainingSubstep.FORWARD_BACKWARD.value] == {
        "start": 9.0,
        "duration_s": 4.0,
        "intervals": [
            {
                "step_id": 0,
                "start": 10.0,
                "duration_s": 2.0,
                "training_role": "actor",
            },
            {
                "step_id": 0,
                "start": 9.0,
                "duration_s": 4.0,
                "training_role": "critic",
            },
        ],
    }


def test_aggregate_async_step_times_selects_slowest_reported_rank():
    events = []
    for rank, wall_start, wall_finish, monotonic_start, monotonic_finish in (
        (0, 10.0, 20.0, 100.0, 102.0),
        (1, 11.0, 14.0, 200.0, 205.0),
    ):
        events.extend(
            [
                _async_event(
                    0,
                    TrainingSubstep.FORWARD_BACKWARD.value,
                    event_type,
                    timestamp,
                    occurrence_id=0,
                    role="actor",
                    rank=rank,
                    world_size=2,
                    monotonic_timestamp=monotonic_timestamp,
                    timeline_lane="training",
                    parent_phase="training",
                )
                for event_type, timestamp, monotonic_timestamp in (
                    ("phase_start", wall_start, monotonic_start),
                    ("phase_finish", wall_finish, monotonic_finish),
                )
            ]
        )

    _, substep_times = aggregate_async_step_times(
        events,
        1,
        training_attempt=1,
    )

    assert substep_times["1"][TrainingSubstep.FORWARD_BACKWARD.value] == {
        "start": 11.0,
        "duration_s": 5.0,
        "intervals": [
            {
                "step_id": 0,
                "start": 11.0,
                "duration_s": 5.0,
                "training_role": "actor",
                "slowest_rank": 1,
                "reported_rank_count": 2,
                "training_world_size": 2,
                "timeline_lane": "training",
                "parent_phase": "training",
            }
        ],
    }


def test_aggregate_async_step_times_selects_rank_for_single_phase():
    events = []
    for rank, duration in ((0, 1.0), (1, 2.5)):
        events.extend(
            [
                _async_event(
                    0,
                    "value_inference",
                    event_type,
                    timestamp,
                    role="critic",
                    rank=rank,
                    world_size=4,
                    monotonic_timestamp=monotonic_timestamp,
                    timeline_lane="training",
                    parent_phase="training",
                )
                for event_type, timestamp, monotonic_timestamp in (
                    ("phase_start", 10.0 + rank, 100.0),
                    ("phase_finish", 10.0 + rank + duration, 100.0 + duration),
                )
            ]
        )

    _, substep_times = aggregate_async_step_times(events, 1, training_attempt=1)

    interval = substep_times["1"]["value_inference"]["intervals"][0]
    assert interval["duration_s"] == 2.5
    assert interval["slowest_rank"] == 1
    assert interval["reported_rank_count"] == 2
    assert interval["training_world_size"] == 4


def test_custom_reward_occurrences_are_stored_as_wall_clock_coverage():
    events = []
    for occurrence_id, start, finish in ((7, 10.0, 13.0), (8, 11.0, 12.0)):
        events.extend(
            [
                _async_event(
                    0,
                    "custom_reward",
                    event_type,
                    timestamp,
                    occurrence_id=occurrence_id,
                    timeline_lane="reward",
                    parent_phase="generate_rollouts",
                    display_name="Custom reward function",
                )
                for event_type, timestamp in (
                    ("phase_start", start),
                    ("phase_finish", finish),
                )
            ]
        )

    _, substep_times = aggregate_async_step_times(events, 1, training_attempt=1)

    assert substep_times["1"]["custom_reward"] == {
        "start": 10.0,
        "duration_s": 3.0,
        "intervals": [
            {
                "start": 10.0,
                "duration_s": 3.0,
                "timeline_lane": "reward",
                "parent_phase": "generate_rollouts",
                "display_name": "Custom reward function",
            }
        ],
    }


def test_async_substeps_keep_explicit_overlapping_durations():
    timing_events: list[AsyncTimingEvent] = []

    def record(
        rollout_id: int,
        timestamp: float,
        phase: str,
        event_type: AsyncTimingEventType,
        *,
        occurrence_id: int | None = None,
    ) -> None:
        timing_events.append(
            _async_event(
                rollout_id,
                phase,
                event_type,
                timestamp,
                occurrence_id=occurrence_id,
                role="actor" if occurrence_id is not None else None,
            )
        )

    record(0, 6.0, ROLLOUT_LOGGING, "phase_start")
    record(0, 10.0, "iteration", "rollout_start")
    record(0, 11.0, ROLLOUT_LOGGING, "phase_finish")
    record(1, 12.0, ROLLOUT_LOGGING, "phase_start")
    record(0, 11.0, SlimeStatus.TRAINING.value, "phase_start")
    record(
        0,
        11.1,
        SlimeStatus.COMPUTE_LOG_PROBS.value,
        "phase_start",
        occurrence_id=0,
    )
    record(
        0,
        11.8,
        SlimeStatus.COMPUTE_LOG_PROBS.value,
        "phase_finish",
        occurrence_id=0,
    )
    record(
        0,
        12.0,
        TrainingSubstep.FORWARD_BACKWARD.value,
        "phase_start",
        occurrence_id=0,
    )
    record(
        0,
        13.0,
        TrainingSubstep.FORWARD_BACKWARD.value,
        "phase_finish",
        occurrence_id=0,
    )
    record(0, 13.0, OPTIMIZER_STEP, "phase_start", occurrence_id=0)
    record(0, 13.5, OPTIMIZER_STEP, "phase_finish", occurrence_id=0)
    record(
        0,
        13.5,
        TrainingSubstep.FORWARD_BACKWARD.value,
        "phase_start",
        occurrence_id=1,
    )
    record(
        0,
        14.5,
        TrainingSubstep.FORWARD_BACKWARD.value,
        "phase_finish",
        occurrence_id=1,
    )
    record(0, 14.5, OPTIMIZER_STEP, "phase_start", occurrence_id=1)
    record(0, 15.0, OPTIMIZER_STEP, "phase_finish", occurrence_id=1)
    record(0, 15.0, SlimeStatus.TRAINING.value, "phase_finish")
    record(0, 15.0, CHECKPOINT_SAVE, "phase_start")
    record(0, 16.0, CHECKPOINT_SAVE, "phase_finish")
    record(1, 18.0, ROLLOUT_LOGGING, "phase_finish")
    record(0, 16.0, WEIGHT_SYNC, "phase_start")
    record(0, 20.0, WEIGHT_SYNC, "phase_finish")
    record(0, 20.0, EVAL_AFTER, "phase_start")
    record(0, 21.0, EVAL_AFTER, "phase_finish")
    record(0, 21.0, "iteration", "rollout_finish")

    record(1, 21.0, "iteration", "rollout_start")
    record(1, 22.0, SlimeStatus.TRAINING.value, "phase_start")
    record(
        1,
        22.1,
        SlimeStatus.COMPUTE_LOG_PROBS.value,
        "phase_start",
        occurrence_id=0,
    )
    record(
        1,
        22.3,
        SlimeStatus.COMPUTE_LOG_PROBS.value,
        "phase_finish",
        occurrence_id=0,
    )
    record(
        1,
        23.0,
        TrainingSubstep.FORWARD_BACKWARD.value,
        "phase_start",
        occurrence_id=0,
    )
    record(
        1,
        24.0,
        TrainingSubstep.FORWARD_BACKWARD.value,
        "phase_finish",
        occurrence_id=0,
    )
    record(1, 24.0, OPTIMIZER_STEP, "phase_start", occurrence_id=0)
    record(1, 25.0, OPTIMIZER_STEP, "phase_finish", occurrence_id=0)
    record(1, 25.0, SlimeStatus.TRAINING.value, "phase_finish")
    record(1, 25.0, "iteration", "rollout_finish")

    step_times, substep_times = aggregate_async_step_times(
        timing_events,
        2,
        training_attempt=1,
    )

    assert step_times["1"] == {"start": 10, "end": 21, "duration_s": 11}
    assert substep_times["1"] == {
        ROLLOUT_LOGGING: {
            "start": 6.0,
            "duration_s": 5.0,
            "intervals": [{"start": 6.0, "duration_s": 5.0}],
        },
        SlimeStatus.TRAINING.value: {
            "start": 11.0,
            "duration_s": 4.0,
            "intervals": [{"start": 11.0, "duration_s": 4.0}],
        },
        SlimeStatus.COMPUTE_LOG_PROBS.value: {
            "start": 11.1,
            "duration_s": 0.7,
            "intervals": [
                {
                    "step_id": 0,
                    "training_role": "actor",
                    "start": 11.1,
                    "duration_s": 0.7,
                }
            ],
        },
        TrainingSubstep.FORWARD_BACKWARD.value: {
            "start": 12.0,
            "duration_s": 2.0,
            "intervals": [
                {
                    "step_id": 0,
                    "training_role": "actor",
                    "start": 12.0,
                    "duration_s": 1.0,
                },
                {
                    "step_id": 1,
                    "training_role": "actor",
                    "start": 13.5,
                    "duration_s": 1.0,
                },
            ],
        },
        OPTIMIZER_STEP: {
            "start": 13.0,
            "duration_s": 1.0,
            "intervals": [
                {
                    "step_id": 0,
                    "training_role": "actor",
                    "start": 13.0,
                    "duration_s": 0.5,
                },
                {
                    "step_id": 1,
                    "training_role": "actor",
                    "start": 14.5,
                    "duration_s": 0.5,
                },
            ],
        },
        CHECKPOINT_SAVE: {
            "start": 15.0,
            "duration_s": 1.0,
            "intervals": [{"start": 15.0, "duration_s": 1.0}],
        },
        WEIGHT_SYNC: {
            "start": 16.0,
            "duration_s": 4.0,
            "intervals": [{"start": 16.0, "duration_s": 4.0}],
        },
        EVAL_AFTER: {
            "start": 20.0,
            "duration_s": 1.0,
            "intervals": [{"start": 20.0, "duration_s": 1.0}],
        },
    }
    assert step_times["2"] == {"start": 21, "end": 25, "duration_s": 4}
    assert substep_times["2"][ROLLOUT_LOGGING] == {
        "start": 12.0,
        "duration_s": 6.0,
        "intervals": [{"start": 12.0, "duration_s": 6.0}],
    }
    assert substep_times["2"][SlimeStatus.COMPUTE_LOG_PROBS.value] == {
        "start": 22.1,
        "duration_s": 0.2,
        "intervals": [
            {
                "step_id": 0,
                "training_role": "actor",
                "start": 22.1,
                "duration_s": 0.2,
            }
        ],
    }
