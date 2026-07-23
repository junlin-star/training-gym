from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Any, TypeAlias

from modal_training_gym.common.status import SlimeStatus

if TYPE_CHECKING:
    from modal_training_gym.common.run import TrainingRun

TrainingPhaseInterval: TypeAlias = dict[str, int | float | str]
SubstepTiming: TypeAlias = dict[str, float | None | list[TrainingPhaseInterval]]
SubstepTimes: TypeAlias = dict[str, dict[str, SubstepTiming]]
StepTimes: TypeAlias = dict[str, dict[str, int | float | None]]

TRAINING_TIMING_EVENT_BATCH_KIND = "timing_event_batch"
TRAINING_TIMING_ATTEMPT_CLOSED_KIND = "timing_attempt_closed"
AGGREGATED_TRAINING_STEP_TIMING_KIND = "aggregated_training_step_timing"
TRAINING_ROLE_FINISH_EVENT = "training_role_finish"
SUBSTEP_FINISH_EVENT = "substep_finish"
SYNCHRONOUS_TRAINING_ROLES = ("driver", "rollout", "actor", "critic")
SYNCHRONOUS_TIMING_ATTEMPT_KEY = "synchronous_step_timings_persisted_attempt"
SYNCHRONOUS_TIMING_WATERMARK_KEY = "synchronous_step_timings_persisted_through"
TRAINING_STEP_TIMING_ATTEMPTS_KEY = "training_step_timing_attempts"


class Substep(str, Enum):
    EVAL_BEFORE = SlimeStatus.EVAL_ROLLOUT_LOGGING.value
    GENERATE_ROLLOUTS = SlimeStatus.ROLLOUT_LOGGING.value
    OFFLOAD_ROLLOUT = SlimeStatus.OFFLOAD_ROLLOUT.value
    COMPUTE_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
    TRAIN_MODEL = SlimeStatus.TRAIN_MODEL.value
    OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
    CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
    OFFLOAD_TRAIN = SlimeStatus.OFFLOAD_TRAIN.value
    WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value
    EVAL_AFTER = f"{SlimeStatus.EVAL_ROLLOUT_LOGGING.value}_end"


SYNC_SUBSTEP_ORDER = [substep.value for substep in Substep]
SYNC_OPTIONAL_SUBSTEPS = {
    Substep.EVAL_BEFORE.value,
    Substep.OFFLOAD_ROLLOUT.value,
    Substep.CHECKPOINT_SAVE.value,
    Substep.OFFLOAD_TRAIN.value,
    Substep.EVAL_AFTER.value,
}


def training_timing_event_batch_key(
    training_run_id: str,
    training_attempt: int,
    training_role: str,
    rollout_id: int,
) -> tuple[str, str, int, str, int]:
    return (
        training_run_id,
        TRAINING_TIMING_EVENT_BATCH_KIND,
        training_attempt,
        training_role,
        rollout_id,
    )


def training_timing_event_batch_keys(
    training_run_id: str,
    training_attempt: int,
    num_steps: int,
) -> list[tuple[str, str, int, str, int]]:
    return [
        training_timing_event_batch_key(
            training_run_id,
            training_attempt,
            role,
            step - 1,
        )
        for step in range(1, num_steps + 1)
        for role in SYNCHRONOUS_TRAINING_ROLES
    ]


def training_timing_attempt_closed_key(
    training_run_id: str,
    training_attempt: int,
) -> tuple[str, str, int]:
    return (
        training_run_id,
        TRAINING_TIMING_ATTEMPT_CLOSED_KIND,
        training_attempt,
    )


def aggregated_training_step_timing_key(
    training_run_id: str,
    training_attempt: int,
    step: int,
) -> tuple[str, str, int, int]:
    return (
        training_run_id,
        AGGREGATED_TRAINING_STEP_TIMING_KIND,
        training_attempt,
        step,
    )


def aggregated_training_step_timing_keys(
    training_run_id: str,
    training_attempt: int,
    num_steps: int,
    *,
    first_step: int = 1,
) -> list[tuple[str, str, int, int]]:
    return [
        aggregated_training_step_timing_key(
            training_run_id,
            training_attempt,
            step,
        )
        for step in range(max(first_step, 1), num_steps + 1)
    ]


def build_aggregated_training_step_timing(
    training_run_id: str,
    training_attempt: int,
    step: int,
    step_time: Mapping[str, Any],
    substep_times: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "training_run_id": training_run_id,
        "training_attempt": training_attempt,
        "step": step,
        "rollout_id": step - 1,
        "source": source,
        "step_time": dict(step_time),
        "substep_times": dict(substep_times),
    }


def timing_maps_from_aggregated_steps(
    aggregated_steps: Iterable[Mapping[str, Any]],
) -> tuple[StepTimes, SubstepTimes]:
    step_times: StepTimes = {}
    substep_times: SubstepTimes = {}
    for aggregated_step in aggregated_steps:
        try:
            step = int(aggregated_step["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if step <= 0:
            continue
        step_key = str(step)
        step_time = aggregated_step.get("step_time")
        if isinstance(step_time, Mapping):
            step_times[step_key] = dict(step_time)
        phases = aggregated_step.get("substep_times")
        if isinstance(phases, Mapping):
            substep_times[step_key] = {
                str(phase): dict(timing)
                for phase, timing in phases.items()
                if isinstance(timing, Mapping)
            }
    return step_times, substep_times


def normalize_persisted_step_timing_keys(
    step_times: Mapping[str, Any],
    substep_times: Mapping[str, Any],
) -> tuple[StepTimes, SubstepTimes]:
    """Normalize legacy rollout-id keys to the current one-based step keys."""
    uses_rollout_ids = "0" in step_times or "0" in substep_times
    key_offset = 1 if uses_rollout_ids else 0

    normalized_step_times: StepTimes = {}
    for raw_step, timing in step_times.items():
        try:
            step = int(raw_step) + key_offset
        except (TypeError, ValueError):
            continue
        if step > 0 and isinstance(timing, Mapping):
            normalized_step_times[str(step)] = dict(timing)

    normalized_substep_times: SubstepTimes = {}
    for raw_step, phases in substep_times.items():
        try:
            step = int(raw_step) + key_offset
        except (TypeError, ValueError):
            continue
        if step <= 0 or not isinstance(phases, Mapping):
            continue
        normalized_substep_times[str(step)] = {
            str(phase): dict(timing)
            for phase, timing in phases.items()
            if isinstance(timing, Mapping)
        }
    return normalized_step_times, normalized_substep_times


def contiguous_timing_step_watermark(
    step_times: Mapping[str, Any],
    substep_times: Mapping[str, Any],
    *,
    max_step: int,
) -> int:
    persisted_steps = step_times.keys() | substep_times.keys()
    return next(
        (
            step - 1
            for step in range(1, max_step + 1)
            if str(step) not in persisted_steps
        ),
        max_step,
    )


def legacy_step_time_keys(
    training_run_id: str,
    num_steps: int,
) -> list[str]:
    suffixes = (
        "start",
        "finish",
        "substep_start",
        SUBSTEP_FINISH_EVENT,
        *(f"substep:{phase}" for phase in SYNC_SUBSTEP_ORDER),
    )
    return [
        f"{training_run_id}:{step}:{suffix}"
        for step in range(1, num_steps + 1)
        for suffix in suffixes
    ]


def record_step_time_event(
    step_times: MutableMapping[str, Any],
    training_run_id: str,
    current_step: Any,
    phase: str,
    step_event: str,
    event_ts: float,
) -> None:
    if not (isinstance(current_step, int) and current_step > 0):
        return
    event_time = round(float(event_ts), 3)
    step_window_start_key = f"{training_run_id}:{current_step}:substep_start"

    def record_first_in_step_window(
        key: str, *, allow_before_step_window: bool = False
    ) -> None:
        existing = step_times.get(key)
        raw_step_window_start = step_times.get(step_window_start_key)
        step_window_start = (
            float(raw_step_window_start) if raw_step_window_start is not None else None
        )
        if existing is not None and (
            step_window_start is None or float(existing) >= step_window_start
        ):
            return

        recorded_time = event_time
        if step_window_start is not None and not allow_before_step_window:
            recorded_time = max(recorded_time, step_window_start)
        step_times[key] = recorded_time

    if step_event == "start":
        step_times[f"{training_run_id}:{current_step}:start"] = event_time
    elif step_event == "finish":
        step_times[f"{training_run_id}:{current_step}:finish"] = event_time

    if step_event == "substep_start":
        step_times[step_window_start_key] = event_time
    elif step_event == SUBSTEP_FINISH_EVENT:
        record_first_in_step_window(f"{training_run_id}:{current_step}:substep_finish")
    elif step_event in ("eval_begin", "eval_end"):
        substep = (
            Substep.EVAL_BEFORE if step_event == "eval_begin" else Substep.EVAL_AFTER
        )
        record_first_in_step_window(
            f"{training_run_id}:{current_step}:substep:{substep.value}",
            allow_before_step_window=step_event == "eval_begin",
        )
    elif not step_event and phase == Substep.EVAL_BEFORE.value:
        pass
    elif step_event == "finish":
        pass
    else:
        record_first_in_step_window(f"{training_run_id}:{current_step}:substep:{phase}")


def build_step_event_times(
    training_timing_events: Iterable[Mapping[str, Any]],
    *,
    training_attempt: int,
) -> dict[str, float]:
    step_event_times: dict[str, float] = {}
    parsed_events: list[tuple[float, str, int, str, str]] = []
    for event in training_timing_events:
        try:
            event_attempt = int(event["training_attempt"])
            step_event = str(event.get("step_event", ""))
            if event_attempt != training_attempt or step_event in {
                "phase_start",
                "phase_finish",
                TRAINING_ROLE_FINISH_EVENT,
            }:
                continue
            parsed_events.append(
                (
                    float(event["event_ts"]),
                    str(event["training_run_id"]),
                    int(event["progress_current"]),
                    str(event["phase"]),
                    step_event,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    for event_ts, run_id, current_step, phase, step_event in sorted(parsed_events):
        record_step_time_event(
            step_event_times,
            run_id,
            current_step,
            phase,
            step_event,
            event_ts,
        )
    return step_event_times


def aggregate_step_times(
    step_times_dict: Mapping[str, Any],
    run_id: str,
    num_steps: int,
    substep_order: list[str] | None = None,
    optional_substeps: set[str] | None = None,
    *,
    training_timing_events: Iterable[Mapping[str, Any]] = (),
    training_attempt: int | None = None,
    first_step: int = 1,
) -> tuple[StepTimes, SubstepTimes]:
    """Organize synchronous step and substep timings."""
    order = SYNC_SUBSTEP_ORDER if substep_order is None else substep_order
    optional = (
        SYNC_OPTIONAL_SUBSTEPS if optional_substeps is None else optional_substeps
    )
    step_times: StepTimes = {}
    substep_times: SubstepTimes = {}
    training_intervals = aggregate_training_time_intervals(
        training_timing_events,
        training_attempt=training_attempt,
        first_step=first_step,
        num_steps=num_steps,
    )

    for current_step_num in range(max(first_step, 1), num_steps + 1):
        step_key = str(current_step_num)
        start_key = f"{run_id}:{current_step_num}:start"
        finish_key = f"{run_id}:{current_step_num}:finish"

        step_start_time = step_times_dict.get(start_key)
        step_end_time = step_times_dict.get(finish_key)
        if step_start_time is not None:
            step_start_time = float(step_start_time)
        if step_end_time is not None:
            step_end_time = float(step_end_time)

        start_time = int(step_start_time) if step_start_time is not None else None
        end_time = int(step_end_time) if step_end_time is not None else None
        duration = None
        if start_time is not None and end_time is not None:
            duration = end_time - start_time

        step_window_start = step_times_dict.get(
            f"{run_id}:{current_step_num}:substep_start"
        )
        if step_window_start is not None:
            step_window_start = float(step_window_start)
        full_step_start_time = (
            step_window_start if step_window_start is not None else step_start_time
        )
        full_step_end_time = step_times_dict.get(
            f"{run_id}:{current_step_num}:substep_finish"
        )
        if full_step_end_time is not None:
            full_step_end_time = float(full_step_end_time)
            if step_window_start is not None and full_step_end_time < step_window_start:
                full_step_end_time = step_end_time
        else:
            full_step_end_time = step_end_time

        full_step_duration = None
        if full_step_start_time is not None and full_step_end_time is not None:
            full_step_duration = round(
                max(full_step_end_time - full_step_start_time, 0.0), 3
            )

        step_times[step_key] = {
            "start": start_time,
            "end": end_time,
            "duration_s": duration,
            "full_step_duration_s": full_step_duration,
        }

        substep_times[step_key] = {}
        eval_before = Substep.EVAL_BEFORE.value
        present: set[str] = set()
        recorded: list[tuple[float, int, str]] = []
        for order_idx, substep in enumerate(order):
            interval_timing = training_intervals.get(step_key, {}).get(substep)
            substep_start = (
                interval_timing.get("start")
                if interval_timing is not None
                else step_times_dict.get(
                    f"{run_id}:{current_step_num}:substep:{substep}"
                )
            )
            if substep_start is None:
                continue
            substep_start = float(substep_start)
            if (
                step_window_start is not None
                and substep_start < step_window_start
                and substep != eval_before
            ):
                continue
            if full_step_start_time is not None and substep != eval_before:
                substep_start = max(substep_start, full_step_start_time)
            if full_step_end_time is not None:
                substep_start = min(substep_start, full_step_end_time)
            present.add(substep)
            recorded.append((substep_start, order_idx, substep))
        recorded.sort()

        for idx, (substep_start, order_idx, substep) in enumerate(recorded):
            if idx + 1 < len(recorded):
                next_start, next_idx = recorded[idx + 1][0], recorded[idx + 1][1]
            else:
                next_start, next_idx = full_step_end_time, len(order)

            gap = order[order_idx + 1 : next_idx]
            dropped_mandatory = any(
                phase not in optional and phase not in present for phase in gap
            )
            if next_start is None or dropped_mandatory:
                substep_duration = None
            else:
                substep_duration = round(max(next_start - substep_start, 0.0), 3)

            substep_times[step_key][substep] = {
                "start": round(substep_start, 3),
                "duration_s": substep_duration,
            }
        substep_times[step_key].update(training_intervals.get(step_key, {}))
    return step_times, substep_times


def aggregate_training_time_intervals(
    training_timing_events: Iterable[Mapping[str, Any]],
    *,
    training_attempt: int | None,
    first_step: int,
    num_steps: int,
) -> SubstepTimes:
    intervals: dict[
        tuple[int, str], dict[tuple[str, int, int | None], dict[str, Any]]
    ] = {}
    for event in training_timing_events:
        try:
            attempt = int(event.get("training_attempt", 0) or 0)
            step = int(event["progress_current"])
            phase = str(event["phase"])
            boundary = str(event["step_event"])
            step_id = int(event.get("step_id", -1))
            event_ts = float(event["event_ts"])
            training_rank = event.get("training_rank")
            training_rank = int(training_rank) if training_rank is not None else None
        except (KeyError, TypeError, ValueError):
            continue
        if training_attempt is not None and attempt != training_attempt:
            continue
        if step < first_step or step > num_steps:
            continue
        if boundary not in ("phase_start", "phase_finish"):
            continue
        role = str(event.get("training_role", "driver"))
        update = intervals.setdefault((step, phase), {}).setdefault(
            (role, step_id, training_rank), {}
        )
        update[boundary.removeprefix("phase_")] = event_ts
        for key in (
            "active_duration_s",
            "training_world_size",
            "timeline_lane",
            "parent_phase",
            "display_name",
        ):
            if event.get(key) is not None:
                update[key] = event[key]

    aggregated: SubstepTimes = {}
    for (step, phase), updates in intervals.items():
        phase_intervals: list[TrainingPhaseInterval] = []
        for (role, step_id, training_rank), update in sorted(
            updates.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2] is None,
                item[0][2] if item[0][2] is not None else -1,
            ),
        ):
            start = update.get("start")
            finish = update.get("finish")
            if not isinstance(start, (int, float)) or not isinstance(
                finish, (int, float)
            ):
                continue
            if finish < start:
                continue
            active_duration = update.get("active_duration_s")
            if (
                not isinstance(active_duration, (int, float))
                or not isfinite(active_duration)
                or active_duration < 0
            ):
                active_duration = None
            duration = (
                float(active_duration)
                if active_duration is not None
                else float(finish - start)
            )
            interval: TrainingPhaseInterval = {
                "step_id": step_id,
                "start": round(float(start), 3),
                "duration_s": round(duration, 3),
            }
            if active_duration is not None:
                interval["active_duration_s"] = round(duration, 3)
            if role:
                interval["training_role"] = role
            if training_rank is not None:
                interval["training_rank"] = training_rank
            for key in (
                "training_world_size",
                "timeline_lane",
                "parent_phase",
                "display_name",
            ):
                value = update.get(key)
                if isinstance(value, int | str):
                    interval[key] = value
            phase_intervals.append(interval)
        if not phase_intervals:
            continue
        aggregated.setdefault(str(step), {})[phase] = {
            "start": min(float(interval["start"]) for interval in phase_intervals),
            "duration_s": round(
                sum(float(interval["duration_s"]) for interval in phase_intervals), 3
            ),
            "intervals": phase_intervals,
        }
    return aggregated


def aggregate_completed_step_times(
    training_timing_events: Iterable[Mapping[str, Any]],
    run_id: str,
    num_steps: int,
    *,
    training_attempt: int,
    first_step: int = 1,
) -> tuple[StepTimes, SubstepTimes]:
    events = list(training_timing_events)
    step_event_times = build_step_event_times(events, training_attempt=training_attempt)
    step_times, substep_times = aggregate_step_times(
        step_event_times,
        run_id,
        num_steps,
        training_timing_events=events,
        training_attempt=training_attempt,
        first_step=first_step,
    )
    completed_steps = {
        str(step)
        for step in range(max(first_step, 1), num_steps + 1)
        if step_event_times.get(f"{run_id}:{step}:substep_finish") is not None
    }
    return (
        {
            step: timing
            for step, timing in step_times.items()
            if step in completed_steps
        },
        {
            step: timing
            for step, timing in substep_times.items()
            if step in completed_steps
        },
    )


def merge_step_times(
    run: TrainingRun,
    new_step_times: StepTimes,
    new_substep_times: SubstepTimes,
) -> None:
    """Merge completed timings into the displayed training attempt."""
    displayed_steps = dict(run.step_times or {})
    displayed_substeps = dict(run.substep_times or {})
    for step, timing in new_step_times.items():
        displayed_timing = dict(displayed_steps.get(step, {}))
        displayed_timing.update(
            {
                key: value
                for key, value in timing.items()
                if value is not None or key not in displayed_timing
            }
        )
        displayed_steps[step] = displayed_timing

    for step, phases in new_substep_times.items():
        displayed_phases = dict(displayed_substeps.get(step, {}))
        for phase, timing in phases.items():
            existing = displayed_phases.get(phase)
            existing_intervals = (
                existing.get("intervals") if isinstance(existing, Mapping) else None
            )
            new_intervals = timing.get("intervals")
            existing_quality = (
                len(existing_intervals) if isinstance(existing_intervals, list) else 0,
                isinstance(existing, Mapping)
                and existing.get("duration_s") is not None,
            )
            new_quality = (
                len(new_intervals) if isinstance(new_intervals, list) else 0,
                timing.get("duration_s") is not None,
            )
            if existing is None or new_quality >= existing_quality:
                displayed_phases[phase] = timing
        displayed_substeps[step] = displayed_phases
    run.step_times = displayed_steps
    run.substep_times = displayed_substeps


def synchronous_timing_watermark(run: TrainingRun, training_attempt: int) -> int:
    metadata = run.metadata or {}
    try:
        watermark_attempt = int(metadata.get(SYNCHRONOUS_TIMING_ATTEMPT_KEY) or 0)
        persisted_through_step = int(
            metadata.get(SYNCHRONOUS_TIMING_WATERMARK_KEY) or 0
        )
    except (TypeError, ValueError):
        return 0
    return persisted_through_step if watermark_attempt == training_attempt else 0


def advance_synchronous_timing_watermark(
    run: TrainingRun,
    training_attempt: int,
    *,
    completed_through_step: int = 0,
) -> int:
    """Advance through persisted rows and steps already covered by a checkpoint."""
    persisted_through_step = max(
        synchronous_timing_watermark(run, training_attempt),
        completed_through_step,
    )
    completed_steps = run.step_times or {}
    while str(persisted_through_step + 1) in completed_steps:
        persisted_through_step += 1

    metadata = dict(run.metadata or {})
    metadata[SYNCHRONOUS_TIMING_ATTEMPT_KEY] = training_attempt
    metadata[SYNCHRONOUS_TIMING_WATERMARK_KEY] = persisted_through_step
    run.metadata = metadata
    return persisted_through_step
