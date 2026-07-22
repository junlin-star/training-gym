from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, TypedDict, cast

from modal_training_gym.common.run import (
    SingleSubstepTiming,
    StepTimes,
    SubstepTimes,
    SubstepTimingInterval,
    TrainingRun,
)


class _PhaseMetadata(TypedDict, total=False):
    timeline_lane: str
    parent_phase: str
    display_name: str


class _AggregatedTimingEvent(TypedDict):
    rollout_step: int
    phase: str
    step_event: str
    event_ts: float
    event_monotonic: float | None
    update_id: int | None
    training_attempt: int
    training_role: str
    training_rank: int | None
    training_world_size: int | None


def aggregate_async_step_times(
    events: Iterable[Mapping[str, Any]],
    num_steps: int,
    substep_order: list[str],
    *,
    attempt: int | None = None,
    selected_step: int | None = None,
) -> tuple[StepTimes, SubstepTimes]:
    """Aggregate async events directly into stored timing metadata."""
    aggregated_events: list[_AggregatedTimingEvent] = []
    phase_metadata: dict[tuple[int, str], _PhaseMetadata] = {}
    for stored_event in events:
        stored_batch = stored_event.get("events")
        batch_events = (
            stored_batch if isinstance(stored_batch, list) else [stored_event]
        )
        for event in batch_events:
            if not isinstance(event, Mapping):
                continue
            try:
                rollout_step = int(
                    event.get("progress_current", stored_event["progress_current"])
                )
                event_ts = float(event["event_ts"])
                event_attempt = int(
                    event.get(
                        "training_attempt",
                        stored_event.get("training_attempt", 0),
                    )
                    or 0
                )
                update_id = event.get("step_id")
                update_id = int(update_id) if update_id is not None else None
                training_rank = event.get(
                    "training_rank", stored_event.get("training_rank")
                )
                training_rank = (
                    int(training_rank) if training_rank is not None else None
                )
                training_world_size = event.get(
                    "training_world_size", stored_event.get("training_world_size")
                )
                training_world_size = (
                    int(training_world_size)
                    if training_world_size is not None
                    else None
                )
            except (KeyError, TypeError, ValueError):
                continue
            if attempt is not None and event_attempt != attempt:
                continue
            if selected_step is not None and rollout_step != selected_step:
                continue
            phase = event.get("phase")
            step_event = event.get("step_event")
            training_role = event.get(
                "training_role", stored_event.get("training_role", "")
            )
            if (
                not math.isfinite(event_ts)
                or not isinstance(phase, str)
                or not isinstance(step_event, str)
                or not isinstance(training_role, str)
            ):
                continue
            event_monotonic = event.get("event_monotonic")
            try:
                event_monotonic = (
                    float(event_monotonic) if event_monotonic is not None else None
                )
            except (TypeError, ValueError):
                event_monotonic = None
            if event_monotonic is not None and not math.isfinite(event_monotonic):
                event_monotonic = None
            aggregated_events.append(
                {
                    "rollout_step": rollout_step,
                    "phase": phase,
                    "step_event": step_event,
                    "event_ts": event_ts,
                    "event_monotonic": event_monotonic,
                    "update_id": update_id,
                    "training_attempt": event_attempt,
                    "training_role": training_role,
                    "training_rank": training_rank,
                    "training_world_size": training_world_size,
                }
            )
            metadata: _PhaseMetadata = {}
            if isinstance(timeline_lane := event.get("timeline_lane"), str):
                metadata["timeline_lane"] = timeline_lane
            if isinstance(parent_phase := event.get("parent_phase"), str):
                metadata["parent_phase"] = parent_phase
            if isinstance(display_name := event.get("display_name"), str):
                metadata["display_name"] = display_name
            if metadata:
                phase_metadata[rollout_step, phase] = metadata

    step_timestamps: dict[int, dict[str, float]] = {}
    phase_occurrences: dict[
        tuple[int, str],
        dict[
            tuple[str, int | None, int | None],
            dict[str, _AggregatedTimingEvent],
        ],
    ] = {}
    for event in sorted(aggregated_events, key=lambda event: event["event_ts"]):
        rollout_step = event["rollout_step"]
        phase = event["phase"]
        step_event = event["step_event"]
        event_ts = event["event_ts"]
        if step_event in ("start", "finish"):
            timestamps = step_timestamps.setdefault(rollout_step, {})
            if step_event == "start":
                timestamps[step_event] = min(
                    timestamps.get(step_event, event_ts), event_ts
                )
            else:
                timestamps[step_event] = max(
                    timestamps.get(step_event, event_ts), event_ts
                )
        elif step_event in ("phase_start", "phase_finish"):
            timestamp_kind = step_event.removeprefix("phase_")
            phase_occurrences.setdefault((rollout_step, phase), {}).setdefault(
                (
                    event["training_role"],
                    event["update_id"],
                    event["training_rank"],
                ),
                {},
            )[timestamp_kind] = event

    step_times: StepTimes = {}
    substep_times: SubstepTimes = {}
    phases = [
        *substep_order,
        *sorted(
            {event["phase"] for event in aggregated_events}.difference(substep_order)
        ),
    ]
    rollout_steps = (
        range(1, num_steps + 1) if selected_step is None else (selected_step,)
    )
    for current_step in rollout_steps:
        step_key = str(current_step)
        timestamps = step_timestamps.get(current_step, {})
        start = int(timestamps["start"]) if "start" in timestamps else None
        end = int(timestamps["finish"]) if "finish" in timestamps else None
        step_times[step_key] = {
            "start": start,
            "end": end,
            "duration_s": end - start
            if start is not None and end is not None
            else None,
        }
        substep_times[step_key] = {}

        for phase in phases:
            intervals: list[SubstepTimingInterval] = []
            phase_starts: list[float] = []
            completed_duration = 0.0
            has_completed_occurrence = False
            logical_occurrences: dict[
                tuple[str, int | None],
                list[
                    tuple[
                        int | None,
                        dict[str, _AggregatedTimingEvent],
                    ]
                ],
            ] = {}
            for (
                training_role,
                update_id,
                training_rank,
            ), occurrence in phase_occurrences.get((current_step, phase), {}).items():
                logical_occurrences.setdefault((training_role, update_id), []).append(
                    (training_rank, occurrence)
                )

            for (training_role, update_id), occurrences in sorted(
                logical_occurrences.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1] is None,
                    item[0][1] if item[0][1] is not None else -1,
                ),
            ):
                ranked_occurrences = [
                    occurrence
                    for occurrence in occurrences
                    if occurrence[0] is not None
                ]
                if ranked_occurrences:
                    occurrences = ranked_occurrences
                completed_occurrences = []
                for training_rank, occurrence in occurrences:
                    start_event = occurrence.get("start")
                    finish_event = occurrence.get("finish")
                    if start_event is not None:
                        phase_starts.append(start_event["event_ts"])
                    if start_event is None or finish_event is None:
                        continue
                    start_monotonic = start_event["event_monotonic"]
                    finish_monotonic = finish_event["event_monotonic"]
                    duration = (
                        finish_monotonic - start_monotonic
                        if start_monotonic is not None and finish_monotonic is not None
                        else finish_event["event_ts"] - start_event["event_ts"]
                    )
                    if duration < 0:
                        continue
                    completed_occurrences.append(
                        (duration, training_rank, start_event, finish_event)
                    )
                if not completed_occurrences:
                    continue
                duration, slowest_rank, start_event, _ = max(
                    completed_occurrences,
                    key=lambda occurrence: (
                        occurrence[0],
                        -(occurrence[1] if occurrence[1] is not None else 0),
                    ),
                )
                completed_duration += duration
                has_completed_occurrence = True
                if update_id is None and not training_role and slowest_rank is None:
                    continue
                interval: SubstepTimingInterval = {
                    "start": round(start_event["event_ts"], 3),
                    "duration_s": round(duration, 3),
                }
                if update_id is not None:
                    interval["step_id"] = update_id
                if training_role:
                    interval["training_role"] = training_role
                if slowest_rank is not None:
                    interval["slowest_rank"] = slowest_rank
                    interval["reported_rank_count"] = len(
                        {
                            rank
                            for _, rank, _, _ in completed_occurrences
                            if rank is not None
                        }
                    )
                    world_sizes = [
                        event["training_world_size"]
                        for _, _, start, finish in completed_occurrences
                        for event in (start, finish)
                        if event["training_world_size"] is not None
                    ]
                    if world_sizes:
                        interval["training_world_size"] = max(world_sizes)
                intervals.append(interval)

            if phase_starts:
                phase_start = min(phase_starts)
                duration = (
                    round(completed_duration, 3) if has_completed_occurrence else None
                )
            else:
                continue

            timing: SingleSubstepTiming = {
                "start": round(phase_start, 3),
                "duration_s": duration,
            }
            metadata = phase_metadata.get((current_step, phase), {})
            if metadata and duration is not None:
                if not intervals:
                    intervals.append(
                        {
                            "start": round(phase_start, 3),
                            "duration_s": duration,
                        }
                    )
                for interval in intervals:
                    if "timeline_lane" in metadata:
                        interval["timeline_lane"] = metadata["timeline_lane"]
                    if "parent_phase" in metadata:
                        interval["parent_phase"] = metadata["parent_phase"]
                    if "display_name" in metadata:
                        interval["display_name"] = metadata["display_name"]
            if intervals:
                timing["intervals"] = intervals
            substep_times[step_key][phase] = timing

    return step_times, substep_times


def reconcile_attempt_step_times(
    run_record: TrainingRun,
    new_step_times: StepTimes,
    new_substep_times: SubstepTimes,
    *,
    training_attempt: int,
    resumed_from_checkpoint: bool,
) -> None:
    current_step_times = {
        step: timing
        for step, timing in new_step_times.items()
        if any(value is not None for value in timing.values())
    }
    current_substep_times = {
        step: timings for step, timings in new_substep_times.items() if timings
    }
    steps_with_current_timing = current_step_times.keys() | current_substep_times.keys()
    if not steps_with_current_timing:
        return

    metadata = dict(run_record.metadata or {})
    previous_substep_times: SubstepTimes = {
        step: {
            phase: cast(SingleSubstepTiming, dict(timing))
            for phase, timing in timings.items()
        }
        for step, timings in (run_record.substep_times or {}).items()
    }
    legacy_intervals = metadata.pop("substep_timing_intervals", {})
    if isinstance(legacy_intervals, Mapping):
        for step, phases in legacy_intervals.items():
            if not isinstance(step, str) or not isinstance(phases, Mapping):
                continue
            for phase, intervals in phases.items():
                if not isinstance(phase, str) or not isinstance(intervals, list):
                    continue
                timing = previous_substep_times.get(step, {}).get(phase)
                if timing is not None:
                    timing["intervals"] = intervals

    previous_step_times = dict(run_record.step_times or {})
    timing_attempts: dict[str, Any] = {}
    if training_attempt > 1:
        stored_attempts = metadata.get("timing_attempts")
        timing_attempts = (
            dict(stored_attempts) if isinstance(stored_attempts, Mapping) else {}
        )
        previous_attempt = str(training_attempt - 1)
        if previous_attempt not in timing_attempts and (
            previous_step_times or previous_substep_times
        ):
            timing_attempts[previous_attempt] = {
                "step_times": previous_step_times,
                "substep_times": previous_substep_times,
            }

    if resumed_from_checkpoint:
        first_new_step = min(int(step) for step in steps_with_current_timing)
        displayed_step_times = {
            step: timing
            for step, timing in previous_step_times.items()
            if int(step) < first_new_step
        }
        displayed_substep_times = {
            step: timing
            for step, timing in previous_substep_times.items()
            if int(step) < first_new_step
        }
        displayed_step_times.update(current_step_times)
        displayed_substep_times.update(current_substep_times)
    else:
        displayed_step_times = current_step_times
        displayed_substep_times = current_substep_times

    if training_attempt > 1:
        timing_attempts[str(training_attempt)] = {
            "step_times": displayed_step_times,
            "substep_times": displayed_substep_times,
        }
        metadata["timing_attempts"] = timing_attempts

    run_record.step_times = displayed_step_times
    run_record.substep_times = displayed_substep_times
    run_record.metadata = metadata


def reconcile_completed_step_times(
    run_record: TrainingRun,
    step_times: StepTimes,
    substep_times: SubstepTimes,
    *,
    training_attempt: int,
    resumed_from_checkpoint: bool,
) -> None:
    stored_attempts = (run_record.metadata or {}).get("timing_attempts")
    continuing_attempt = training_attempt <= 1 or (
        isinstance(stored_attempts, Mapping)
        and str(training_attempt) in stored_attempts
    )
    if continuing_attempt:
        step_times = {**(run_record.step_times or {}), **step_times}
        substep_times = {**(run_record.substep_times or {}), **substep_times}
    reconcile_attempt_step_times(
        run_record,
        step_times,
        substep_times,
        training_attempt=training_attempt,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )


def apply_step_timing_snapshot(
    run_record: TrainingRun,
    snapshot: Mapping[str, Any],
    *,
    snapshot_attempt: int,
    training_attempt: int,
) -> None:
    snapshot_step_times = snapshot.get("step_times") or {}
    snapshot_substep_times = snapshot.get("substep_times") or {}
    if snapshot_attempt < training_attempt:
        run_record.step_times = {
            **(run_record.step_times or {}),
            **snapshot_step_times,
        }
        run_record.substep_times = {
            **(run_record.substep_times or {}),
            **snapshot_substep_times,
        }
    else:
        run_record.step_times = snapshot_step_times
        run_record.substep_times = snapshot_substep_times
    if "timing_attempts" in snapshot:
        metadata = dict(run_record.metadata or {})
        metadata["timing_attempts"] = {
            **(metadata.get("timing_attempts") or {}),
            **snapshot["timing_attempts"],
        }
        run_record.metadata = metadata
