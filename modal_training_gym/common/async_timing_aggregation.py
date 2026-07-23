from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict, cast

from modal_training_gym.common.async_timing_types import AsyncTimingEvent
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.timing_types import (
    StepTimes,
    SubstepTimes,
    SubstepTiming,
    SubstepTimingInterval,
    TimingLane,
)


class TimingAttemptSnapshot(TypedDict):
    step_times: StepTimes
    substep_times: SubstepTimes


class StepTimingSnapshot(TimingAttemptSnapshot):
    displayed_training_attempt: int
    archived_timing_attempts: dict[str, TimingAttemptSnapshot]
    timing_event_counts: dict[int, int]


class _PhaseMetadata(TypedDict, total=False):
    timeline_lane: TimingLane
    parent_phase: str
    display_name: str


class _Occurrence(TypedDict, total=False):
    start: AsyncTimingEvent
    finish: AsyncTimingEvent


def aggregate_async_step_times(
    events: Iterable[AsyncTimingEvent],
    num_rollouts: int,
    *,
    training_attempt: int,
) -> tuple[StepTimes, SubstepTimes]:
    """Aggregate one canonical async event format into dashboard timing data."""
    timing_events = [
        event
        for event in events
        if event["training_attempt"] == training_attempt
        and 0 <= event["rollout_id"] < num_rollouts
    ]
    rollout_timestamps: dict[int, dict[str, float]] = {}
    phase_occurrences: dict[
        tuple[int, str],
        dict[tuple[str | None, int | None, int | None], _Occurrence],
    ] = {}
    phases_by_rollout: dict[int, set[str]] = {}
    phase_metadata: dict[tuple[int, str], _PhaseMetadata] = {}

    for event in sorted(timing_events, key=lambda item: item["timestamp"]):
        rollout_id = event["rollout_id"]
        event_type = event["event_type"]
        if event_type in ("rollout_start", "rollout_finish"):
            timestamp_name = event_type.removeprefix("rollout_")
            timestamps = rollout_timestamps.setdefault(rollout_id, {})
            if timestamp_name == "start":
                timestamps[timestamp_name] = min(
                    timestamps.get(timestamp_name, event["timestamp"]),
                    event["timestamp"],
                )
            else:
                timestamps[timestamp_name] = max(
                    timestamps.get(timestamp_name, event["timestamp"]),
                    event["timestamp"],
                )
            continue

        phases_by_rollout.setdefault(rollout_id, set()).add(event["phase"])
        occurrence = phase_occurrences.setdefault(
            (rollout_id, event["phase"]), {}
        ).setdefault(
            (event["role"], event["occurrence_id"], event["rank"]),
            {},
        )
        if event_type == "phase_start":
            occurrence["start"] = event
        else:
            occurrence["finish"] = event

        metadata: _PhaseMetadata = {}
        if event["timeline_lane"] is not None:
            metadata["timeline_lane"] = event["timeline_lane"]
        if event["parent_phase"] is not None:
            metadata["parent_phase"] = event["parent_phase"]
        if event["display_name"] is not None:
            metadata["display_name"] = event["display_name"]
        if metadata:
            phase_metadata[rollout_id, event["phase"]] = metadata

    step_times: StepTimes = {}
    substep_times: SubstepTimes = {}

    for rollout_id in range(num_rollouts):
        step_number = rollout_id + 1
        step_key = str(step_number)
        timestamps = rollout_timestamps.get(rollout_id, {})
        start = int(timestamps["start"]) if "start" in timestamps else None
        end = int(timestamps["finish"]) if "finish" in timestamps else None
        step_times[step_key] = {
            "start": start,
            "end": end,
            "duration_s": (
                end - start if start is not None and end is not None else None
            ),
        }
        substep_times[step_key] = {}

        for phase in sorted(phases_by_rollout.get(rollout_id, ())):
            logical_occurrences: dict[
                tuple[str | None, int | None],
                list[tuple[int | None, _Occurrence]],
            ] = {}
            for (
                role,
                occurrence_id,
                rank,
            ), occurrence in phase_occurrences.get((rollout_id, phase), {}).items():
                logical_occurrences.setdefault((role, occurrence_id), []).append(
                    (rank, occurrence)
                )

            phase_starts: list[float] = []
            completed_ranges: list[tuple[float, float]] = []
            detailed_intervals: list[SubstepTimingInterval] = []
            keep_individual_intervals = False

            for (role, occurrence_id), rank_occurrences in sorted(
                logical_occurrences.items(),
                key=lambda item: (
                    item[0][0] or "",
                    item[0][1] is None,
                    item[0][1] if item[0][1] is not None else -1,
                ),
            ):
                completed_occurrences: list[
                    tuple[
                        float,
                        int | None,
                        AsyncTimingEvent,
                        AsyncTimingEvent,
                    ]
                ] = []
                for rank, occurrence in rank_occurrences:
                    start_event = occurrence.get("start")
                    finish_event = occurrence.get("finish")
                    if start_event is not None:
                        phase_starts.append(start_event["timestamp"])
                    if start_event is None or finish_event is None:
                        continue
                    duration = (
                        finish_event["monotonic_timestamp"]
                        - start_event["monotonic_timestamp"]
                    )
                    if duration < 0:
                        continue
                    completed_occurrences.append(
                        (duration, rank, start_event, finish_event)
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
                interval_start = start_event["timestamp"]
                completed_ranges.append((interval_start, interval_start + duration))

                if role is None and slowest_rank is None:
                    continue
                keep_individual_intervals = True
                interval: SubstepTimingInterval = {
                    "start": round(interval_start, 3),
                    "duration_s": round(duration, 3),
                }
                if occurrence_id is not None:
                    interval["step_id"] = occurrence_id
                if role is not None:
                    interval["training_role"] = role
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
                        event["world_size"]
                        for _, _, start_event, finish_event in completed_occurrences
                        for event in (start_event, finish_event)
                        if event["world_size"] is not None
                    ]
                    if world_sizes:
                        interval["training_world_size"] = max(world_sizes)
                detailed_intervals.append(interval)

            if not phase_starts:
                continue

            merged_ranges: list[tuple[float, float]] = []
            for range_start, range_end in sorted(completed_ranges):
                if merged_ranges and range_start <= merged_ranges[-1][1]:
                    previous_start, previous_end = merged_ranges[-1]
                    merged_ranges[-1] = (
                        previous_start,
                        max(previous_end, range_end),
                    )
                else:
                    merged_ranges.append((range_start, range_end))

            timing: SubstepTiming = {
                "start": round(
                    merged_ranges[0][0] if merged_ranges else min(phase_starts),
                    3,
                ),
                "duration_s": (
                    round(
                        sum(
                            range_end - range_start
                            for range_start, range_end in merged_ranges
                        ),
                        3,
                    )
                    if merged_ranges
                    else None
                ),
            }
            metadata = phase_metadata.get((rollout_id, phase), {})
            intervals: list[SubstepTimingInterval]
            if keep_individual_intervals:
                intervals = detailed_intervals
            else:
                intervals = [
                    SubstepTimingInterval(
                        start=round(range_start, 3),
                        duration_s=round(range_end - range_start, 3),
                    )
                    for range_start, range_end in merged_ranges
                ]
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
    resume_from_iteration: int | None,
) -> None:
    current_step_times = {
        step: timing
        for step, timing in new_step_times.items()
        if any(value is not None for value in timing.values())
    }
    current_substep_times = {
        step: timings for step, timings in new_substep_times.items() if timings
    }
    current_steps = current_step_times.keys() | current_substep_times.keys()
    previous_step_times = dict(run_record.step_times or {})
    previous_substep_times = dict(run_record.substep_times or {})
    metadata = dict(run_record.metadata or {})
    if previous_step_times or previous_substep_times:
        stored_attempt = metadata.get("displayed_training_attempt")
        displayed_attempt = (
            stored_attempt
            if isinstance(stored_attempt, int)
            else max(1, training_attempt - 1)
        )
    else:
        displayed_attempt = None
    stored_attempts = metadata.get("archived_timing_attempts")
    archived_attempts = (
        cast(dict[str, TimingAttemptSnapshot], dict(stored_attempts))
        if isinstance(stored_attempts, dict)
        else {}
    )

    if not current_steps:
        if displayed_attempt is not None:
            metadata["displayed_training_attempt"] = displayed_attempt
            archived_attempts.pop(str(displayed_attempt), None)
        if archived_attempts:
            metadata["archived_timing_attempts"] = archived_attempts
        else:
            metadata.pop("archived_timing_attempts", None)
        run_record.metadata = metadata or None
        return

    archived_attempts.pop(str(training_attempt), None)
    if displayed_attempt is not None and displayed_attempt != training_attempt:
        previous_attempt = str(displayed_attempt)
        if previous_attempt not in archived_attempts:
            archived_attempts[previous_attempt] = {
                "step_times": previous_step_times,
                "substep_times": previous_substep_times,
            }

    if resume_from_iteration is not None:
        first_new_step = resume_from_iteration + 2
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

    metadata["displayed_training_attempt"] = training_attempt
    if archived_attempts:
        metadata["archived_timing_attempts"] = archived_attempts
    else:
        metadata.pop("archived_timing_attempts", None)
    run_record.step_times = displayed_step_times
    run_record.substep_times = displayed_substep_times
    run_record.metadata = metadata


def reconcile_completed_step_times(
    run_record: TrainingRun,
    step_times: StepTimes,
    substep_times: SubstepTimes,
    *,
    training_attempt: int,
    resume_from_iteration: int | None,
) -> None:
    metadata = run_record.metadata or {}
    if (
        training_attempt <= 1
        or metadata.get("displayed_training_attempt") == training_attempt
    ):
        step_times = {**(run_record.step_times or {}), **step_times}
        substep_times = {**(run_record.substep_times or {}), **substep_times}
    reconcile_attempt_step_times(
        run_record,
        step_times,
        substep_times,
        training_attempt=training_attempt,
        resume_from_iteration=resume_from_iteration,
    )


def apply_step_timing_snapshot(
    run_record: TrainingRun,
    snapshot: StepTimingSnapshot,
    *,
    training_attempt: int,
) -> None:
    snapshot_attempt = snapshot["displayed_training_attempt"]
    if snapshot_attempt < training_attempt:
        run_record.step_times = {
            **snapshot["step_times"],
            **(run_record.step_times or {}),
        }
        run_record.substep_times = {
            **snapshot["substep_times"],
            **(run_record.substep_times or {}),
        }
    else:
        run_record.step_times = snapshot["step_times"]
        run_record.substep_times = snapshot["substep_times"]

    metadata = dict(run_record.metadata or {})
    stored_attempts = metadata.get("archived_timing_attempts")
    existing_attempts = (
        cast(dict[str, TimingAttemptSnapshot], dict(stored_attempts))
        if isinstance(stored_attempts, dict)
        else {}
    )
    if snapshot_attempt < training_attempt:
        archived_attempts = {
            **snapshot["archived_timing_attempts"],
            **existing_attempts,
        }
    else:
        archived_attempts = {
            **existing_attempts,
            **snapshot["archived_timing_attempts"],
        }
    if snapshot_attempt >= training_attempt or not isinstance(
        metadata.get("displayed_training_attempt"), int
    ):
        metadata["displayed_training_attempt"] = snapshot_attempt
    displayed_attempt = str(metadata["displayed_training_attempt"])
    archived_attempts.pop(displayed_attempt, None)
    if archived_attempts:
        metadata["archived_timing_attempts"] = archived_attempts
    else:
        metadata.pop("archived_timing_attempts", None)
    run_record.metadata = metadata
