from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, MutableMapping
from enum import Enum
from typing import Any

from modal_training_gym.common.status import SlimeStatus


class Substep(str, Enum):
    EVAL_BEFORE = SlimeStatus.EVAL_ROLLOUT_LOGGING.value
    GENERATE_ROLLOUTS = SlimeStatus.ROLLOUT_LOGGING.value
    OFFLOAD_ROLLOUT = SlimeStatus.OFFLOAD_ROLLOUT.value
    COMPUTE_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
    OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
    CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
    OFFLOAD_TRAIN = SlimeStatus.OFFLOAD_TRAIN.value
    WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value
    EVAL_AFTER = f"{SlimeStatus.EVAL_ROLLOUT_LOGGING.value}_end"


AsyncUpdateTimestamps = dict[tuple[int, str], dict[tuple[str, int], dict[str, float]]]


def record_sync_step_time(
    step_times: MutableMapping[str, float],
    training_run_id: str,
    current_step: object,
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
    elif step_event == "substep_finish":
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


def record_async_step_times(
    step_times: MutableMapping[str, float],
    training_run_id: str,
    events: Iterable[Mapping[str, Any]],
) -> AsyncUpdateTimestamps:
    aggregated_events: list[tuple[int, str, str, float, int | None, int, str]] = []
    for event in events:
        # Rollout steps are one-based for progress; Slime rollout IDs are zero-based.
        try:
            rollout_step = int(event["progress_current"])
            event_ts = float(event["event_ts"])
            training_attempt = int(event.get("training_attempt", 0) or 0)
            update_id = event.get("step_id")
            update_id = int(update_id) if update_id is not None else None
        except (KeyError, TypeError, ValueError):
            continue
        phase = event.get("phase")
        step_event = event.get("step_event", "")
        training_role = event.get("training_role", "")
        if not isinstance(training_role, str):
            training_role = ""
        if (
            not math.isfinite(event_ts)
            or not isinstance(phase, str)
            or not isinstance(step_event, str)
        ):
            continue
        aggregated_events.append(
            (
                rollout_step,
                phase,
                step_event,
                event_ts,
                update_id,
                training_attempt,
                training_role,
            )
        )

    latest_attempt_by_rollout_phase: dict[tuple[int, str, str], int] = {}
    for (
        rollout_step,
        phase,
        _,
        _,
        _,
        training_attempt,
        training_role,
    ) in aggregated_events:
        rollout_phase = (rollout_step, phase, training_role)
        latest_attempt_by_rollout_phase[rollout_phase] = max(
            training_attempt,
            latest_attempt_by_rollout_phase.get(rollout_phase, training_attempt),
        )
    update_timestamps: AsyncUpdateTimestamps = {}

    def record_first(rollout_step: int, key: str, event_time: float) -> None:
        step_window_start = step_times.get(
            f"{training_run_id}:{rollout_step}:substep_start"
        )
        existing = step_times.get(key)
        if existing is not None and (
            step_window_start is None or existing >= step_window_start
        ):
            return
        step_times[key] = (
            max(event_time, step_window_start)
            if step_window_start is not None
            else event_time
        )

    for (
        rollout_step,
        phase,
        step_event,
        event_ts,
        update_id,
        training_attempt,
        training_role,
    ) in sorted(aggregated_events, key=lambda event: (event[5], event[3])):
        if (
            training_attempt
            != latest_attempt_by_rollout_phase[(rollout_step, phase, training_role)]
        ):
            continue
        if update_id is not None and step_event in ("phase_start", "phase_finish"):
            timestamp_kind = step_event.removeprefix("phase_")
            updates = update_timestamps.setdefault((rollout_step, phase), {})
            updates.setdefault((training_role, update_id), {})[timestamp_kind] = (
                event_ts
            )
            if step_event == "phase_finish":
                continue

        event_time = round(event_ts, 3)
        if step_event == "start":
            step_times[f"{training_run_id}:{rollout_step}:start"] = event_time
        elif step_event == "finish":
            step_times[f"{training_run_id}:{rollout_step}:finish"] = event_time
        elif step_event == "substep_start":
            step_times[f"{training_run_id}:{rollout_step}:substep_start"] = event_time
        elif step_event == "substep_finish":
            record_first(
                rollout_step,
                f"{training_run_id}:{rollout_step}:substep_finish",
                event_time,
            )
        elif step_event == "phase_start":
            record_first(
                rollout_step,
                f"{training_run_id}:{rollout_step}:substep:{phase}",
                event_time,
            )
        elif step_event == "phase_finish":
            record_first(
                rollout_step,
                f"{training_run_id}:{rollout_step}:substep:{phase}:finish",
                event_time,
            )

    return update_timestamps
