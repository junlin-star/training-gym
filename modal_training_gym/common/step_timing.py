from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, MutableMapping
from enum import Enum

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


SubstepTimingBoundaries = dict[tuple[int, str], dict[int, dict[str, float]]]


def record_step_time_event(
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
    phase_start_key = f"{training_run_id}:{current_step}:substep:{phase}"
    phase_finish_key = f"{phase_start_key}:finish"

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
    elif step_event == "phase_start":
        record_first_in_step_window(phase_start_key)
    elif step_event == "phase_finish":
        record_first_in_step_window(phase_finish_key)
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


def reconcile_async_step_time_events(
    step_times: MutableMapping[str, float],
    training_run_id: str,
    events: Iterable[Mapping[str, object]],
) -> SubstepTimingBoundaries:
    normalized_events: list[tuple[int, str, str, float, int | None, int]] = []
    for event in events:
        if event.get("training_run_id") != training_run_id:
            continue
        raw_step = event.get("progress_current")
        raw_event_ts = event.get("event_ts")
        raw_attempt = event.get("training_attempt", 0)
        if not isinstance(raw_step, (int, float, str)) or not isinstance(
            raw_event_ts, (int, float, str)
        ):
            continue
        if not isinstance(raw_attempt, (int, float, str)):
            raw_attempt = 0
        try:
            step = int(raw_step)
            event_ts = float(raw_event_ts)
            training_attempt = int(raw_attempt or 0)
        except (TypeError, ValueError):
            continue
        phase = event.get("phase")
        step_event = event.get("step_event", "")
        if (
            step <= 0
            or not math.isfinite(event_ts)
            or not isinstance(phase, str)
            or not isinstance(step_event, str)
        ):
            continue
        raw_step_id = event.get("step_id")
        if raw_step_id is not None and not isinstance(raw_step_id, (int, float, str)):
            raw_step_id = None
        try:
            step_id = int(raw_step_id) if raw_step_id is not None else None
        except (TypeError, ValueError):
            step_id = None
        normalized_events.append(
            (step, phase, step_event, event_ts, step_id, training_attempt)
        )

    latest_attempt_by_phase: dict[tuple[int, str], int] = {}
    for step, phase, _, _, _, training_attempt in normalized_events:
        phase_key = (step, phase)
        latest_attempt_by_phase[phase_key] = max(
            training_attempt,
            latest_attempt_by_phase.get(phase_key, training_attempt),
        )
    boundaries: SubstepTimingBoundaries = {}
    for step, phase, step_event, event_ts, step_id, training_attempt in sorted(
        normalized_events, key=lambda event: (event[5], event[3])
    ):
        if training_attempt != latest_attempt_by_phase[(step, phase)]:
            continue
        if step_id is not None and step_event in ("phase_start", "phase_finish"):
            boundary = step_event.removeprefix("phase_")
            updates = boundaries.setdefault((step, phase), {})
            updates.setdefault(step_id, {})[boundary] = event_ts
            if step_event == "phase_finish":
                continue
        record_step_time_event(
            step_times,
            training_run_id,
            step,
            phase,
            step_event,
            event_ts,
        )
    return boundaries
