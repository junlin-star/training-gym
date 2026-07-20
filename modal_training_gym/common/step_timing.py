from __future__ import annotations

from collections.abc import MutableMapping
from enum import Enum
from typing import Any, Iterable, Mapping

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


def record_step_time_event(
    step_times: MutableMapping[Any, Any],
    training_run_id: str,
    current_step: Any,
    phase: str,
    step_event: str,
    event_ts: float,
    step_id: int | None = None,
) -> None:
    if not (isinstance(current_step, int) and current_step > 0):
        return
    event_time = float(event_ts)
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

    if step_id is not None:
        if step_event in ("phase_start", "phase_finish"):
            boundary = step_event.removeprefix("phase_")
            key = (
                training_run_id,
                "substep_boundary",
                current_step,
                phase,
                step_id,
                boundary,
            )
            step_times[key] = event_time

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
        if step_id is not None:
            return
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


def record_step_time_events(
    step_times: MutableMapping[Any, Any],
    training_run_id: str,
    events: Iterable[Mapping[str, Any]],
) -> None:
    run_events = [
        event
        for event in events
        if event.get("training_run_id") == training_run_id
        and event.get("event_ts") is not None
    ]

    def training_attempt(event: Mapping[str, Any]) -> int:
        try:
            return int(event.get("training_attempt") or 0)
        except (TypeError, ValueError):
            return 0

    latest_attempt_by_phase: dict[tuple[Any, str], int] = {}
    for event in run_events:
        phase = (event.get("progress_current"), str(event.get("phase", "")))
        attempt = training_attempt(event)
        latest_attempt_by_phase[phase] = max(
            attempt,
            latest_attempt_by_phase.get(phase, attempt),
        )

    def event_order(event: Mapping[str, Any]) -> tuple[int, float]:
        try:
            event_ts = float(event["event_ts"])
        except (TypeError, ValueError):
            event_ts = 0
        return training_attempt(event), event_ts

    for event in sorted(run_events, key=event_order):
        phase = (event.get("progress_current"), str(event.get("phase", "")))
        if training_attempt(event) != latest_attempt_by_phase[phase]:
            continue
        record_step_time_event(
            step_times,
            training_run_id,
            event.get("progress_current"),
            str(event.get("phase", "")),
            str(event.get("step_event", "")),
            float(event["event_ts"]),
            event.get("step_id"),
        )
