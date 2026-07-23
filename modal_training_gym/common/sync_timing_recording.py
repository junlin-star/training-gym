from collections.abc import MutableMapping

from modal_training_gym.common.timing_types import Substep


def record_sync_timing_event(
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
