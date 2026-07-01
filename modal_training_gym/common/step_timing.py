from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

EVAL_BEFORE_SUBSTEP = "evaluate_rollouts"
EVAL_AFTER_SUBSTEP = "evaluate_rollouts_end"


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

    def put_once(key: str) -> None:
        if step_times.get(key) is None:
            step_times[key] = event_time

    if step_event == "start":
        step_times[f"{training_run_id}:{current_step}:start"] = event_time
    elif step_event == "finish":
        step_times[f"{training_run_id}:{current_step}:finish"] = event_time

    if step_event == "substep_start":
        put_once(f"{training_run_id}:{current_step}:substep_start")
    elif step_event == "substep_finish":
        put_once(f"{training_run_id}:{current_step}:substep_finish")
    elif step_event in ("eval_begin", "eval_end"):
        substep = (
            EVAL_BEFORE_SUBSTEP if step_event == "eval_begin" else EVAL_AFTER_SUBSTEP
        )
        put_once(f"{training_run_id}:{current_step}:substep:{substep}")
    elif not step_event and phase == EVAL_BEFORE_SUBSTEP:
        pass
    elif step_event == "finish":
        pass
    else:
        put_once(f"{training_run_id}:{current_step}:substep:{phase}")
