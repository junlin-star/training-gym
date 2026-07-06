from __future__ import annotations

from collections.abc import MutableMapping
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
            Substep.EVAL_BEFORE if step_event == "eval_begin" else Substep.EVAL_AFTER
        )
        put_once(f"{training_run_id}:{current_step}:substep:{substep.value}")
    elif not step_event and phase == Substep.EVAL_BEFORE.value:
        pass
    elif step_event == "finish":
        pass
    else:
        put_once(f"{training_run_id}:{current_step}:substep:{phase}")
