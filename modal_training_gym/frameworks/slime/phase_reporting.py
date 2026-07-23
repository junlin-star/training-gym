"""Stable import surface for slime's in-container dashboard reporting.

The build-time patches (``modal_helpers/patches/patch_rollout_status_reporting.py``
and ``patch_advantage_distribution.py``) and the recipe's default custom-function
paths import from this module *inside the training container*, so everything
they reference must stay importable here. The implementation is split across:

- :mod:`.reporting` — HTTP queue/URL/token plumbing + run-context helpers
- :mod:`.sample_extraction` — trace/image/trajectory extraction from Samples
- :mod:`.advantage_reporting` — torch/megatron advantage-distribution math

This module keeps the reporting entry points (``report_*``, ``log_*``,
``before_*``) and re-exports the split internals for compatibility.
"""

from __future__ import annotations

import os
import time
from functools import wraps
from typing import Any, Callable

from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.common.step_timing import TRAINING_ROLE_FINISH_EVENT

from .advantage_reporting import (
    _advantage_samples_payload as _advantage_samples_payload,
    report_advantage_distribution as report_advantage_distribution,
)
from .reporting import (
    _enqueue,
    _enqueue_completed_step_status,
    _enqueue_rollout,
    _enqueue_timing_event,
    _run_context,
    _step_progress,
)
from .reporting import (
    _advantage_url as _advantage_url,
    _arg_value as _arg_value,
    _derive_url as _derive_url,
    _enqueue_advantage as _enqueue_advantage,
    _phase_url as _phase_url,
    _positive_int as _positive_int,
    _report_token as _report_token,
    _rollout_url as _rollout_url,
    _total_steps as _total_steps,
)
from .sample_extraction import (
    _image_sample_limit,
    _metrics_to_dict,
    _reward_function_timing,
    _resolve_hook,
    _response_parser,
    _sample_to_dict,
    _trace_enabled,
    _trace_sample_limit,
    _trajectory_sample_limit,
)
from .sample_extraction import (
    CAPTURE_TRACE_ENV as CAPTURE_TRACE_ENV,
    IMAGE_SAMPLE_LIMIT_ENV as IMAGE_SAMPLE_LIMIT_ENV,
    RESPONSE_PARSER_PATH_ENV as RESPONSE_PARSER_PATH_ENV,
    TRACE_SAMPLE_LIMIT_ENV as TRACE_SAMPLE_LIMIT_ENV,
    TRAJECTORY_SAMPLE_LIMIT_ENV as TRAJECTORY_SAMPLE_LIMIT_ENV,
    _IMAGE_MAX_BYTES as _IMAGE_MAX_BYTES,
    _IMAGE_MAX_DIM as _IMAGE_MAX_DIM,
    _IMAGE_SAMPLE_LIMIT_DEFAULT as _IMAGE_SAMPLE_LIMIT_DEFAULT,
    _TRACE_ATTR_STR_MAX as _TRACE_ATTR_STR_MAX,
    _TRACE_MAX_SPANS as _TRACE_MAX_SPANS,
    _TRACE_SAMPLE_LIMIT_DEFAULT as _TRACE_SAMPLE_LIMIT_DEFAULT,
    _TRAJECTORY_MAX_MESSAGES as _TRAJECTORY_MAX_MESSAGES,
    _TRAJECTORY_MSG_CHARS_MAX as _TRAJECTORY_MSG_CHARS_MAX,
    _TRAJECTORY_SAMPLE_LIMIT_DEFAULT as _TRAJECTORY_SAMPLE_LIMIT_DEFAULT,
    _coerce_float as _coerce_float,
    _sample_score as _sample_score,
    _coerce_text as _coerce_text,
    _compact_trajectory_messages as _compact_trajectory_messages,
    _extract_audio_from_prompt as _extract_audio_from_prompt,
    _extract_image_from_sample as _extract_image_from_sample,
    _extract_trace as _extract_trace,
    _image_to_data_uri as _image_to_data_uri,
    _normalize_span as _normalize_span,
    _normalize_trace as _normalize_trace,
    _parsed_response_dict as _parsed_response_dict,
    _trace_attrs as _trace_attrs,
    _trace_scalar as _trace_scalar,
)

CUSTOM_ROLLOUT_LOG_FUNCTION_PATH_KEY = "training_gym_custom_rollout_log_function_path"
CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH_KEY = (
    "training_gym_custom_eval_rollout_log_function_path"
)
CUSTOM_BEFORE_LOG_PROB_HOOK_PATH_KEY = (
    "training_gym_custom_megatron_before_log_prob_hook_path"
)
CUSTOM_BEFORE_TRAIN_STEP_HOOK_PATH_KEY = (
    "training_gym_custom_megatron_before_train_step_hook_path"
)
CUSTOM_REWARD_FUNCTION_PHASE = "custom_reward_function"


def _hook_path_from_args(args: Any, path_key: str) -> str | None:
    direct = getattr(args, path_key, None)
    if isinstance(direct, str) and direct.strip():
        return direct

    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            value = container.get(path_key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _training_role_name(value: Any, default: str = "driver") -> str:
    return str(getattr(value, "value", value) or default)


def report_phase(
    status: SlimeStatus,
    args: Any = None,
    *,
    record_timing: bool = True,
    **extra: Any,
) -> None:
    payload = {
        **_run_context(args),
        "phase": status.value,
        "event_ts": time.time(),
        **extra,
    }
    if record_timing and payload.get("rollout_id") is not None:
        _enqueue_timing_event(payload)
    _enqueue(payload)


def report_rollout_samples(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: Any,
    rollout_time: Any,
) -> None:
    """Post one TrainingRolloutResult-shaped payload to the dashboard."""
    if samples is None:
        return
    parser = _response_parser()
    # Trace/image only the first N samples (traces also gated by an enable flag) so
    # the payload stays small — the caps keep volume growth well under 1%.
    trace_limit = _trace_sample_limit() if _trace_enabled() else 0
    image_limit = _image_sample_limit()
    trajectory_limit = _trajectory_sample_limit()
    try:
        sample_dicts = [
            _sample_to_dict(
                s,
                parser,
                include_trace=(i < trace_limit),
                include_image=(i < image_limit),
                include_trajectory=(i < trajectory_limit),
            )
            for i, s in enumerate(samples)
        ]
    except TypeError:
        return
    payload = {
        **_run_context(args),
        "rollout_id": int(rollout_id),
        "created_at": int(time.time()),
        "samples": sample_dicts,
        "metrics": _metrics_to_dict(rollout_extra_metrics),
    }
    if rollout_time is not None:
        try:
            payload["rollout_time"] = float(rollout_time)
        except (TypeError, ValueError):
            pass
    _enqueue_rollout(payload)


def _report_custom_reward_function_timing(
    rollout_id: int,
    args: Any,
    samples: Any,
) -> None:
    if os.environ.get("TRAINING_GYM_ASYNC_MODE") == "1" or not _arg_value(
        args, "custom_rm_path"
    ):
        return
    try:
        timing = _reward_function_timing(samples)
    except TypeError:
        return
    if timing is None:
        return
    start, finish, active_duration = timing
    for boundary, timestamp in (
        ("phase_start", start),
        ("phase_finish", finish),
    ):
        report_step_event(
            CUSTOM_REWARD_FUNCTION_PHASE,
            args,
            rollout_id,
            boundary,
            step_id=0,
            event_ts=timestamp,
            active_duration_s=active_duration,
            training_role="rollout",
            timeline_lane="rollout",
            parent_phase=SlimeStatus.ROLLOUT_LOGGING.value,
            display_name="Custom reward function",
        )


def _call_hook(path_key: str, args: Any, *hook_args: Any, **hook_kwargs: Any) -> Any:
    hook = _resolve_hook(_hook_path_from_args(args, path_key))
    if hook is None:
        return None
    return hook(*hook_args, **hook_kwargs)


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: Any,
    rollout_time: Any,
) -> bool:
    progress = _step_progress(args, rollout_id)
    report_phase(
        SlimeStatus.ROLLOUT_LOGGING,
        args,
        record_timing=False,
        **progress,
        rollout_id=rollout_id,
        sample_count=len(samples) if hasattr(samples, "__len__") else None,
        metrics=rollout_extra_metrics,
        rollout_time=rollout_time,
    )
    _report_custom_reward_function_timing(rollout_id, args, samples)
    report_step_event(
        SlimeStatus.TRAINING,
        args,
        rollout_id,
        TRAINING_ROLE_FINISH_EVENT,
        training_role="rollout",
    )
    report_rollout_samples(
        rollout_id, args, samples, rollout_extra_metrics, rollout_time
    )
    result = _call_hook(
        CUSTOM_ROLLOUT_LOG_FUNCTION_PATH_KEY,
        args,
        rollout_id,
        args,
        samples,
        rollout_extra_metrics,
        rollout_time,
    )
    if result is None:
        return False
    return bool(result)


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: Any,
    extra_metrics: Any,
) -> bool:
    report_phase(
        SlimeStatus.EVAL_ROLLOUT_LOGGING,
        args,
        record_timing=False,
        **_step_progress(args, rollout_id),
        rollout_id=rollout_id,
        sample_count=len(data) if hasattr(data, "__len__") else None,
        metrics=extra_metrics,
    )
    result = _call_hook(
        CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH_KEY,
        args,
        rollout_id,
        args,
        data,
        extra_metrics,
    )
    if result is None:
        return False
    return bool(result)


def before_log_prob_hook(args: Any, model: Any, store_prefix: str) -> None:
    # NOTE: this hook runs in the Megatron train actor, which has no current
    # rollout_id, so the compute_log_probs substep is reported (id-tagged) from
    # the driver loop right before actor_model.async_train(); see the patch.
    _call_hook(
        CUSTOM_BEFORE_LOG_PROB_HOOK_PATH_KEY,
        args,
        args,
        model,
        store_prefix,
    )


def before_train_step_hook(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    async_mode = (
        bool(getattr(args, "async_mode", False))
        or os.environ.get("TRAINING_GYM_ASYNC_MODE") == "1"
    )
    training_role = _training_role_name(
        getattr(args, "training_gym_role", None), default="actor"
    )
    primary_rank = _is_primary_training_rank(args)
    if primary_rank:
        report_phase(
            SlimeStatus.OPTIMIZER_STEP if async_mode else SlimeStatus.TRAIN_MODEL,
            args,
            **_step_progress(args, rollout_id),
            rollout_id=rollout_id,
            step_id=step_id,
            training_role=training_role,
        )
    _call_hook(
        CUSTOM_BEFORE_TRAIN_STEP_HOOK_PATH_KEY,
        args,
        args,
        rollout_id,
        step_id,
        model,
        optimizer,
        opt_param_scheduler,
    )
    if primary_rank and not async_mode:
        before_optimizer_hook(
            args,
            rollout_id,
            step_id,
            optimizer,
            training_role=training_role,
        )


_OPTIMIZER_STEP_TIMING_CONTEXT = "_training_gym_optimizer_step_timing_context"
_OPTIMIZER_STEP_TIMING_INSTALLED = "_training_gym_optimizer_step_timing_installed"


def _is_primary_training_rank(args: Any) -> bool:
    rank = getattr(args, "rank", None)
    if rank is not None:
        try:
            return int(rank) == 0
        except (TypeError, ValueError):
            pass
    try:
        import torch.distributed as dist
    except ImportError:
        return True
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def before_optimizer_hook(
    args: Any,
    rollout_id: int,
    step_id: int,
    optimizer: object,
    *,
    training_role: str = "actor",
) -> None:
    if not getattr(optimizer, _OPTIMIZER_STEP_TIMING_INSTALLED, False):
        original_step: Callable[..., object] = getattr(optimizer, "step")
        original_zero_grad: Callable[..., object] = getattr(optimizer, "zero_grad")

        def finish_train_model() -> tuple[Any, int, int, str, float] | None:
            context = getattr(optimizer, _OPTIMIZER_STEP_TIMING_CONTEXT, None)
            if context is None:
                return None
            setattr(optimizer, _OPTIMIZER_STEP_TIMING_CONTEXT, None)
            hook_args, current_rollout_id, current_step_id, current_role = context
            finished_at = time.time()
            report_step_event(
                SlimeStatus.TRAIN_MODEL,
                hook_args,
                current_rollout_id,
                "phase_finish",
                step_id=current_step_id,
                event_ts=finished_at,
                training_role=current_role,
            )
            return (
                hook_args,
                current_rollout_id,
                current_step_id,
                current_role,
                finished_at,
            )

        @wraps(original_step)
        def timed_step(*step_args: object, **step_kwargs: object) -> object:
            context = finish_train_model()
            if context is None:
                return original_step(*step_args, **step_kwargs)
            (
                hook_args,
                current_rollout_id,
                current_step_id,
                current_role,
                optimizer_started,
            ) = context
            report_step_event(
                SlimeStatus.OPTIMIZER_STEP,
                hook_args,
                current_rollout_id,
                "phase_start",
                step_id=current_step_id,
                event_ts=optimizer_started,
                training_role=current_role,
            )
            try:
                return original_step(*step_args, **step_kwargs)
            finally:
                report_step_event(
                    SlimeStatus.OPTIMIZER_STEP,
                    hook_args,
                    current_rollout_id,
                    "phase_finish",
                    step_id=current_step_id,
                    event_ts=time.time(),
                    training_role=current_role,
                )

        @wraps(original_zero_grad)
        def timed_zero_grad(
            *zero_grad_args: object, **zero_grad_kwargs: object
        ) -> object:
            finish_train_model()
            return original_zero_grad(*zero_grad_args, **zero_grad_kwargs)

        setattr(optimizer, "step", timed_step)
        setattr(optimizer, "zero_grad", timed_zero_grad)
        setattr(optimizer, _OPTIMIZER_STEP_TIMING_INSTALLED, True)

    setattr(
        optimizer,
        _OPTIMIZER_STEP_TIMING_CONTEXT,
        (args, rollout_id, step_id, training_role),
    )
    report_step_event(
        SlimeStatus.TRAIN_MODEL,
        args,
        rollout_id,
        "phase_start",
        step_id=step_id,
        training_role=training_role,
    )


def report_training_role_finished(
    args: Any,
    rollout_id: int,
    training_role: str,
) -> None:
    if os.environ.get("TRAINING_GYM_ASYNC_MODE") == "1":
        return
    if not _is_primary_training_rank(args):
        return
    report_step_event(
        SlimeStatus.TRAINING,
        args,
        rollout_id,
        TRAINING_ROLE_FINISH_EVENT,
        training_role=_training_role_name(training_role, default="actor"),
    )


def report_step_event(
    status: SlimeStatus | str,
    args: Any = None,
    rollout_id: int | None = None,
    step_event: str = "",
    *,
    step_id: int | None = None,
    event_ts: float | None = None,
    event_monotonic: float | None = None,
    active_duration_s: float | None = None,
    training_role: str = "driver",
    timeline_lane: str | None = None,
    parent_phase: str | None = None,
    display_name: str | None = None,
    expected_training_roles: list[str] | None = None,
) -> None:
    """Report one step/substep event tagged with the ``status`` phase.

    ``status`` may be a plain string — the patched slime train.py passes phase
    names as literals so the injected code stays stdlib-only.
    """
    async_mode = os.environ.get("TRAINING_GYM_ASYNC_MODE") == "1"
    payload = {
        **_run_context(args),
        "phase": status.value if isinstance(status, SlimeStatus) else str(status),
        **_step_progress(args, rollout_id),
        "rollout_id": rollout_id,
        "event_ts": event_ts if event_ts is not None else time.time(),
        "training_role": _training_role_name(training_role),
    }
    if event_monotonic is not None:
        payload["event_monotonic"] = event_monotonic
    if active_duration_s is not None:
        payload["active_duration_s"] = active_duration_s
    if step_event:
        payload["step_event"] = step_event
    if step_id is not None:
        payload["step_id"] = step_id
    if timeline_lane is not None:
        payload["timeline_lane"] = timeline_lane
    if parent_phase is not None:
        payload["parent_phase"] = parent_phase
    if display_name is not None:
        payload["display_name"] = display_name
    if rollout_id is not None:
        _enqueue_timing_event(payload)
    if expected_training_roles is not None:
        payload["expected_training_roles"] = expected_training_roles
    if step_event in {"phase_start", "phase_finish", TRAINING_ROLE_FINISH_EVENT}:
        return
    if step_event == "substep_finish":
        if async_mode:
            payload.pop("step_event", None)
            _enqueue(payload)
        else:
            _enqueue_completed_step_status(
                {
                    **payload,
                    "completed_step": payload["progress_current"],
                }
            )
    else:
        _enqueue(payload)


__all__ = [
    "before_log_prob_hook",
    "before_optimizer_hook",
    "before_train_step_hook",
    "report_advantage_distribution",
    "report_phase",
    "report_rollout_samples",
    "report_step_event",
    "report_training_role_finished",
    "log_eval_rollout_data",
    "log_rollout_data",
]
