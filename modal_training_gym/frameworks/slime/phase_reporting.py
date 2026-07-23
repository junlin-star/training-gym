"""Stable import surface for slime's in-container dashboard reporting.

The build-time patches (``modal_helpers/patches/patch_rollout_status_reporting.py``
and ``patch_advantage_distribution.py``) and the recipe's default custom-function
paths import from this module *inside the training container*, so everything
they reference must stay importable here. The implementation is split across:

- :mod:`.reporting` — background delivery and run-context helpers
- :mod:`.sample_extraction` — trace/image/trajectory extraction from Samples
- :mod:`.advantage_reporting` — torch/megatron advantage-distribution math

This module keeps the reporting entry points (``report_*``, ``log_*``,
``before_*``) and re-exports the split internals for compatibility.
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.common.timing_types import TrainingSubstep

if TYPE_CHECKING:
    from modal_training_gym.common.async_timing_types import (
        AsyncTimingEvent,
        AsyncTimingEventType,
    )
    from modal_training_gym.common.timing_types import TimingLane

from .advantage_reporting import (
    _advantage_samples_payload as _advantage_samples_payload,
    report_advantage_distribution as report_advantage_distribution,
)
from .reporting import (
    _STEP_EVENT_TIMEOUT_SECONDS,
    _enqueue,
    _enqueue_async_timing_event,
    _enqueue_rollout,
    _post_framework_status,
    _run_context,
    _step_progress,
    flush_async_timing_events,
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
ROLLOUT_FUNCTION_PATH_KEY = "training_gym_rollout_function_path"
CUSTOM_REWARD_FUNCTION_PATH_KEY = "training_gym_custom_reward_function_path"
_LOG_PROB_PHASES = {
    "": (
        TrainingSubstep.POLICY_LOG_PROBS,
        "log_probs",
        "Policy log probabilities",
    ),
    "ref_": (
        TrainingSubstep.REFERENCE_LOG_PROBS,
        "ref_log_probs",
        "Reference log probabilities",
    ),
    "teacher_": (
        TrainingSubstep.TEACHER_LOG_PROBS,
        "teacher_log_probs",
        "Teacher log probabilities",
    ),
}
_LOG_PROB_STARTS: dict[str, tuple[float, float, float]] = {}
_CURRENT_ROLLOUT_ID: ContextVar[int | None] = ContextVar(
    "training_gym_rollout_id",
    default=None,
)
_P = ParamSpec("_P")
_R = TypeVar("_R")


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


def report_phase(
    status: SlimeStatus,
    args: Any = None,
    **extra: Any,
) -> None:
    _enqueue(
        {**_run_context(args), "phase": status.value, "event_ts": time.time(), **extra}
    )


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


def _call_hook(path_key: str, args: Any, *hook_args: Any, **hook_kwargs: Any) -> Any:
    hook = _resolve_hook(_hook_path_from_args(args, path_key))
    if hook is None:
        return None
    return hook(*hook_args, **hook_kwargs)


def flush_async_timing_queue_before_reraise(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except BaseException:
            flush_async_timing_events()
            raise

    return wrapped


@flush_async_timing_queue_before_reraise
def rollout_with_timing_context(
    args: object,
    rollout_id: int,
    data_source: object,
    evaluation: bool = False,
) -> object:
    rollout_function = cast(
        Callable[..., object],
        _resolve_hook(_hook_path_from_args(args, ROLLOUT_FUNCTION_PATH_KEY)),
    )
    token = _CURRENT_ROLLOUT_ID.set(None if evaluation else rollout_id)
    try:
        result = rollout_function(
            args,
            rollout_id,
            data_source,
            evaluation=evaluation,
        )
    finally:
        _CURRENT_ROLLOUT_ID.reset(token)

    num_rollouts = getattr(args, "num_rollout", None)
    if (
        not evaluation
        and isinstance(num_rollouts, int)
        and rollout_id == num_rollouts - 1
    ):
        flush_async_timing_events()
    return result


async def custom_reward_with_timing(
    args: object,
    sample: object,
    **kwargs: object,
) -> object:
    reward_function = cast(
        Callable[..., Awaitable[object]],
        _resolve_hook(_hook_path_from_args(args, CUSTOM_REWARD_FUNCTION_PATH_KEY)),
    )
    rollout_id = _CURRENT_ROLLOUT_ID.get()
    if rollout_id is None:
        return await reward_function(args, sample, **kwargs)

    occurrence_id = getattr(sample, "index", None)
    if not isinstance(occurrence_id, int) and isinstance(sample, (list, tuple)):
        occurrence_id = next(
            (
                index
                for item in sample
                if isinstance(index := getattr(item, "index", None), int)
            ),
            None,
        )
    if not isinstance(occurrence_id, int):
        return await reward_function(args, sample, **kwargs)
    started = time.time()
    started_monotonic = time.monotonic()
    try:
        return await reward_function(args, sample, **kwargs)
    finally:
        finished = time.time()
        finished_monotonic = time.monotonic()
        report_async_timing_event(
            "custom_reward",
            args,
            rollout_id,
            "phase_start",
            occurrence_id=occurrence_id,
            timestamp=started,
            monotonic_timestamp=started_monotonic,
            timeline_lane="reward",
            parent_phase="generate_rollouts",
            display_name="Custom reward function",
        )
        report_async_timing_event(
            "custom_reward",
            args,
            rollout_id,
            "phase_finish",
            occurrence_id=occurrence_id,
            timestamp=finished,
            monotonic_timestamp=finished_monotonic,
            timeline_lane="reward",
            parent_phase="generate_rollouts",
            display_name="Custom reward function",
        )


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: Any,
    rollout_time: Any,
) -> bool:
    progress = _step_progress(args, rollout_id)
    if not (
        bool(getattr(args, "async_mode", False))
        or os.environ.get("TRAINING_GYM_ASYNC_MODE") == "1"
    ):
        report_phase(
            SlimeStatus.ROLLOUT_LOGGING,
            args,
            **progress,
            rollout_id=rollout_id,
            sample_count=len(samples) if hasattr(samples, "__len__") else None,
            metrics=rollout_extra_metrics,
            rollout_time=rollout_time,
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
    _call_hook(
        CUSTOM_BEFORE_LOG_PROB_HOOK_PATH_KEY,
        args,
        args,
        model,
        store_prefix,
    )
    if (
        getattr(args, "training_gym_role", None) != "critic"
        and store_prefix in _LOG_PROB_PHASES
        and (
            bool(getattr(args, "async_mode", False))
            or os.environ.get("TRAINING_GYM_ASYNC_MODE") == "1"
        )
    ):
        try:
            from slime.utils.timer import Timer
        except ImportError:
            pass
        else:
            timer_name = _LOG_PROB_PHASES[store_prefix][1]
            elapsed = float(Timer().log_dict().get(timer_name, 0.0))
            _LOG_PROB_STARTS[store_prefix] = (
                time.time(),
                time.monotonic(),
                elapsed,
            )


_OPTIMIZER_TIMING_CONTEXT = "_training_gym_timing_context"
_OPTIMIZER_TIMING_INSTALLED = "_training_gym_timing_installed"


def _install_optimizer_timing(optimizer: object) -> None:
    if getattr(optimizer, _OPTIMIZER_TIMING_INSTALLED, False):
        return
    original_step: Callable[..., object] = getattr(optimizer, "step")
    original_zero_grad = getattr(optimizer, "zero_grad", None)

    def finish_forward_backward():
        context = getattr(optimizer, _OPTIMIZER_TIMING_CONTEXT, None)
        if context is None:
            return None
        setattr(optimizer, _OPTIMIZER_TIMING_CONTEXT, None)
        hook_args, rollout_id, occurrence_id = context
        finished = time.time()
        finished_monotonic = time.monotonic()
        report_async_timing_event(
            TrainingSubstep.FORWARD_BACKWARD,
            hook_args,
            rollout_id,
            "phase_finish",
            occurrence_id=occurrence_id,
            timestamp=finished,
            monotonic_timestamp=finished_monotonic,
            timeline_lane="training",
            parent_phase=SlimeStatus.TRAINING.value,
            display_name="Forward / backward",
        )
        return context, finished, finished_monotonic

    @wraps(original_step)
    def timed_step(*args: object, **kwargs: object) -> object:
        finished_forward_backward = finish_forward_backward()
        if finished_forward_backward is None:
            return original_step(*args, **kwargs)
        context, _, _ = finished_forward_backward
        hook_args, rollout_id, occurrence_id = context
        optimizer_started = time.time()
        optimizer_started_monotonic = time.monotonic()
        try:
            return original_step(*args, **kwargs)
        finally:
            optimizer_finished = time.time()
            optimizer_finished_monotonic = time.monotonic()
            report_async_timing_event(
                TrainingSubstep.OPTIMIZER_STEP,
                hook_args,
                rollout_id,
                "phase_start",
                occurrence_id=occurrence_id,
                timestamp=optimizer_started,
                monotonic_timestamp=optimizer_started_monotonic,
                timeline_lane="training",
                parent_phase=SlimeStatus.TRAINING.value,
                display_name="Optimizer step",
            )
            report_async_timing_event(
                TrainingSubstep.OPTIMIZER_STEP,
                hook_args,
                rollout_id,
                "phase_finish",
                occurrence_id=occurrence_id,
                timestamp=optimizer_finished,
                monotonic_timestamp=optimizer_finished_monotonic,
                timeline_lane="training",
                parent_phase=SlimeStatus.TRAINING.value,
                display_name="Optimizer step",
            )

    setattr(optimizer, "step", timed_step)
    if callable(original_zero_grad):

        @wraps(original_zero_grad)
        def timed_zero_grad(*args: object, **kwargs: object) -> object:
            finish_forward_backward()
            return original_zero_grad(*args, **kwargs)

        setattr(optimizer, "zero_grad", timed_zero_grad)
    setattr(optimizer, _OPTIMIZER_TIMING_INSTALLED, True)


def before_train_step_hook(
    args: object,
    rollout_id: int,
    step_id: int,
    model: object,
    optimizer: object,
    opt_param_scheduler: object,
) -> None:
    global _LOG_PROB_STARTS
    async_mode = (
        bool(getattr(args, "async_mode", False))
        or os.environ.get("TRAINING_GYM_ASYNC_MODE") == "1"
    )
    if not async_mode:
        report_phase(
            SlimeStatus.OPTIMIZER_STEP,
            args,
            **_step_progress(args, rollout_id),
            rollout_id=rollout_id,
            step_id=step_id,
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
    # This is separate from the mode branch so the custom hook runs first.
    if async_mode:
        _install_optimizer_timing(optimizer)
        if step_id == 0:
            log_prob_starts = _LOG_PROB_STARTS
            _LOG_PROB_STARTS = {}
            if log_prob_starts:
                try:
                    from slime.utils.timer import Timer
                except ImportError:
                    pass
                else:
                    timers = Timer().log_dict()
                    for store_prefix, (
                        started,
                        started_monotonic,
                        previous_elapsed,
                    ) in log_prob_starts.items():
                        phase, timer_name, display_name = _LOG_PROB_PHASES[store_prefix]
                        total_elapsed = timers.get(timer_name)
                        if total_elapsed is None:
                            continue
                        duration = float(total_elapsed) - previous_elapsed
                        if duration < 0:
                            continue
                        report_async_timing_event(
                            phase,
                            args,
                            rollout_id,
                            "phase_start",
                            occurrence_id=step_id,
                            timestamp=started,
                            monotonic_timestamp=started_monotonic,
                            timeline_lane="training",
                            parent_phase=SlimeStatus.TRAINING.value,
                            display_name=display_name,
                        )
                        report_async_timing_event(
                            phase,
                            args,
                            rollout_id,
                            "phase_finish",
                            occurrence_id=step_id,
                            timestamp=started + float(duration),
                            monotonic_timestamp=started_monotonic + float(duration),
                            timeline_lane="training",
                            parent_phase=SlimeStatus.TRAINING.value,
                            display_name=display_name,
                        )
        update_started = time.time()
        update_started_monotonic = time.monotonic()
        setattr(
            optimizer,
            _OPTIMIZER_TIMING_CONTEXT,
            (args, rollout_id, step_id),
        )
        report_async_timing_event(
            TrainingSubstep.FORWARD_BACKWARD,
            args,
            rollout_id,
            "phase_start",
            occurrence_id=step_id,
            timestamp=update_started,
            monotonic_timestamp=update_started_monotonic,
            timeline_lane="training",
            parent_phase=SlimeStatus.TRAINING.value,
            display_name="Forward / backward",
        )


@contextmanager
def record_async_phase_interval(
    phase: SlimeStatus | TrainingSubstep | str,
    args: object,
    rollout_id: int,
    *,
    occurrence_id: int | None = None,
    timeline_lane: TimingLane | None = None,
    parent_phase: str | None = None,
    display_name: str | None = None,
) -> Iterator[None]:
    started = time.time()
    started_monotonic = time.monotonic()
    try:
        yield
    finally:
        finished = time.time()
        finished_monotonic = time.monotonic()
        report_async_timing_event(
            phase,
            args,
            rollout_id,
            "phase_start",
            occurrence_id=occurrence_id,
            timestamp=started,
            monotonic_timestamp=started_monotonic,
            timeline_lane=timeline_lane,
            parent_phase=parent_phase,
            display_name=display_name,
        )
        report_async_timing_event(
            phase,
            args,
            rollout_id,
            "phase_finish",
            occurrence_id=occurrence_id,
            timestamp=finished,
            monotonic_timestamp=finished_monotonic,
            timeline_lane=timeline_lane,
            parent_phase=parent_phase,
            display_name=display_name,
        )


def report_async_timing_event(
    phase: SlimeStatus | TrainingSubstep | str,
    args: object,
    rollout_id: int,
    event_type: AsyncTimingEventType,
    occurrence_id: int | None = None,
    timestamp: float | None = None,
    monotonic_timestamp: float | None = None,
    *,
    timeline_lane: TimingLane | None = None,
    parent_phase: str | None = None,
    display_name: str | None = None,
) -> None:
    phase = (
        phase.value if isinstance(phase, (SlimeStatus, TrainingSubstep)) else str(phase)
    )
    role = getattr(args, "training_gym_role", None)
    role = role if isinstance(role, str) else None
    rank = getattr(args, "rank", None) if role is not None else None
    world_size = getattr(args, "world_size", None) if role is not None else None
    event: AsyncTimingEvent = {
        "training_run_id": str(_run_context(args)["training_run_id"]),
        "training_attempt": int(os.environ.get("TRAINING_GYM_TRAINING_ATTEMPT") or "1"),
        "rollout_id": rollout_id,
        "phase": phase,
        "event_type": event_type,
        "timestamp": time.time() if timestamp is None else timestamp,
        "monotonic_timestamp": (
            time.monotonic() if monotonic_timestamp is None else monotonic_timestamp
        ),
        "occurrence_id": occurrence_id,
        "role": role,
        "rank": rank if isinstance(rank, int) else None,
        "world_size": world_size if isinstance(world_size, int) else None,
        "timeline_lane": timeline_lane,
        "parent_phase": parent_phase,
        "display_name": display_name,
    }
    _enqueue_async_timing_event(event)


def report_step_event(
    status: SlimeStatus | str,
    args: object | None = None,
    rollout_id: int | None = None,
    step_event: str = "",
) -> None:
    """Report one sync step/substep event tagged with the ``status`` phase.

    ``status`` may be a plain string — the patched slime train.py passes phase
    names as literals so the injected code stays stdlib-only.
    """

    payload = {
        **_run_context(args),
        "phase": status.value if isinstance(status, SlimeStatus) else str(status),
        **_step_progress(args, rollout_id),
        "rollout_id": rollout_id,
        "event_ts": time.time(),
    }
    if step_event:
        payload["step_event"] = step_event
    match step_event:
        case "start":
            _post_framework_status(payload, _STEP_EVENT_TIMEOUT_SECONDS)
        case "finish":
            if rollout_id is not None:
                _post_framework_status(payload, _STEP_EVENT_TIMEOUT_SECONDS)
        case _:
            _enqueue(payload)


__all__ = [
    "before_log_prob_hook",
    "before_train_step_hook",
    "flush_async_timing_events",
    "flush_async_timing_queue_before_reraise",
    "report_advantage_distribution",
    "report_async_timing_event",
    "report_phase",
    "report_rollout_samples",
    "record_async_phase_interval",
    "report_step_event",
    "custom_reward_with_timing",
    "rollout_with_timing_context",
    "log_eval_rollout_data",
    "log_rollout_data",
]
