from __future__ import annotations

import json
import importlib
import os
import threading
import time
from queue import Queue
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from modal_training_gym.common.status import SlimeStatus

PHASE_REPORT_URL_ENV = "SLIME_PHASE_REPORT_URL"
PHASE_REPORT_TOKEN_ENV = "SLIME_PHASE_REPORT_TOKEN"
# Import path of the run model's response parser (a (str) -> ParsedResponse
# callable). The launcher exports it; the recorder resolves and applies it.
RESPONSE_PARSER_PATH_ENV = "TRAINING_GYM_RESPONSE_PARSER_PATH"
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

# Internal queue entry: each item is {"_url": str, "_timeout": float, **payload}.
# Status reports use the SLIME_PHASE_REPORT_URL with a short 1s timeout;
# rollout-data reports derive a /api/training-rollouts URL from the same base
# with a longer timeout because payloads can be 100KB+.
_REPORT_QUEUE: Queue[dict[str, Any] | None] = Queue(maxsize=512)
_REPORTER_STARTED = False
_REPORTER_LOCK = threading.Lock()
_PHASE_PATH = "/api/framework-status"
_ROLLOUT_PATH = "/api/training-rollouts"
_PHASE_TIMEOUT_SECONDS = 1.0
_ROLLOUT_TIMEOUT_SECONDS = 10.0


def _arg_value(args: Any, key: str) -> Any:
    value = getattr(args, key, None)
    if value not in (None, ""):
        return value

    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            value = container.get(key)
            if value not in (None, ""):
                return value
    return None


def _run_context(args: Any) -> dict[str, Any]:
    return {
        "training_run_id": _arg_value(args, "training_run_id")
        or _arg_value(args, "training_gym_training_run_id")
        or os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", "")
        or "",
        "app_name": _arg_value(args, "app_name")
        or _arg_value(args, "training_gym_app_name")
        or os.environ.get("TRAINING_GYM_APP_NAME", "")
        or "",
        "modal_app_id": os.environ.get("MODAL_APP_ID", ""),
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _total_steps(args: Any) -> int | None:
    for key in ("num_rollout", "training_gym_total_steps"):
        total = _positive_int(_arg_value(args, key))
        if total is not None:
            return total
    return _positive_int(os.environ.get("TRAINING_GYM_TOTAL_STEPS"))


def _step_progress(args: Any, rollout_id: int | None = None) -> dict[str, Any]:
    total = _total_steps(args)
    if rollout_id is None:
        current = 0
    else:
        current = max(0, int(rollout_id) + 1)
    if total is not None:
        current = min(current, total)
    return {
        "progress_current": current,
        "progress_total": total,
        "progress_unit": "step",
    }


def _phase_url() -> str:
    return os.environ.get(PHASE_REPORT_URL_ENV, "").strip()


def _rollout_url() -> str:
    base = _phase_url()
    if not base:
        return ""
    if base.endswith(_PHASE_PATH):
        return base[: -len(_PHASE_PATH)] + _ROLLOUT_PATH
    return base.rstrip("/") + _ROLLOUT_PATH


def _ensure_worker() -> None:
    global _REPORTER_STARTED
    if _REPORTER_STARTED:
        return
    with _REPORTER_LOCK:
        if _REPORTER_STARTED:
            return
        thread = threading.Thread(
            target=_worker,
            name="slime-phase-reporter",
            daemon=True,
        )
        thread.start()
        _REPORTER_STARTED = True


def _enqueue(payload: dict[str, Any]) -> None:
    """Enqueue a framework-status payload (small, 1s timeout)."""
    url = _phase_url()
    if not url:
        return
    _ensure_worker()
    item = {"_url": url, "_timeout": _PHASE_TIMEOUT_SECONDS, **payload}
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Exception:
        pass


def _enqueue_rollout(payload: dict[str, Any]) -> None:
    """Enqueue a rollout-data payload (large, longer timeout)."""
    url = _rollout_url()
    if not url:
        return
    _ensure_worker()
    item = {"_url": url, "_timeout": _ROLLOUT_TIMEOUT_SECONDS, **payload}
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Exception:
        pass


def _worker() -> None:
    while True:
        try:
            payload = _REPORT_QUEUE.get()
        except Exception:
            continue
        if payload is None:
            return
        try:
            _post(payload)
        finally:
            _REPORT_QUEUE.task_done()


def _post(item: dict[str, Any]) -> None:
    url = item.pop("_url", "")
    timeout = float(
        item.pop("_timeout", _PHASE_TIMEOUT_SECONDS) or _PHASE_TIMEOUT_SECONDS
    )
    if not url:
        return

    body = json.dumps(item, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get(PHASE_REPORT_TOKEN_ENV, "").strip():
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
    except (OSError, URLError):
        return


def _resolve_hook(path: str | None) -> Any:
    if not path:
        return None
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, attr, None)


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
    _enqueue({**_run_context(args), "phase": status.value, **extra})


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


_RESPONSE_PARSER: Any = None
_RESPONSE_PARSER_LOADED = False


def _response_parser() -> Any:
    """Resolve the run model's response parser — a ``(str) -> ParsedResponse``
    callable — from the import path the launcher exports in
    ``RESPONSE_PARSER_PATH_ENV``. Cached per process; ``None`` if unset."""
    global _RESPONSE_PARSER, _RESPONSE_PARSER_LOADED
    if not _RESPONSE_PARSER_LOADED:
        _RESPONSE_PARSER_LOADED = True
        _RESPONSE_PARSER = _resolve_hook(
            os.environ.get(RESPONSE_PARSER_PATH_ENV, "").strip()
        )
    return _RESPONSE_PARSER


def _parsed_response_dict(text: str, parser: Any) -> dict[str, Any] | None:
    """Run ``text`` through ``parser`` and return a JSON-able dict, or ``None``
    when there's no text or no parser."""
    if parser is None or not text:
        return None
    try:
        parsed = parser(text)
    except Exception:
        return None
    return {
        "content": getattr(parsed, "content", "") or "",
        "thinking": getattr(parsed, "thinking", None),
        "tool_calls": [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in (getattr(parsed, "tool_calls", None) or [])
        ],
    }


def _sample_to_dict(sample: Any, parser: Any = None) -> dict[str, Any]:
    """Best-effort extraction of (prompt, response, reward, metadata) from a
    slime Sample-like object. Duck-typed so we don't import slime here."""
    if isinstance(sample, dict):
        attrs = sample
        get = sample.get
    else:
        attrs = None

        def get(key: str, default: Any = None) -> Any:
            return getattr(sample, key, default)

    prompt = get("prompt") if attrs is not None else get("prompt", "")
    response = get("response") if attrs is not None else get("response", "")
    reward = get("reward") if attrs is not None else get("reward", 0.0)

    metadata: dict[str, Any] = {}
    for key in ("response_length", "prompt_length", "rollout_id", "rollout_idx"):
        value = get(key) if attrs is not None else get(key, None)
        if value is not None:
            metadata[key] = value

    response_text = _coerce_text(response)
    out: dict[str, Any] = {
        "score": _coerce_score(reward),
        "prompt": _coerce_text(prompt),
        "response": response_text,
        "metadata": metadata,
    }
    # Store raw + parsed (mirrors eval's EvalRowResult) so the dashboard can show
    # cleaned content without re-parsing. Parsing happens here, in the recorder.
    parsed = _parsed_response_dict(response_text, parser)
    if parsed is not None:
        out["parsed_response"] = parsed
    return out


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    if isinstance(metrics, dict):
        return {str(k): v for k, v in metrics.items()}
    return {}


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
    try:
        sample_dicts = [_sample_to_dict(s, parser) for s in samples]
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


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: Any,
    rollout_time: Any,
) -> bool:
    report_phase(
        SlimeStatus.ROLLOUT_LOGGING,
        args,
        **_step_progress(args, rollout_id),
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
    report_phase(
        SlimeStatus.COMPUTE_LOG_PROBS,
        args,
        store_prefix=store_prefix,
    )
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


def report_rollout_initializing(args: Any) -> None:
    report_phase(
        SlimeStatus.ROLLOUT_INITIALIZING,
        args,
        **_step_progress(args),
    )


def report_weight_sync(args: Any) -> None:
    report_phase(
        SlimeStatus.WEIGHT_SYNC,
        args,
    )


__all__ = [
    "before_log_prob_hook",
    "before_train_step_hook",
    "report_phase",
    "report_rollout_initializing",
    "report_rollout_samples",
    "report_weight_sync",
    "log_eval_rollout_data",
    "log_rollout_data",
]
