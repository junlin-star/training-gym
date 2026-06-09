from __future__ import annotations

import json
import importlib
import os
import threading
from queue import Queue
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from modal_training_gym.common.status import SlimeStatus

PHASE_REPORT_URL_ENV = "SLIME_PHASE_REPORT_URL"
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
_REPORT_QUEUE: Queue[dict[str, Any] | None] = Queue(maxsize=512)
_REPORTER_STARTED = False
_REPORTER_LOCK = threading.Lock()


def _run_context(args: Any) -> dict[str, Any]:
    return {
        "training_run_id": getattr(args, "training_run_id", "") or "",
        "app_name": getattr(args, "app_name", "") or "",
        "modal_app_id": os.environ.get("MODAL_APP_ID", ""),
    }


def _enqueue(payload: dict[str, Any]) -> None:
    global _REPORTER_STARTED
    if not os.environ.get(PHASE_REPORT_URL_ENV, "").strip():
        return

    with _REPORTER_LOCK:
        if not _REPORTER_STARTED:
            thread = threading.Thread(
                target=_worker,
                name="slime-phase-reporter",
                daemon=True,
            )
            thread.start()
            _REPORTER_STARTED = True

    try:
        _REPORT_QUEUE.put_nowait(payload)
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


def _post(payload: dict[str, Any]) -> None:
    url = os.environ.get(PHASE_REPORT_URL_ENV, "").strip()
    if not url:
        return

    body = json.dumps(payload, default=str).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=1) as response:
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
        rollout_id=rollout_id,
        sample_count=len(samples) if hasattr(samples, "__len__") else None,
        metrics=rollout_extra_metrics,
        rollout_time=rollout_time,
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
    report_phase(SlimeStatus.ROLLOUT_INITIALIZING, args)


__all__ = [
    "before_log_prob_hook",
    "before_train_step_hook",
    "report_phase",
    "report_rollout_initializing",
    "log_eval_rollout_data",
    "log_rollout_data",
]
