"""HTTP queue / URL / token plumbing + run-context helpers for slime's
in-container dashboard reporting.

Split out of :mod:`.phase_reporting` (which re-exports these). Everything here
is duck-typed and dependency-light so it stays importable inside the training
container without slime/torch present.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from queue import Full, Queue
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

PHASE_REPORT_URL_ENV = "SLIME_PHASE_REPORT_URL"
PHASE_REPORT_TOKEN_ENV = "SLIME_PHASE_REPORT_TOKEN"

# Each report carries its destination and timeout alongside the JSON payload.
_REPORT_QUEUE: "Queue[dict[str, Any] | threading.Event]" = Queue(maxsize=512)
_REPORTER_STARTED = False
_REPORTER_LOCK = threading.Lock()
_PHASE_PATH = "/api/framework-status"
_TRAINING_TIMING_EVENT_PATH = "/api/timing-events"
_ROLLOUT_PATH = "/api/training-rollouts"
_ADVANTAGE_PATH = "/api/advantage-distributions"
_PHASE_TIMEOUT_SECONDS = 1.0
_TIMING_EVENT_TIMEOUT_SECONDS = 2.0
_COMPLETED_STEP_TIMEOUT_SECONDS = 5.0
_ROLLOUT_TIMEOUT_SECONDS = 10.0
_TRAINING_TIMING_EVENT_DELIVERY_ATTEMPTS = 3
_TRAINING_TIMING_EVENT_RETRY_DELAY_SECONDS = 0.1
_COMPLETED_STEP_DELIVERY_ATTEMPTS = 10
_COMPLETED_STEP_RETRY_DELAY_SECONDS = 0.1
_REPORT_FLUSH_TIMEOUT_SECONDS = 30.0


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
    context: dict[str, Any] = {
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
    training_attempt = _positive_int(os.environ.get("TRAINING_GYM_TRAINING_ATTEMPT"))
    if training_attempt is not None:
        context["training_attempt"] = training_attempt
    return context


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
    return (
        os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL", "")
        or os.environ.get(PHASE_REPORT_URL_ENV, "")
    ).strip()


def _derive_url(path: str) -> str:
    base = _phase_url()
    if not base:
        return ""
    if base.endswith(_PHASE_PATH):
        return base[: -len(_PHASE_PATH)] + path
    return base.rstrip("/") + path


def _rollout_url() -> str:
    return _derive_url(_ROLLOUT_PATH)


def _training_timing_event_url() -> str:
    return _derive_url(_TRAINING_TIMING_EVENT_PATH)


def _advantage_url() -> str:
    return _derive_url(_ADVANTAGE_PATH)


def _report_token() -> str:
    return (
        os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_TOKEN", "")
        or os.environ.get(PHASE_REPORT_TOKEN_ENV, "")
    ).strip()


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
        atexit.register(flush_dashboard_reports)


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


def _enqueue_timing_event(payload: dict[str, Any]) -> None:
    if os.environ.get("TRAINING_GYM_ASYNC_MODE") == "1":
        return
    if payload.get("training_attempt") is None:
        return
    url = _training_timing_event_url()
    if not url:
        return
    try:
        _ensure_worker()
        _REPORT_QUEUE.put_nowait(
            {
                "_url": url,
                "_timeout": _TIMING_EVENT_TIMEOUT_SECONDS,
                "_delivery_attempts": _TRAINING_TIMING_EVENT_DELIVERY_ATTEMPTS,
                "_delivery_retry_delay": _TRAINING_TIMING_EVENT_RETRY_DELAY_SECONDS,
                "_failure_message": "Failed to deliver training timing event",
                **payload,
            }
        )
    except Full:
        print("Reporting queue is full; training timing data will be incomplete")
    except Exception as exc:
        print(f"Failed to queue training timing event: {exc}")


def flush_dashboard_reports(
    timeout_seconds: float = _REPORT_FLUSH_TIMEOUT_SECONDS,
) -> bool:
    if not _REPORTER_STARTED:
        return True
    barrier = threading.Event()
    deadline = time.monotonic() + timeout_seconds
    try:
        _REPORT_QUEUE.put(barrier, timeout=timeout_seconds)
    except Full:
        print("Failed to flush dashboard reports because the queue remained full")
        return False
    if barrier.wait(max(0.0, deadline - time.monotonic())):
        return True
    print("Timed out flushing dashboard reports")
    return False


def _enqueue_completed_step_status(payload: dict[str, Any]) -> None:
    url = _phase_url()
    if not url:
        return
    _ensure_worker()
    item = {
        "_url": url,
        "_timeout": _COMPLETED_STEP_TIMEOUT_SECONDS,
        "_delivery_attempts": _COMPLETED_STEP_DELIVERY_ATTEMPTS,
        "_delivery_retry_delay": _COMPLETED_STEP_RETRY_DELAY_SECONDS,
        "_failure_message": "Failed to deliver completed step status",
        **payload,
    }
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Full:
        print("Reporting queue is full; completed step status will be incomplete")
    except Exception as exc:
        print(f"Failed to queue completed step status: {exc}")


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


def _enqueue_advantage(payload: dict[str, Any]) -> None:
    """Enqueue an advantage-distribution payload (longer timeout like rollouts)."""
    url = _advantage_url()
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
        payload = _REPORT_QUEUE.get()
        if isinstance(payload, threading.Event):
            payload.set()
            _REPORT_QUEUE.task_done()
            continue
        delivery_attempts = int(payload.pop("_delivery_attempts", 1))
        retry_delay = float(payload.pop("_delivery_retry_delay", 0.0))
        failure_message = payload.pop("_failure_message", "")
        try:
            succeeded = _post_with_retries(
                payload,
                attempts=delivery_attempts,
                retry_delay_seconds=retry_delay,
            )
            if failure_message and not succeeded:
                print(f"{failure_message} after {delivery_attempts} attempts")
        finally:
            _REPORT_QUEUE.task_done()


def _post_with_retries(
    item: dict[str, Any], *, attempts: int, retry_delay_seconds: float
) -> bool:
    for attempt in range(1, attempts + 1):
        if _post(dict(item)):
            return True
        if attempt < attempts:
            time.sleep(retry_delay_seconds * attempt)
    return False


def _post(item: dict[str, Any]) -> bool:
    url = item.pop("_url", "")
    timeout = float(
        item.pop("_timeout", _PHASE_TIMEOUT_SECONDS) or _PHASE_TIMEOUT_SECONDS
    )
    if not url:
        return False

    body = json.dumps(item, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = _report_token()
    if token:
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
        return True
    except (OSError, URLError):
        return False
