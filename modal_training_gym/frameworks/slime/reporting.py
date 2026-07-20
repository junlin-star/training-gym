"""Background delivery and run-context helpers for slime's in-container
dashboard reporting.

Split out of :mod:`.phase_reporting` (which re-exports these). Everything here
is duck-typed and dependency-light so it stays importable inside the training
container without slime/torch present.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from queue import Full, Queue
from typing import Any, TypeAlias
from urllib.error import URLError
from urllib.request import Request, urlopen

from modal_training_gym.utils.metadata import _step_times_dict

PHASE_REPORT_URL_ENV = "SLIME_PHASE_REPORT_URL"
PHASE_REPORT_TOKEN_ENV = "SLIME_PHASE_REPORT_TOKEN"

# Internal queue entry: each item is {"_url": str, "_timeout": float, **payload}.
# Status reports use the framework-status URL with a short 1s timeout;
# rollout-data reports derive a /api/training-rollouts URL from the same base
# with a longer timeout because payloads can be 100KB+.
_REPORT_QUEUE: "Queue[dict[str, Any] | None]" = Queue(maxsize=512)
_REPORTER_STARTED = False
_REPORTER_LOCK = threading.Lock()
_TIMING_QUEUE_SIZE = 4096
_TIMING_QUEUE: "Queue[TimingQueueItem]" = Queue(maxsize=_TIMING_QUEUE_SIZE)
_TIMING_WORKER_STARTED = False
_TIMING_WORKER_LOCK = threading.Lock()
_TIMING_DROPPED_EVENTS = 0
_TIMING_FAILED_EVENTS = 0
_TIMING_REPORTED_DROPS = 0
_TIMING_REPORTED_FAILURES = 0
_PHASE_PATH = "/api/framework-status"
_ROLLOUT_PATH = "/api/training-rollouts"
_ADVANTAGE_PATH = "/api/advantage-distributions"
_PHASE_TIMEOUT_SECONDS = 1.0
_STEP_EVENT_TIMEOUT_SECONDS = 5.0
_ROLLOUT_TIMEOUT_SECONDS = 10.0
_TIMING_DELIVERY_ATTEMPTS = 5
_TIMING_RETRY_DELAY_SECONDS = 0.1

TimingKey: TypeAlias = tuple[str, str, int, str, int, str, str, int]
TimingEvent: TypeAlias = tuple[TimingKey, dict[str, Any]]
TimingQueueItem: TypeAlias = TimingEvent | threading.Event


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


def _ensure_timing_worker() -> None:
    global _TIMING_WORKER_STARTED
    if _TIMING_WORKER_STARTED:
        return
    with _TIMING_WORKER_LOCK:
        if _TIMING_WORKER_STARTED:
            return
        threading.Thread(
            target=_timing_worker,
            name="slime-timing-reporter",
            daemon=True,
        ).start()
        _TIMING_WORKER_STARTED = True


def _enqueue_async_timing_event(payload: Mapping[str, Any]) -> None:
    global _TIMING_DROPPED_EVENTS
    try:
        training_run_id = str(payload["training_run_id"])
        training_attempt = int(payload.get("training_attempt", 0) or 0)
        training_role = str(payload.get("training_role", ""))
        rollout_id = int(payload["rollout_id"])
        phase = str(payload["phase"])
        step_event = str(payload["step_event"])
        step_id = int(payload.get("step_id", -1))
    except (KeyError, TypeError, ValueError):
        print("Skipping async timing event without a complete identity")
        return
    if not training_run_id or not phase or not step_event:
        print("Skipping async timing event without a complete identity")
        return

    key: TimingKey = (
        training_run_id,
        "timing_event",
        training_attempt,
        training_role,
        rollout_id,
        phase,
        step_event,
        step_id,
    )
    try:
        _ensure_timing_worker()
        _TIMING_QUEUE.put_nowait((key, dict(payload)))
    except Full:
        _TIMING_DROPPED_EVENTS += 1
        if _TIMING_DROPPED_EVENTS == 1:
            print("Async timing queue is full; timing data will be incomplete")
    except Exception as exc:
        _TIMING_DROPPED_EVENTS += 1
        print(f"Failed to queue async timing event: {exc}")


def _timing_worker() -> None:
    global _TIMING_FAILED_EVENTS
    while True:
        item = _TIMING_QUEUE.get()
        if isinstance(item, threading.Event):
            item.set()
            _TIMING_QUEUE.task_done()
            continue

        key, payload = item
        error: Exception | None = None
        # TODO: ask Joy about design of retries
        for attempt in range(1, _TIMING_DELIVERY_ATTEMPTS + 1):
            try:
                _step_times_dict()[key] = payload
                error = None
                break
            except Exception as exc:
                error = exc
                if attempt < _TIMING_DELIVERY_ATTEMPTS:
                    time.sleep(_TIMING_RETRY_DELAY_SECONDS * attempt)
        if error is not None:
            _TIMING_FAILED_EVENTS += 1
            print(
                "Failed to write async timing event after "
                f"{_TIMING_DELIVERY_ATTEMPTS} attempts: {error}"
            )
        _TIMING_QUEUE.task_done()


def flush_async_timing_events(timeout_seconds: float = 5.0) -> bool:
    global _TIMING_REPORTED_DROPS, _TIMING_REPORTED_FAILURES
    if not _TIMING_WORKER_STARTED:
        return True

    barrier = threading.Event()
    deadline = time.monotonic() + timeout_seconds
    try:
        _TIMING_QUEUE.put(barrier, timeout=timeout_seconds)
    except Exception as exc:
        print(f"Failed to flush async timing events: {exc}")
        return False
    flushed = barrier.wait(max(0.0, deadline - time.monotonic()))
    if not flushed:
        print("Timed out flushing async timing events")
        return False
    complete = (
        _TIMING_DROPPED_EVENTS == _TIMING_REPORTED_DROPS
        and _TIMING_FAILED_EVENTS == _TIMING_REPORTED_FAILURES
    )
    _TIMING_REPORTED_DROPS = _TIMING_DROPPED_EVENTS
    _TIMING_REPORTED_FAILURES = _TIMING_FAILED_EVENTS
    return complete


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


def _post_framework_status(payload: dict[str, Any], timeout: float) -> None:
    url = _phase_url()
    if not url:
        return
    _post({"_url": url, "_timeout": timeout, **payload})


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
    except (OSError, URLError):
        return
