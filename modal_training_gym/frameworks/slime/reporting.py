"""HTTP queue / URL / token plumbing + run-context helpers for slime's
in-container dashboard reporting.

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
from queue import Queue
from typing import Any
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
_PHASE_PATH = "/api/framework-status"
_ROLLOUT_PATH = "/api/training-rollouts"
_ADVANTAGE_PATH = "/api/advantage-distributions"
_PHASE_TIMEOUT_SECONDS = 1.0
_STEP_EVENT_TIMEOUT_SECONDS = 5.0
_ROLLOUT_TIMEOUT_SECONDS = 10.0
_TIMING_RETRY_DELAY_SECONDS = 0.1
_TIMING_RETRY_MAX_DELAY_SECONDS = 5.0


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


def _persist_async_timing_event(payload: Mapping[str, object]) -> None:
    training_run_id = str(payload.get("training_run_id") or "")
    phase = payload.get("phase")
    step_event = payload.get("step_event")
    training_role = payload.get("training_role", "")
    training_attempt = payload.get("training_attempt", 0)
    rollout_id = payload.get("rollout_id")
    step_id = payload.get("step_id")
    number_like = (int, float, str)
    if (
        not isinstance(training_attempt, number_like)
        or not isinstance(rollout_id, number_like)
        or (step_id is not None and not isinstance(step_id, number_like))
        or not isinstance(training_role, str)
    ):
        print("Skipping async timing event without a complete identity")
        return
    try:
        training_attempt = int(training_attempt or 0)
        rollout_id = int(rollout_id)
        step_id = int(step_id) if step_id is not None else -1
    except (TypeError, ValueError):
        print("Skipping async timing event without a complete identity")
        return
    if (
        not training_run_id
        or not isinstance(phase, str)
        or not isinstance(step_event, str)
    ):
        print("Skipping async timing event without a complete identity")
        return

    key = (
        training_run_id,
        "timing_event",
        training_attempt,
        training_role,
        rollout_id,
        phase,
        step_event,
        step_id,
    )
    failures = 0
    while True:
        try:
            _step_times_dict()[key] = dict(payload)
            return
        except Exception as exc:
            failures += 1
            if failures == 1 or failures % 10 == 0:
                print(
                    f"Retrying async timing event after {failures} failed writes: {exc}"
                )
            time.sleep(
                min(
                    _TIMING_RETRY_DELAY_SECONDS * failures,
                    _TIMING_RETRY_MAX_DELAY_SECONDS,
                )
            )


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
