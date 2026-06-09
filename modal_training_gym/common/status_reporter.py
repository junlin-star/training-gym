"""Fire-and-forget HTTP poster for framework-status updates.

Used by the slime/miles launcher and by ``common/train.py`` so that
``framework_status`` writes don't block training/orchestration on Modal Volume
round-trips. A single daemon thread drains a bounded queue and POSTs JSON to
the dashboard's ``/api/framework-status`` endpoint.

The dashboard URL is resolved at enqueue time from:
1. Explicit ``url`` argument
2. ``TRAINING_GYM_FRAMEWORK_STATUS_URL`` env var
3. ``SLIME_PHASE_REPORT_URL`` env var (same payload contract)
4. ``~/.training-gym.toml`` via :mod:`modal_training_gym.common.config`

If none of those resolve, ``enqueue`` is a no-op — training continues, just
without dashboard updates.
"""

from __future__ import annotations

import json
import os
import threading
from queue import Queue
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


_QUEUE: Queue[dict[str, Any] | None] = Queue(maxsize=256)
_STARTED = False
_LOCK = threading.Lock()
_DEFAULT_TIMEOUT_SECONDS = 2.0
_STATUS_TOKEN_ENV = "TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"


def _resolve_url() -> str:
    url = (
        os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL")
        or os.environ.get("SLIME_PHASE_REPORT_URL")
        or ""
    ).strip()
    if url:
        return url
    try:
        from modal_training_gym.common.config import get_framework_status_url

        return get_framework_status_url() or ""
    except Exception:
        return ""


def _resolve_token() -> str:
    return os.environ.get(_STATUS_TOKEN_ENV, "").strip()


def _ensure_worker() -> None:
    global _STARTED
    if _STARTED:
        return
    with _LOCK:
        if _STARTED:
            return
        thread = threading.Thread(
            target=_worker, name="training-gym-status-reporter", daemon=True
        )
        thread.start()
        _STARTED = True


def _worker() -> None:
    while True:
        item = _QUEUE.get()
        if item is None:
            return
        try:
            _post(item)
        finally:
            _QUEUE.task_done()


def _post(item: dict[str, Any]) -> None:
    url = item.pop("_url", "")
    timeout = float(
        item.pop("_timeout", _DEFAULT_TIMEOUT_SECONDS) or _DEFAULT_TIMEOUT_SECONDS
    )
    token = str(item.pop("_token", "") or "").strip() or _resolve_token()
    if not url:
        return
    body = json.dumps(item, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}
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


def enqueue_framework_status(
    training_run_id: str,
    phase: str,
    *,
    url: str | None = None,
    token: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    **extra: Any,
) -> None:
    """Fire a framework-status POST in the background.

    Returns immediately. If no dashboard URL is configured the call is a
    silent no-op.
    """
    resolved = (url or "").strip() or _resolve_url()
    if not resolved or not training_run_id or not phase:
        return
    _ensure_worker()
    payload: dict[str, Any] = {
        "_url": resolved,
        "_timeout": timeout_seconds,
        "_token": (token or "").strip() or _resolve_token(),
        "training_run_id": training_run_id,
        "phase": phase,
        **extra,
    }
    try:
        _QUEUE.put_nowait(payload)
    except Exception:
        pass


def flush(timeout_seconds: float = 5.0) -> None:
    """Block until the queue drains (best-effort). Used at process exit so
    terminal-state writes have a chance to land before the container dies."""
    deadline = threading.Event()
    timer = threading.Timer(timeout_seconds, deadline.set)
    timer.daemon = True
    timer.start()
    try:
        while not _QUEUE.empty() and not deadline.is_set():
            deadline.wait(0.05)
    finally:
        timer.cancel()
