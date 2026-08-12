from __future__ import annotations

import atexit
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

from modal_training_gym.common import status_reporter

TIMING_MODE_ENV = "TRAINING_GYM_SUBSTEP_TIMING"
TIMING_PATH = "/api/timing-events"
STATUS_PATH = "/api/framework-status"
TIMING_TIMEOUT_SECONDS = 10.0

MIN_PUBLISH_INTERVAL_S = 3.0
MAX_PHASE_INVOCATIONS = 10_000
MAX_TIMING_PHASES = 64
MAX_POST_RETRIES = 5
MAX_RETRY_BACKOFF_S = 1.0
AUTH_REJECTION_LATCH_THRESHOLD = 3
NOT_FOUND_LATCH_THRESHOLD = 3

PER_SAMPLE_PHASES = frozenset({"reward", "reward_batch", "sample_generation"})

_CLOSED_POSTERS: list[threading.Thread] = []
_PRELOOP_RECORDERS: dict[str, RoleRecorder] = {}
_PRELOOP_LOCK = threading.Lock()
_UNSUPPORTED = False
_NOT_FOUND_COUNT = 0
_FAILURE_REPORTED = False
_UNSUPPORTED_LOCK = threading.Lock()
_TIMING_MODE_CACHE: str | None = None


def timing_mode() -> str:
    """Return the process timing mode, read once before worker hot paths run."""
    global _TIMING_MODE_CACHE
    if _TIMING_MODE_CACHE is None:
        _TIMING_MODE_CACHE = os.environ.get(TIMING_MODE_ENV, "auto")
    return _TIMING_MODE_CACHE


def reset_timing_mode_cache() -> None:
    """Reset the cached mode for tests that change the worker environment."""
    global _TIMING_MODE_CACHE
    _TIMING_MODE_CACHE = None


class _NoopRecorder:
    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        yield


_NOOP_RECORDER = _NoopRecorder()


def _drain_closed_posters() -> None:
    for recorder in list(_PRELOOP_RECORDERS.values()):
        recorder._close()
    deadline = time.monotonic() + TIMING_TIMEOUT_SECONDS
    for poster in list(_CLOSED_POSTERS):
        poster.join(timeout=max(0.0, deadline - time.monotonic()))


atexit.register(_drain_closed_posters)


def timing_url() -> str:
    base = os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL", "").strip()
    if not base:
        return ""
    if base.endswith(STATUS_PATH):
        base = base[: -len(STATUS_PATH)]
    return base.rstrip("/") + TIMING_PATH


class RoleRecorder:
    def __init__(
        self,
        role: str,
        rollout_id: int | None,
        publish_gate: Callable[[], bool | None] | None = None,
        persistent: bool = False,
    ) -> None:
        self.role = role
        self.rollout_id = rollout_id

        self._publish_gate = publish_gate
        self._gate_answer: bool | None = None
        self._t0 = time.monotonic()
        self.lane_start_unix_s = time.time()
        self.phases: dict[str, dict[str, float]] = {}
        self.invocations: dict[str, list[list[float]]] = {}
        self._last_publish_t = float("-inf")
        self._lock = threading.Lock()
        self._snapshot: dict[str, object] | None = None
        self._posted_phases: dict[str, dict[str, object]] | None = None
        self._last_post_not_found = False
        self._snapshot_ready = threading.Event()
        self._poster: threading.Thread | None = None
        self._closed = False
        self._closing = False
        self._closing_deadline: float | None = None
        self._post_retries = 0
        self._auth_rejections = 0
        self._auth_rejected = False
        self._auth_rejection_reported = False
        self._permanent_reported = False
        self._permanent_rejected = False
        self._persistent = persistent

    def __enter__(self) -> "RoleRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self._publish_on_exit()
        if not self._persistent:
            self._close()

    def _publish_on_exit(self) -> None:
        try:
            self._publish(force=True)
        except Exception:
            pass

    def _close(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._closing_deadline = time.monotonic() + TIMING_TIMEOUT_SECONDS
        self._publish_on_exit()
        if self._poster is not None and self._poster not in _CLOSED_POSTERS:
            _CLOSED_POSTERS.append(self._poster)
        self._closed = True
        self._snapshot_ready.set()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if timing_mode() == "off":
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            end = time.monotonic()
            duration = end - start
            with self._lock:
                timing = self.phases.get(name)
                if timing is None and len(self.phases) >= MAX_TIMING_PHASES:
                    timing = {}
                if timing is None:
                    self.phases[name] = {
                        "count": 1,
                        "total_duration_s": duration,
                        "longest_duration_s": duration,
                        "first_start_s": start - self._t0,
                        "last_end_s": end - self._t0,
                    }
                    self.invocations[name] = (
                        []
                        if name in PER_SAMPLE_PHASES
                        else [[start - self._t0, end - self._t0]]
                    )
                elif timing:
                    timing["count"] += 1
                    timing["total_duration_s"] += duration
                    timing["longest_duration_s"] = max(
                        timing["longest_duration_s"], duration
                    )
                    timing["first_start_s"] = min(
                        timing["first_start_s"], start - self._t0
                    )
                    timing["last_end_s"] = max(timing["last_end_s"], end - self._t0)
                    if (
                        name not in PER_SAMPLE_PHASES
                        and len(self.invocations[name]) < MAX_PHASE_INVOCATIONS
                    ):
                        self.invocations[name].append(
                            [start - self._t0, end - self._t0]
                        )
            try:
                self._publish()
            except Exception:
                pass

    def _post_snapshots(self) -> None:
        global _NOT_FOUND_COUNT, _FAILURE_REPORTED
        global _UNSUPPORTED
        while True:
            try:
                self._snapshot_ready.wait()
                self._snapshot_ready.clear()
                with self._lock:
                    snapshot, self._snapshot = self._snapshot, None
                if snapshot is not None and snapshot["phases"] != self._posted_phases:
                    result = status_reporter.post_item_result(dict(snapshot))
                    if result == status_reporter.PostResult.OK:
                        self._last_post_not_found = False
                        self._posted_phases = snapshot["phases"]
                        self._post_retries = 0
                        self._auth_rejections = 0
                        with _UNSUPPORTED_LOCK:
                            _NOT_FOUND_COUNT = 0
                    elif result == status_reporter.PostResult.NOT_FOUND:
                        self._last_post_not_found = True
                        should_report = False
                        with _UNSUPPORTED_LOCK:
                            _NOT_FOUND_COUNT += 1
                            if _NOT_FOUND_COUNT >= NOT_FOUND_LATCH_THRESHOLD:
                                _UNSUPPORTED = True
                                if not _FAILURE_REPORTED:
                                    _FAILURE_REPORTED = True
                                    should_report = True
                        if should_report:
                            message = (
                                "WARNING: this dashboard is too old for substep "
                                "timing (HTTP 404/405 from its timing endpoint); "
                                "redeploy the dashboard to record it."
                            )
                            print(message, flush=True)
                    elif result == status_reporter.PostResult.AUTH_FAILED:
                        self._auth_rejections += 1
                        if self._auth_rejections >= AUTH_REJECTION_LATCH_THRESHOLD:
                            with self._lock:
                                self._auth_rejected = True
                                should_report = not self._auth_rejection_reported
                                self._auth_rejection_reported = True
                            if should_report:
                                print(
                                    "WARNING: substep timing upload authentication "
                                    "failed repeatedly; disabling this timing lane.",
                                    flush=True,
                                )
                        else:
                            with self._lock:
                                if self._snapshot is None:
                                    self._snapshot = snapshot
                            self._snapshot_ready.set()
                    elif result == status_reporter.PostResult.PERMANENT:
                        with self._lock:
                            self._permanent_rejected = True
                            should_report = not self._permanent_reported
                            self._permanent_reported = True
                        if should_report:
                            print(
                                "WARNING: substep timing upload rejected with a "
                                "permanent client error; dropping snapshot.",
                                flush=True,
                            )
                    elif result == status_reporter.PostResult.FAILED:
                        self._auth_rejections = 0
                        self._post_retries += 1
                        if self._post_retries >= MAX_POST_RETRIES:
                            self._post_retries = 0
                            with _UNSUPPORTED_LOCK:
                                should_report = not _FAILURE_REPORTED
                                _FAILURE_REPORTED = True
                            if should_report:
                                print(
                                    "WARNING: substep timing upload failed after "
                                    f"{MAX_POST_RETRIES} attempts; check dashboard "
                                    "authentication or connectivity.",
                                    flush=True,
                                )
                            if self._closing:
                                deadline = self._closing_deadline
                                remaining = (
                                    deadline - time.monotonic()
                                    if deadline is not None
                                    else 0.0
                                )
                                if remaining > 0:
                                    with self._lock:
                                        if self._snapshot is None:
                                            self._snapshot = snapshot
                                    time.sleep(min(MAX_RETRY_BACKOFF_S, remaining))
                                    self._snapshot_ready.set()
                        else:
                            with self._lock:
                                if self._snapshot is None:
                                    self._snapshot = snapshot
                            time.sleep(
                                min(
                                    MAX_RETRY_BACKOFF_S,
                                    0.1 * 2 ** (self._post_retries - 1),
                                )
                            )
                            self._snapshot_ready.set()
                if self._closed and self._snapshot is None:
                    if self._poster in _CLOSED_POSTERS:
                        _CLOSED_POSTERS.remove(self._poster)
                    return
            except Exception:
                if self._closing:
                    deadline = self._closing_deadline
                    remaining = (
                        deadline - time.monotonic() if deadline is not None else 0.0
                    )
                    if remaining <= 0:
                        return
                    time.sleep(min(0.05, remaining))
                else:
                    time.sleep(0.05)

    def _publish(self, force: bool = False) -> None:
        with self._lock:
            if self._permanent_rejected or self._auth_rejected or not self.phases:
                return
        if timing_mode() == "off":
            return
        url = timing_url()
        training_run_id = os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", "")
        if not url or not training_run_id:
            return
        with _UNSUPPORTED_LOCK:
            unsupported = _UNSUPPORTED
        if unsupported:
            return
        if self._publish_gate is not None:
            if self._gate_answer is None:
                self._gate_answer = self._publish_gate()
            if self._gate_answer is None and force:
                self._gate_answer = True
            if not self._gate_answer:
                return

        now = time.monotonic()
        if not force and now - self._last_publish_t < MIN_PUBLISH_INTERVAL_S:
            return
        self._last_publish_t = now
        with self._lock:
            phases = {
                name: {
                    **{key: round(value, 6) for key, value in timing.items()},
                    "invocations": [
                        [round(start, 6), round(end, 6)]
                        for start, end in self.invocations[name]
                    ],
                }
                for name, timing in self.phases.items()
            }
        snapshot = {
            "_url": url,
            "_timeout": TIMING_TIMEOUT_SECONDS,
            "_token": os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_TOKEN", ""),
            "training_run_id": training_run_id,
            "rollout_id": self.rollout_id,
            "role": self.role,
            "lane_start_unix_s": self.lane_start_unix_s,
            "phases": phases,
        }
        with self._lock:
            if (
                not force
                and self._snapshot is not None
                and self._snapshot["phases"] == snapshot["phases"]
            ):
                return
            if force and self._last_post_not_found:
                return
            if (
                force
                and self._posted_phases is not None
                and self._posted_phases == snapshot["phases"]
            ):
                return
            self._snapshot = snapshot
            if self._poster is None:
                self._poster = threading.Thread(
                    target=self._post_snapshots,
                    name=f"training-gym-timing-{self.role}-{self.rollout_id}",
                    daemon=True,
                )
                self._poster.start()
        self._snapshot_ready.set()


_ACTIVE_LANE: ContextVar[RoleRecorder | None] = ContextVar(
    "training_gym_active_lane", default=None
)


@contextmanager
def time_phase(name: str) -> Iterator[None]:
    if timing_mode() == "off":
        yield
        return
    rec = _ACTIVE_LANE.get()
    if rec is None:
        yield
        return
    with rec.phase(name):
        yield


@contextmanager
def recording_lane(
    role: str,
    rollout_id: int | None,
    publish_gate: Callable[[], bool | None] | None = None,
) -> Iterator[RoleRecorder | _NoopRecorder]:
    if timing_mode() == "off":
        token = _ACTIVE_LANE.set(None)
        try:
            yield _NOOP_RECORDER
        finally:
            _ACTIVE_LANE.reset(token)
        return
    if rollout_id is None:
        with _PRELOOP_LOCK:
            rec = _PRELOOP_RECORDERS.get(role)
            if rec is None:
                rec = RoleRecorder(role, rollout_id, publish_gate, persistent=True)
                _PRELOOP_RECORDERS[role] = rec
    else:
        rec = RoleRecorder(role, rollout_id, publish_gate)
    token = _ACTIVE_LANE.set(rec)
    try:
        with rec:
            yield rec
    finally:
        _ACTIVE_LANE.reset(token)


def _lowest_rank_publishes() -> bool | None:
    try:
        import torch.distributed as dist
    except ImportError:
        return True
    if not dist.is_initialized():
        return None
    get_group_ranks = getattr(dist, "get_process_group_ranks", None)
    if get_group_ranks is None:
        return dist.get_rank() == 0
    return dist.get_rank() == min(get_group_ranks(dist.group.WORLD))


@contextmanager
def recording_lane_on_reporting_rank(
    rollout_id: int, role: str = "actor"
) -> Iterator[RoleRecorder | _NoopRecorder]:
    with recording_lane(role, rollout_id, _lowest_rank_publishes) as rec:
        yield rec


__all__ = [
    "RoleRecorder",
    "time_phase",
    "recording_lane",
    "recording_lane_on_reporting_rank",
    "timing_url",
]
