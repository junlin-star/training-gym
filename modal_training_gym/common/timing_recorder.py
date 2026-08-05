"""Container-side substep timing recorder, shared by slime and miles.

Runs inside the training image, so is dependency-light and importable without a
framework, torch or pydantic present.

A record measures one ``(rollout_id, role)`` lane: per phase, how many times it
ran, its summed and longest duration, when it first started and last ended, and
each run's start and end unless it ran once per sample, which is thousands of
runs a step.
"""

from __future__ import annotations

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

MIN_PUBLISH_INTERVAL_S = 1.0

# Past this a phase keeps its aggregate only: every publish re-posts the record.
MAX_DRAWN_INVOCATIONS = 1024


def timing_url() -> str:
    base = os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL", "").strip()
    if not base:
        return ""
    if base.endswith(STATUS_PATH):
        base = base[: -len(STATUS_PATH)]
    return base.rstrip("/") + TIMING_PATH


class RoleRecorder:
    """Accumulates measured phase timings for one ``(rollout_id, role)``.

    Constructed by the patched driver loop (as ``_tg_rec``) and by
    :func:`recording_lane` for the other lanes; publishes to
    ``/api/timing-events``, which overwrites the lane's stored record.

    Two timings exist: monotonic offsets relative to ``_t0`` to guarantee
    positive duration, and ``lane_start_unix_s`` to align multiple processes.

    A lane measured on every rank of a distributed model is published by one of
    them, so the ranks do not overwrite each other's record.
    """

    def __init__(
        self,
        role: str,
        rollout_id: int,
        publish_gate: Callable[[], bool | None] | None = None,
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
        # Rewards are scored concurrently on the same lane
        self._lock = threading.Lock()
        # A snapshot replaces the lane's stored record, so only the newest one
        # is worth posting.
        self._snapshot: dict[str, object] | None = None
        self._snapshot_ready = threading.Event()
        self._poster: threading.Thread | None = None
        self._closed = False

    def __enter__(self) -> "RoleRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        # The poster stops first, so the final snapshot is not overtaken.
        self._closed = True
        self._snapshot_ready.set()
        if self._poster is not None:
            self._poster.join(timeout=TIMING_TIMEOUT_SECONDS)
        self._publish(force=True)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Times a phase, even if it raises."""
        start = time.monotonic()
        try:
            yield
        finally:
            end = time.monotonic()
            duration = end - start
            with self._lock:
                timing = self.phases.get(name)
                if timing is None:
                    self.phases[name] = {
                        "count": 1,
                        "total_duration_s": duration,
                        "longest_duration_s": duration,
                        "first_start_s": start - self._t0,
                        "last_end_s": end - self._t0,
                    }
                    self.invocations[name] = [[start - self._t0, end - self._t0]]
                else:
                    timing["count"] += 1
                    timing["total_duration_s"] += duration
                    timing["longest_duration_s"] = max(
                        timing["longest_duration_s"], duration
                    )
                    # Concurrent runs neither start nor finish in order
                    timing["first_start_s"] = min(
                        timing["first_start_s"], start - self._t0
                    )
                    timing["last_end_s"] = max(timing["last_end_s"], end - self._t0)
                    if timing["count"] > MAX_DRAWN_INVOCATIONS:
                        # Emptied, not truncated: a prefix of the runs would
                        # draw as if the phase had stopped early.
                        self.invocations[name] = []
                    else:
                        self.invocations[name].append(
                            [start - self._t0, end - self._t0]
                        )
            self._publish()

    def _post_snapshots(self) -> None:
        while not self._closed:
            self._snapshot_ready.wait()
            self._snapshot_ready.clear()
            with self._lock:
                snapshot, self._snapshot = self._snapshot, None
            if snapshot is not None:
                status_reporter.post_item(snapshot)

    def _publish(self, force: bool = False) -> None:
        """Snapshot the record so far; the dashboard overwrites the stored lane."""
        if not self.phases:
            return
        if os.environ.get(TIMING_MODE_ENV, "auto") == "off":
            return
        url = timing_url()
        training_run_id = os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", "")
        if not url or not training_run_id:
            return
        if self._publish_gate is not None:
            if self._gate_answer is None:
                self._gate_answer = self._publish_gate()
            if self._gate_answer is None and force:
                # Closing without ranks having formed: this process is the only
                # one measuring the lane, so it is the one that writes it.
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
        if force:
            # The lane is closing, so no later snapshot carries these phases.
            status_reporter.post_item(snapshot)
            return
        with self._lock:
            self._snapshot = snapshot
            # Started holding the lock: two phases ending at once would
            # otherwise each start a poster, and the lane joins only one.
            if self._poster is None:
                self._poster = threading.Thread(
                    target=self._post_snapshots,
                    name=f"training-gym-timing-{self.role}-{self.rollout_id}",
                    daemon=True,
                )
                self._poster.start()
        self._snapshot_ready.set()


# The lane the phases of the running code are recorded on, so that a phase timed
# deep in the framework's call stack (a reward on its event loop thread) finds it
# rather than being passed it. A context variable because a rollout actor can
# have two lanes open at once -- an eval and a generate are separate async calls
# on one actor -- and each of their phases belongs to its own.
_ACTIVE_LANE: ContextVar[RoleRecorder | None] = ContextVar(
    "training_gym_active_lane", default=None
)


@contextmanager
def time_phase(name: str) -> Iterator[None]:
    """Record a phase on the active lane; a no-op when there is none."""
    rec = _ACTIVE_LANE.get()
    if rec is None:
        yield
        return
    with rec.phase(name):
        yield


@contextmanager
def recording_lane(
    role: str,
    rollout_id: int,
    publish_gate: Callable[[], bool | None] | None = None,
) -> Iterator[RoleRecorder]:
    """Install a :class:`RoleRecorder` as the active lane for a block."""
    rec = RoleRecorder(role, rollout_id, publish_gate)
    token = _ACTIVE_LANE.set(rec)
    try:
        with rec:
            yield rec
    finally:
        _ACTIVE_LANE.reset(token)


def _rank_zero_publishes() -> bool | None:
    """Whether this rank writes its lane; ``None`` until ranks are known."""
    try:
        import torch.distributed as dist
    except ImportError:
        return True
    if not dist.is_initialized():
        return None
    return dist.get_rank() == 0


@contextmanager
def recording_lane_on_reporting_rank(
    rollout_id: int, role: str = "actor"
) -> Iterator[RoleRecorder]:
    """An actor/critic lane measured on every rank, written by one of them."""
    with recording_lane(role, rollout_id, _rank_zero_publishes) as rec:
        yield rec


__all__ = [
    "RoleRecorder",
    "time_phase",
    "recording_lane",
    "recording_lane_on_reporting_rank",
    "timing_url",
]
