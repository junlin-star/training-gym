"""Container-side substep timing recorder, shared by slime and miles.

Runs inside the training image, so is dependency-light and importable without a
framework, torch or pydantic present.

A record measures one ``(rollout_id, role)`` lane: per phase, how many times it
ran, its summed and longest duration, when it first started and last ended, and
each run's start and end unless it ran once per sample, which is thousands of
runs a step.
"""

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

PER_SAMPLE_PHASES = frozenset({"reward", "sample_generation"})

_CLOSED_POSTERS: list[threading.Thread] = []


def _drain_closed_posters() -> None:
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
        rollout_id: int | None,
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
        self._lock = threading.Lock()
        self._snapshot: dict[str, object] | None = None
        self._posted_phases: dict[str, dict[str, object]] | None = None
        self._snapshot_ready = threading.Event()
        self._poster: threading.Thread | None = None
        self._closed = False
        self._unsupported = False

    def __enter__(self) -> "RoleRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        """Hand the final snapshot over without waiting for it to be sent.

        A lane closes on the framework's own loop, which in the async
        entrypoints is the loop the next rollout runs on, so waiting here for a
        POST would stall training on an unresponsive dashboard. The sender is
        joined at process exit instead, the only point the snapshot could
        otherwise be lost at.
        """
        self._publish(force=True)
        if self._poster is not None:
            _CLOSED_POSTERS.append(self._poster)
        self._closed = True
        self._snapshot_ready.set()

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
                    self.invocations[name] = (
                        []
                        if name in PER_SAMPLE_PHASES
                        else [[start - self._t0, end - self._t0]]
                    )
                else:
                    timing["count"] += 1
                    timing["total_duration_s"] += duration
                    timing["longest_duration_s"] = max(
                        timing["longest_duration_s"], duration
                    )
                    timing["first_start_s"] = min(
                        timing["first_start_s"], start - self._t0
                    )
                    timing["last_end_s"] = max(timing["last_end_s"], end - self._t0)
                    if name not in PER_SAMPLE_PHASES:
                        self.invocations[name].append(
                            [start - self._t0, end - self._t0]
                        )
            self._publish()

    def _post_snapshots(self) -> None:
        while True:
            self._snapshot_ready.wait()
            self._snapshot_ready.clear()
            with self._lock:
                snapshot, self._snapshot = self._snapshot, None
            if snapshot is not None and snapshot["phases"] != self._posted_phases:
                self._posted_phases = snapshot["phases"]
                result = status_reporter.post_item_result(dict(snapshot))
                if result == "not_found":
                    self._unsupported = True
                    if os.environ.get(TIMING_MODE_ENV, "auto") == "require":
                        print(
                            "ERROR: substep_timing='require' was rejected with "
                            "HTTP 404; timing is unavailable on this dashboard.",
                            flush=True,
                        )
            if self._closed and self._snapshot is None:
                if self._poster in _CLOSED_POSTERS:
                    _CLOSED_POSTERS.remove(self._poster)
                return

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
        if self._unsupported:
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


def _lowest_rank_publishes() -> bool | None:
    """Whether global rank zero writes this lane; ``None`` until ranks are known.

    The lane is measured on every rank of the model and stored under one key, so
    one rank writes it. This assumes the initialized world process group is the
    model's train group, as established by the current Slime and Miles launchers.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        return True
    if not dist.is_initialized():
        return None
    return dist.get_rank() == min(dist.get_process_group_ranks(dist.group.WORLD))


@contextmanager
def recording_lane_on_reporting_rank(
    rollout_id: int, role: str = "actor"
) -> Iterator[RoleRecorder]:
    """An actor/critic lane measured on every rank, written by one of them."""
    with recording_lane(role, rollout_id, _lowest_rank_publishes) as rec:
        yield rec


__all__ = [
    "RoleRecorder",
    "time_phase",
    "recording_lane",
    "recording_lane_on_reporting_rank",
    "timing_url",
]
