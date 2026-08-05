"""Container-side substep timing recorder, shared by slime and miles.

Runs inside the training image, so it is dependency-light: stdlib plus
:mod:`.status_reporter`, which owns the background HTTP queue. It must stay
importable in a container without torch, pydantic or a framework present, since
build-time patches inject calls into framework source that also runs on ranks
that record nothing.

A record measures one ``(rollout_id, role)`` lane. Per phase it keeps how many
times the phase ran, the summed and longest duration, and when it first started
and last ended, so the dashboard can show the longest and the average without
shipping one entry per measurement -- a reward phase runs once per sample,
thousands of times per step, and every publish re-sends the whole record.
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

# Publishing every measurement would be one POST per reward sample; the record
# is cumulative, so a snapshot at most this often loses nothing but detail in
# how a step's timing filled in while it ran.
MIN_PUBLISH_INTERVAL_S = 1.0


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

    Two clocks, for two jobs: durations are monotonic offsets from ``_t0`` so
    they cannot go backwards, and ``lane_start_unix_s`` is wall clock, used only
    to line up lanes recorded in different processes.

    Publishing is gated so that, for actor/critic lanes, only the reporting
    megatron rank writes the timing file for a rollout.
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
        self._last_publish_t = float("-inf")
        # A lane can be measured from several threads at once: rewards are
        # scored concurrently, and the framework runs generation in a worker
        # thread that inherits the lane. Accumulating without this loses runs.
        self._lock = threading.Lock()

    def __enter__(self) -> "RoleRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        # Forced, because the rate limit would otherwise hold back the tail of
        # the step -- the part that matters when a step fails or is preempted.
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
                else:
                    timing["count"] += 1
                    timing["total_duration_s"] += duration
                    timing["longest_duration_s"] = max(
                        timing["longest_duration_s"], duration
                    )
                    # Concurrent runs neither start nor finish in order, so the
                    # span is the earliest start to the latest end.
                    timing["first_start_s"] = min(
                        timing["first_start_s"], start - self._t0
                    )
                    timing["last_end_s"] = max(timing["last_end_s"], end - self._t0)
            self._publish()

    def _publish(self, force: bool = False) -> None:
        """Post the record so far; the dashboard overwrites the stored lane."""
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
            if not self._gate_answer:
                return

        now = time.monotonic()
        if not force and now - self._last_publish_t < MIN_PUBLISH_INTERVAL_S:
            return
        self._last_publish_t = now
        with self._lock:
            phases = {
                name: {key: round(value, 6) for key, value in timing.items()}
                for name, timing in self.phases.items()
            }
        status_reporter.enqueue_item(
            {
                "_url": url,
                "_timeout": TIMING_TIMEOUT_SECONDS,
                # Each post replaces the lane's stored record, so a queued
                # older snapshot of this lane is dead weight.
                "_supersede_key": (training_run_id, self.rollout_id, self.role),
                "_token": os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_TOKEN", ""),
                "training_run_id": training_run_id,
                "rollout_id": self.rollout_id,
                "role": self.role,
                "lane_start_unix_s": self.lane_start_unix_s,
                "phases": phases,
            }
        )


_ACTIVE_LANE: ContextVar[RoleRecorder | None] = ContextVar(
    "training_gym_active_lane", default=None
)


@contextmanager
def time_phase(name: str) -> Iterator[None]:
    """Record a phase on the active :class:`RoleRecorder`, if there is one.

    This is a no-op when no lane is active, so injected calls deep in
    framework/megatron/reward code do not crash processes that have no recorder.
    """
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


def _megatron_publish_gate() -> bool | None:
    """Return whether this megatron rank is the unique writer for its lane."""
    try:
        from megatron.core import parallel_state as ps

        if not ps.model_parallel_is_initialized():
            return None
        return (
            ps.get_tensor_model_parallel_rank() == 0
            and ps.get_pipeline_model_parallel_rank()
            == ps.get_pipeline_model_parallel_world_size() - 1
            and ps.get_context_parallel_rank() == 0
            and ps.get_data_parallel_rank() == 0
        )
    except ImportError:
        pass
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return None
        return dist.get_rank() == 0
    except ImportError:
        return True


@contextmanager
def recording_lane_on_reporting_rank(
    rollout_id: int, role: str = "actor"
) -> Iterator[RoleRecorder]:
    """An actor/critic lane measured on every rank, written by one of them."""
    with recording_lane(role, rollout_id, _megatron_publish_gate) as rec:
        yield rec


__all__ = [
    "RoleRecorder",
    "time_phase",
    "recording_lane",
    "recording_lane_on_reporting_rank",
    "timing_url",
]
