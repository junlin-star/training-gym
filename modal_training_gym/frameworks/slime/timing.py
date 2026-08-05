"""Container-side substep timing recorder.

Runs inside the slime training image along with :mod:`.reporting`, so it is
dependency-light and importable without slime, torch or pydantic present.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator

from .reporting import _enqueue_timing


MIN_PUBLISH_INTERVAL_S = 1.0
MAX_INTERVALS_PER_PHASE = 512


class RoleRecorder:
    """Accumulates measured intervals for one ``(rollout_id, role)``.

    Constructed by the patched driver loop (as ``_tg_rec``) and by
    :func:`recording_lane` for the other lanes; publishes dicts to
    ``reporting._enqueue_timing`` which posts to ``/api/timing-events``.

    Two timings exist for visibility: monotonic offset relative to ``_t0``
    to guarantee positive duration, and ``lane_start_unix_s`` for wall-clock time
    to align multiple processes in visualization.

    Publishing is gated so that, for actor/critic lanes, only the
    reporting megatron rank writes the timing file for a rollout.
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
        self.phases: dict[str, list[list[float]]] = {}
        # {phase: (count, summed duration, latest end)} for the intervals past
        # MAX_INTERVALS_PER_PHASE, which are measured but not individually
        # stored. Everything phase_totals prints is derivable from these three.
        self.overflow: dict[str, tuple[int, float, float]] = {}
        self._last_publish_t = float("-inf")

    def __enter__(self) -> "RoleRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._publish(force=True)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Times a phase even if it raises."""
        start = time.monotonic()
        try:
            yield
        finally:
            begin = round(start - self._t0, 6)
            end = round(time.monotonic() - self._t0, 6)
            intervals = self.phases.setdefault(name, [])
            if len(intervals) < MAX_INTERVALS_PER_PHASE:
                intervals.append([begin, end])
            else:
                count, total, last_end = self.overflow.get(name, (0, 0.0, 0.0))
                self.overflow[name] = (
                    count + 1,
                    round(total + end - begin, 6),
                    max(last_end, end),
                )
            self._publish()

    def _publish(self, force: bool = False) -> None:
        """Post a whole-file overwrite of the step timing record so far."""
        if not self.phases:
            return
        if self._publish_gate is not None:
            if self._gate_answer is None:
                self._gate_answer = self._publish_gate()
            if not self._gate_answer:
                return

        now = time.monotonic()
        # Rate limiting, mostly for custom reward driver
        if not force and now - self._last_publish_t < MIN_PUBLISH_INTERVAL_S:
            return
        self._last_publish_t = now
        try:
            _enqueue_timing(
                {
                    "rollout_id": self.rollout_id,
                    "role": self.role,
                    "lane_start_unix_s": self.lane_start_unix_s,
                    "phases": {
                        name: [list(interval) for interval in intervals]
                        for name, intervals in self.phases.items()
                    },
                    "overflow": dict(self.overflow),
                }
            )
        except Exception:
            pass


_ACTIVE_LANE: ContextVar[RoleRecorder | None] = ContextVar(
    "training_gym_active_lane", default=None
)


def time_phase(name: str) -> Iterator[None]:
    """Record a phase on the active :class:`RoleRecorder`, if there is one.

    This is a no-op when no lane is active, so injected calls deep in
    slime/megatron/reward code do not crash processes that have no recorder.
    """
    rec = _ACTIVE_LANE.get()
    if rec is None:
        yield
        return
    with rec.phase(name):
        yield


@contextmanager
def recording_lane(role: str, rollout_id: int) -> Iterator[RoleRecorder]:
    """Install a :class:`RoleRecorder` as the active lane for a block."""
    rec = RoleRecorder(role, rollout_id)
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
    rec = RoleRecorder(role, rollout_id, publish_gate=_megatron_publish_gate)
    token = _ACTIVE_LANE.set(rec)
    try:
        with rec:
            yield rec
    finally:
        _ACTIVE_LANE.reset(token)


__all__ = [
    "RoleRecorder",
    "time_phase",
    "recording_lane",
    "recording_lane_on_reporting_rank",
]
