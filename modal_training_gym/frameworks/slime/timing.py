"""Container-side substep timing recorder.

Runs inside the slime training image along with :mod:`.reporting`, so is
dependency-light and importable without slime, torch or pydantic present. 
"""

from __future__ import annotations
 
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator
 
from .reporting import _count_drop, _enqueue_timing
 
 
MIN_PUBLISH_INTERVAL_S = 1.0

class RoleRecorder:
    """Accumulates measured intervals for one ``(rollout_id, role)``.

    Constructed by the patched driver loop (as ``_tg_rec``) and by
    :func:`role_recorder` for the other lanes; publishes dicts to
    ``reporting._enqueue_timing`` which posts to ``/api/timing-events``.

    Two timings exist for visibility: monotonic offset relative to ``_t0``
    to guarantee positive duration, and ``lane_start_unix_s`` for wall-clock time
    to align multiple processes in visualization. 

    Publishing is gated so that, for actor/critic lanes, only the
    reporting megatron rank writes the timing file for a rollout.”
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
            self.phases.setdefault(name, []).append(
                [
                    round(start - self._t0, 6),
                    round(time.monotonic() - self._t0, 6),
                ]
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
                }
            )
        except Exception:
            _count_drop("timing_publish_error")