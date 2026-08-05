from __future__ import annotations

import time
from enum import Enum
from typing import Any, Awaitable

from pydantic import BaseModel, Field

from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.utils.metadata import MetadataStore, vol_list_prefix, vol_put


PROTOCOL = "training-gym-substep-timing"


class Role(str, Enum):
    DRIVER = "driver"
    ROLLOUT = "rollout"
    ACTOR = "actor"
    CRITIC = "critic"


class RoleTimingRecord(BaseModel):
    """One role's measured timing for one rollout.

    Single writer per key, whole-file overwrite, last write wins.
    """

    training_run_id: str
    rollout_id: int = Field(ge=0)
    role: Role
    created_at: int = 0
    lane_start_unix_s: float | None = None
    phases: dict[str, list[tuple[float, float]]] = Field(default_factory=dict)
    # {phase: (count, summed duration, latest end)} for intervals past the
    # writer's per-phase cap: measured, but not stored one by one. A phase
    # measured per sample is thousands of intervals in one step, and a record
    # is re-posted whole on every publish. Folding them keeps every number
    # phase_totals prints exact at a fixed size; only the individual bars go.
    overflow: dict[str, tuple[int, float, float]] = Field(default_factory=dict)

    @property
    def storage_key(self) -> str:
        return f"{self.rollout_id:08d}__{self.role.value}"

    @staticmethod
    def store(training_run_id: str) -> str:
        return f"{MetadataStore.SUBSTEP_TIMING.value}/{training_run_id}"

    @staticmethod
    def step_prefix(rollout_id: int) -> str:
        """Listing prefix that matches every role's file for one rollout."""
        return f"{rollout_id:08d}__"

    def _touch_created_at(self) -> None:
        if not self.created_at:
            self.created_at = int(time.time())

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch_created_at()
        return vol_put(
            self.store(self.training_run_id),
            self.storage_key,
            self.model_dump(mode="json"),
            is_async=is_async,
        )


class Substep(str, Enum):
    # Included in legacy substep times
    EVAL_BEFORE = SlimeStatus.EVAL_ROLLOUT_LOGGING.value
    GENERATE_ROLLOUTS = SlimeStatus.ROLLOUT_LOGGING.value
    OFFLOAD_ROLLOUT = SlimeStatus.OFFLOAD_ROLLOUT.value
    COMPUTE_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
    OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
    CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
    OFFLOAD_TRAIN = SlimeStatus.OFFLOAD_TRAIN.value
    WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value
    EVAL_AFTER = f"{SlimeStatus.EVAL_ROLLOUT_LOGGING.value}_end"

    WAIT_FOR_ROLLOUT = "wait_for_rollout"  # driver
    TRAIN_MODELS = "train_models"  # driver
    CUSTOM_REWARD = "custom_reward"  # rollout worker
    REWARD_POST_PROCESS = "reward_post_process"  # rollout worker
    FORWARD_BACKWARD = "forward_backward"  # actor / critic


def phase_totals(intervals: list[list[float]]) -> dict[str, float | int]:
    """Deliver the numbers the UI prints for one phase for one role, from its ``[start, end]`` list.

    An interval with a duration of 0.0 is still displayed to show the phase ran.
    """
    if not intervals:
        return {
            "total_duration_s": 0.0,
            "wall_span_s": 0.0,
            "count": 0,
            "start_offset_s": 0.0,
        }
    return {
        "total_duration_s": round(sum(end - start for start, end in intervals), 6),
        "wall_span_s": round(max(end for _, end in intervals) - intervals[0][0], 6),
        "count": len(intervals),
        "start_offset_s": round(intervals[0][0], 6),
    }


def rollout_lanes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns which roles need to be drawn for UI for a run.

    Keyed by role so the frontend indexes lanes by name rather than by position.
    """
    lanes: dict[str, Any] = {}
    for record in records:
        phases = record["phases"]
        lanes[record["role"]] = {
            "role": record["role"],
            "lane_start_unix_s": record["lane_start_unix_s"],
            "phases": phases,
            "totals": {
                name: phase_totals(intervals) for name, intervals in phases.items()
            },
        }
    return {"roles": lanes}


def load_steps(
    training_run_id: str, rollout_ids: list[int]
) -> dict[int, list[dict[str, Any]]]:
    """The <=4 role records of each requested rollout, keyed by rollout id.

    Called by ``_dashboard._timings_for`` in a threadpool. One volume listing
    covers the whole batch — a listing per rollout runs into the volume's list
    rate limit as soon as a page asks for a handful of rows. A rollout with no
    records is left out, which is a normal outcome rather than an error: it can
    be mid-flight, or the run can predate measured timing, in which case the
    caller reads the run's legacy blob instead.
    """
    steps: dict[int, list[dict[str, Any]]] = {}
    for record in vol_list_prefix(
        RoleTimingRecord.store(training_run_id),
        tuple(RoleTimingRecord.step_prefix(i) for i in rollout_ids),
    ):
        steps.setdefault(record["rollout_id"], []).append(record)
    return steps


# ---------- Legacy handling --------------


class DashboardTooOldError(RuntimeError):
    """The dashboard cannot accept measured timing and the recipe requires it."""


def probe_substep_timing(framework_status_url: str, mode: str = "auto") -> bool:
    """Check the dashboard can accept timing records; return whether to enable.

    Called on the host before the training app is spawned, so ``"require"``
    fails while nothing is allocated rather than 40 GPUs in.

    ``"auto"`` warns and disables rather than raising: refusing to start a
    multi-node job over missing telemetry costs more than the telemetry.
    Disabling means the run produces no timing records -- it never falls back
    to the legacy path, which no longer has a writer.
    """
    import json
    import urllib.error
    import urllib.request

    if mode == "off":
        return False
    if not framework_status_url:
        if mode == "require":
            raise DashboardTooOldError(
                "substep_timing='require' but this run has no dashboard URL to "
                "report timing to. Deploy the dashboard "
                "(`modal deploy dashboards/app.py`) or set substep_timing='auto'."
            )
        return False

    base = framework_status_url.rstrip("/")
    suffix = "/api/framework-status"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    url = f"{base}/api/timing-events"

    problem = ""
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            payload = json.loads(response.read() or b"{}")
        if payload.get("protocol") != PROTOCOL:
            problem = f"{url} did not identify as {PROTOCOL!r}"
    except urllib.error.HTTPError as exc:
        # 404 means an older dashboard without the route; anything else is a
        # live dashboard failing, for which "redeploy" is the wrong advice.
        problem = (
            f"{url} returned 404 -- the deployed dashboard predates substep timing"
            if exc.code == 404
            else f"{url} returned HTTP {exc.code}"
        )
    except (OSError, ValueError) as exc:
        problem = f"could not reach {url}: {exc}"

    if not problem:
        return True
    if mode == "require":
        raise DashboardTooOldError(
            f"substep_timing='require' but {problem}. Redeploy the dashboard "
            "(`modal deploy dashboards/app.py`) or set substep_timing='auto'."
        )
    print(f"WARNING: substep timing disabled for this run -- {problem}")
    return False


# Old step timing method grouped general training step i.e. forward/backward
# under "optimizer step". New view more accurately reflects this.
_LEGACY_RENAMES = {Substep.OPTIMIZER_STEP.value: Substep.TRAIN_MODELS.value}


def legacy_run_to_records(
    substep_times: dict[str, dict[str, dict[str, float | None]]] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step_key, subs in (substep_times or {}).items():
        if not step_key.isdigit():
            continue
        starts = [sub["start"] for sub in subs.values() if sub.get("start") is not None]
        if not starts:
            continue
        lane_start = min(starts)
        phases: dict[str, list[list[float]]] = {}
        for name, sub in subs.items():
            start, duration = sub.get("start"), sub.get("duration_s")
            if start is None or duration is None:
                continue
            rel = start - lane_start
            phases[_LEGACY_RENAMES.get(name, name)] = [
                [round(rel, 6), round(rel + duration, 6)]
            ]
        records.append(
            {
                "rollout_id": int(step_key) - 1,
                "role": Role.DRIVER.value,
                "lane_start_unix_s": lane_start,
                "phases": phases,
            }
        )
    return records
