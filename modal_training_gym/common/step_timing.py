from __future__ import annotations

from ast import Continue
from collections.abc import MutableMapping
from enum import Enum
from typing import Any

from modal_training_gym.common.status import SlimeStatus

class Role(str, Enum):
    DRIVER = "driver"
    ROLLOUT = "rollout"
    ACTOR = "actor"
    CRITIC = "critic"

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

    WAIT_FOR_ROLLOUT = "wait_for_rollout"        # driver
    TRAIN_MODELS = "train_models"                # driver
    CUSTOM_REWARD = "custom_reward"              # rollout worker
    REWARD_POST_PROCESS = "reward_post_process"  # rollout worker
    FORWARD_BACKWARD = "forward_backward"        # actor / critic

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

class RoleTimingRecord(BaseModel):
    training_run_id: str
    rollout_id: int = Field(ge=0)
    role: Role
    created_at: int = 0
    lane_start_unix_s: float | None = None
    phases: dict[str, list[tuple[float, float]]] = Field(default_factory=dict)

    @property
    def storage_key(self) -> str:
        return f"{self.rollout_id:08d}__{self.role.value}"

    @staticmethod
    def store(training_run_id: str) -> str:
        return f"{MetadataStore.SUBSTEP_TIMING.value}/{training_run_id}"

    @staticmethod
    def step_prefix(rollout_id: int) -> str:
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


def load_step(training_run_id: str, rollout_id: int) -> list[dict[str, Any]]:
    """The <=4 role records for one rollout, as stored. ``[]`` when none.

    Called by ``_dashboard._timings_for`` in a threadpool, once per uncached
    rollout. An empty result is a normal outcome, not an error: every
    pre-cutover run is in that state, and the caller falls through to the
    legacy adapter.
    """
    return vol_list_prefix(
        RoleTimingRecord.store(training_run_id),
        RoleTimingRecord.step_prefix(rollout_id),
    )



# Old step timing method grouped general training step i.e. forward/backward under "optimizer step". New view more accurately reflects this 
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
