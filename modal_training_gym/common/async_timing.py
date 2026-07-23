from typing import Literal, TypeAlias, TypedDict

from pydantic import BaseModel, Field

from modal_training_gym.common.step_timing import TimingLane


AsyncTimingEventType: TypeAlias = Literal[
    "rollout_start",
    "rollout_finish",
    "phase_start",
    "phase_finish",
]


class AsyncTimingEvent(TypedDict):
    training_run_id: str
    training_attempt: int
    rollout_id: int
    phase: str
    event_type: AsyncTimingEventType
    timestamp: float
    monotonic_timestamp: float
    occurrence_id: int | None
    role: str | None
    rank: int | None
    world_size: int | None
    timeline_lane: TimingLane | None
    parent_phase: str | None
    display_name: str | None


class AsyncStepTimingNotification(TypedDict):
    training_run_id: str
    training_attempt: int
    completed_rollout_id: int


class AsyncStepTimingUpdate(BaseModel):
    training_run_id: str = Field(min_length=1)
    training_attempt: int = Field(ge=1)
    completed_rollout_id: int = Field(ge=0)
