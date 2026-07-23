from enum import Enum
from typing import Literal, NotRequired, TypeAlias, TypedDict

from modal_training_gym.common.status import SlimeStatus

TimingLane: TypeAlias = Literal[
    "rollout",
    "reward",
    "training",
    "coordination",
]


class StepTiming(TypedDict):
    start: int | None
    end: int | None
    duration_s: int | None


class SubstepTimingInterval(TypedDict):
    start: float
    duration_s: float
    step_id: NotRequired[int]
    training_role: NotRequired[str]
    training_rank: NotRequired[int]
    slowest_rank: NotRequired[int]
    training_world_size: NotRequired[int]
    reported_rank_count: NotRequired[int]
    active_duration_s: NotRequired[float]
    timeline_lane: NotRequired[TimingLane]
    parent_phase: NotRequired[str]
    display_name: NotRequired[str]


class SubstepTiming(TypedDict):
    start: float
    duration_s: float | None
    intervals: NotRequired[list[SubstepTimingInterval]]


StepTimes: TypeAlias = dict[str, StepTiming]
SubstepTimes: TypeAlias = dict[str, dict[str, SubstepTiming]]


class Substep(str, Enum):
    EVAL_BEFORE = SlimeStatus.EVAL_ROLLOUT_LOGGING.value
    GENERATE_ROLLOUTS = SlimeStatus.ROLLOUT_LOGGING.value
    OFFLOAD_ROLLOUT = SlimeStatus.OFFLOAD_ROLLOUT.value
    COMPUTE_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
    OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
    CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
    OFFLOAD_TRAIN = SlimeStatus.OFFLOAD_TRAIN.value
    WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value
    EVAL_AFTER = f"{SlimeStatus.EVAL_ROLLOUT_LOGGING.value}_end"


class TrainingSubstep(str, Enum):
    POLICY_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
    REFERENCE_LOG_PROBS = "reference_log_probs"
    TEACHER_LOG_PROBS = "teacher_log_probs"
    FORWARD_BACKWARD = "forward_backward"
    OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
