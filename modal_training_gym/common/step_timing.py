from modal_training_gym.common.sync_timing_recording import record_sync_timing_event
from modal_training_gym.common.timing_types import Substep

record_step_time_event = record_sync_timing_event

__all__ = ["Substep", "record_step_time_event"]
