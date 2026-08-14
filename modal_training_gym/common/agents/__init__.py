"""Framework-neutral agent protocol helpers."""

from modal_training_gym.common.agents.mini_swe import (
    BASH_TOOL,
    FORMAT_ERROR_TEMPLATE,
    INSTANCE_TEMPLATE,
    OBSERVATION_TEMPLATE,
    SUBMIT_SENTINEL,
    SYSTEM_TEMPLATE,
    MiniSweEnvironmentAdapter,
)

__all__ = [
    "BASH_TOOL",
    "FORMAT_ERROR_TEMPLATE",
    "INSTANCE_TEMPLATE",
    "OBSERVATION_TEMPLATE",
    "SUBMIT_SENTINEL",
    "SYSTEM_TEMPLATE",
    "MiniSweEnvironmentAdapter",
]
