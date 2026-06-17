"""Live, sandbox-backed RL environments.

``base`` holds the framework-agnostic abstractions (I/O shapes + lifecycle base
classes); concrete benchmarks (e.g. ``toolathlon``) build on them.
"""

from modal_training_gym.common.environments.base import (
    DirectorySnapshotLibrary,
    Environment,
    EvalVerdict,
    Observation,
    SandboxEnvironment,
    SandboxEnvironmentPool,
    StepResult,
    ToolCall,
)
from modal_training_gym.common.environments.toolathlon import (
    DEFAULT_CONFIG,
    TIER_A_MCPS,
    ToolathlonEnvConfig,
    ToolathlonEnvironment,
    ToolathlonEnvPool,
    ToolathlonTrajectoryDataset,
    build_env_image,
    build_prefix_messages,
    build_snapshot_library,
    default_system_prompt,
    dispatch_tool,
    get_env_pool,
    prune_prefix,
    render_tool_catalog,
    tool_schemas_to_openai,
)

__all__ = [
    # base — shapes
    "ToolCall",
    "Observation",
    "StepResult",
    "EvalVerdict",
    # base — lifecycle
    "Environment",
    "SandboxEnvironment",
    "SandboxEnvironmentPool",
    "DirectorySnapshotLibrary",
    # toolathlon — config + env
    "ToolathlonEnvConfig",
    "DEFAULT_CONFIG",
    "TIER_A_MCPS",
    "ToolathlonEnvironment",
    "ToolathlonEnvPool",
    "get_env_pool",
    "build_env_image",
    "dispatch_tool",
    # toolathlon — snapshots
    "build_snapshot_library",
    # toolathlon — data + prompts
    "ToolathlonTrajectoryDataset",
    "build_prefix_messages",
    "prune_prefix",
    "render_tool_catalog",
    "default_system_prompt",
    "tool_schemas_to_openai",
]
