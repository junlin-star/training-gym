"""Live RL environments.

``base`` holds the framework-agnostic abstractions (I/O shapes + lifecycle base classes); concrete
benchmarks build on them — ``toolathlon`` (Modal-sandbox-backed) and ``bfcl`` (in-process, no
sandbox needed at all; see :mod:`.bfcl`'s module docstring for why).
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
from modal_training_gym.common.environments.bfcl import (
    DEFAULT_CONFIG as BFCL_DEFAULT_CONFIG,
)
from modal_training_gym.common.environments.bfcl import (
    BfclMultiTurnConfig,
    BfclMultiTurnDataset,
    BfclTurnEnvironment,
)
from modal_training_gym.common.environments.bfcl import build_env as build_bfcl_env
from modal_training_gym.common.environments.bfcl import (
    build_instances as build_bfcl_instances,
)
from modal_training_gym.common.environments.bfcl import (
    build_prefix_messages as build_bfcl_prefix_messages,
)
from modal_training_gym.common.environments.bfcl import (
    default_system_prompt as bfcl_default_system_prompt,
)
from modal_training_gym.common.environments.bfcl import (
    execute_call as execute_bfcl_call,
)
from modal_training_gym.common.environments.bfcl import (
    load_func_docs as load_bfcl_func_docs,
)
from modal_training_gym.common.environments.bfcl import (
    parse_call_string as parse_bfcl_call_string,
)
from modal_training_gym.common.environments.bfcl import (
    prune_prefix as prune_bfcl_prefix,
)
from modal_training_gym.common.environments.bfcl import (
    render_tool_catalog as render_bfcl_tool_catalog,
)
from modal_training_gym.common.environments.bfcl import replay as replay_bfcl
from modal_training_gym.common.environments.bfcl import to_json_schema
from modal_training_gym.common.environments.bfcl import (
    tool_schemas_to_openai as bfcl_tool_schemas_to_openai,
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
    # bfcl — config + env
    "BfclMultiTurnConfig",
    "BFCL_DEFAULT_CONFIG",
    "BfclTurnEnvironment",
    "build_bfcl_env",
    "build_bfcl_instances",
    "execute_bfcl_call",
    "replay_bfcl",
    "parse_bfcl_call_string",
    # bfcl — data + prompts
    "BfclMultiTurnDataset",
    "build_bfcl_prefix_messages",
    "prune_bfcl_prefix",
    "render_bfcl_tool_catalog",
    "bfcl_default_system_prompt",
    "bfcl_tool_schemas_to_openai",
    "load_bfcl_func_docs",
    "to_json_schema",
]
