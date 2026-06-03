from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DatasetConfig": ("modal_training_gym.common.dataset", "DatasetConfig"),
    "HarborDataset": ("modal_training_gym.common.dataset", "HarborDataset"),
    "DeploymentConfig": ("modal_training_gym.common.deployment", "DeploymentConfig"),
    "EvalConfig": ("modal_training_gym.common.eval", "EvalConfig"),
    "EvalConfigDurable": ("modal_training_gym.common.eval", "EvalConfigDurable"),
    "EvalResult": ("modal_training_gym.common.eval", "EvalResult"),
    "EvalRowResult": ("modal_training_gym.common.eval", "EvalRowResult"),
    "extract_code": ("modal_training_gym.common.eval", "extract_code"),
    "HarborEval": ("modal_training_gym.common.eval", "HarborEval"),
    "HFModelConfiguration": (
        "modal_training_gym.common.models",
        "HFModelConfiguration",
    ),
    "HuggingFaceDataset": ("modal_training_gym.common.dataset", "HuggingFaceDataset"),
    "list_checkpoints": ("modal_training_gym.common.checkpoint", "list_checkpoints"),
    "METADATA_VOLUME_NAME": (
        "modal_training_gym.utils.metadata",
        "METADATA_VOLUME_NAME",
    ),
    "MetadataStore": ("modal_training_gym.utils.metadata", "MetadataStore"),
    "ModelArchitecture": ("modal_training_gym.common.models", "ModelArchitecture"),
    "ModelConfig": ("modal_training_gym.common.models", "ModelConfig"),
    "ModelDeployment": ("modal_training_gym.common.deployment", "ModelDeployment"),
    "Kimi_K2_5": ("modal_training_gym.common.models", "Kimi_K2_5"),
    "Kimi_K2_6": ("modal_training_gym.common.models", "Kimi_K2_6"),
    "Kimi_K2_5_Recipe": (
        "modal_training_gym.train_recipes.miles_recipe",
        "Kimi_K2_5_Recipe",
    ),
    "Kimi_K2_6_Recipe": (
        "modal_training_gym.train_recipes.miles_recipe",
        "Kimi_K2_6_Recipe",
    ),
    "MilesConfig": ("modal_training_gym.train_recipes.miles_recipe", "MilesConfig"),
    "MultiTurn": ("modal_training_gym.train_recipes.slime_recipe", "MultiTurn"),
    "parse_qwen3_response": (
        "modal_training_gym.common.models",
        "parse_qwen3_response",
    ),
    "ParsedResponse": ("modal_training_gym.common.models", "ParsedResponse"),
    "Qwen3_0_6B": ("modal_training_gym.common.models", "Qwen3_0_6B"),
    "Qwen3_1_7B": ("modal_training_gym.common.models", "Qwen3_1_7B"),
    "Qwen3_4B": ("modal_training_gym.common.models", "Qwen3_4B"),
    "Qwen3_8B": ("modal_training_gym.common.models", "Qwen3_8B"),
    "Qwen3_14B": ("modal_training_gym.common.models", "Qwen3_14B"),
    "Qwen3_30B": ("modal_training_gym.common.models", "Qwen3_30B"),
    "Qwen3_32B": ("modal_training_gym.common.models", "Qwen3_32B"),
    "Qwen3_6_35B": ("modal_training_gym.common.models", "Qwen3_6_35B"),
    "score_in_sandbox": ("modal_training_gym.common.eval", "score_in_sandbox"),
    "SlimeRecipe": ("modal_training_gym.train_recipes.slime_recipe", "SlimeRecipe"),
    "SlimeRecipeBlock": (
        "modal_training_gym.train_recipes.slime_recipe",
        "SlimeRecipeBlock",
    ),
    "ToolCall": ("modal_training_gym.common.models", "ToolCall"),
    "TrainConfig": ("modal_training_gym.common.train", "TrainConfig"),
    "TrainResult": ("modal_training_gym.common.train_result", "TrainResult"),
    "setup": ("modal_training_gym.setup", "setup"),
    "WandbConfig": ("modal_training_gym.common.wandb", "WandbConfig"),
}

__all__ = [
    "DatasetConfig",
    "HarborDataset",
    "DeploymentConfig",
    "EvalConfig",
    "EvalConfigDurable",
    "EvalResult",
    "EvalRowResult",
    "extract_code",
    "HarborEval",
    "HFModelConfiguration",
    "HuggingFaceDataset",
    "list_checkpoints",
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "ModelArchitecture",
    "ModelConfig",
    "ModelDeployment",
    "MilesConfig",
    "MultiTurn",
    "parse_qwen3_response",
    "ParsedResponse",
    "Qwen3_0_6B",
    "Qwen3_1_7B",
    "Qwen3_4B",
    "Qwen3_8B",
    "Qwen3_14B",
    "Qwen3_30B",
    "Qwen3_32B",
    "Qwen3_6_35B",
    "score_in_sandbox",
    "SlimeRecipe",
    "setup",
    "SlimeRecipeBlock",
    "ToolCall",
    "TrainConfig",
    "TrainResult",
    "WandbConfig",
]


def __getattr__(name: str):
    module_name, attr_name = _EXPORTS.get(name, (None, None))
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
