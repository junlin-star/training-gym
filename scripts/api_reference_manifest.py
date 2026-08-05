"""
Curated manifest of public classes for API reference documentation.

Each entry maps a class to its module path, documentation group,
and class type (config_data or behavior).
"""

API_REFERENCE_MANIFEST = [
    # --- Core ---
    {
        "class_name": "ModelConfig",
        "module": "modal_training_gym.common.models.base",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "ModelConfig",
    },
    {
        "class_name": "HFModelConfiguration",
        "module": "modal_training_gym.common.models.base",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "HFModelConfiguration",
    },
    {
        "class_name": "ModelArchitecture",
        "module": "modal_training_gym.common.models.base",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "ModelArchitecture",
    },
    {
        "class_name": "DatasetConfig",
        "module": "modal_training_gym.common.dataset",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "DatasetConfig",
    },
    {
        "class_name": "HuggingFaceDataset",
        "module": "modal_training_gym.common.dataset",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "HuggingFaceDataset",
    },
    {
        "class_name": "HarborDataset",
        "module": "modal_training_gym.common.dataset",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "HarborDataset",
    },
    {
        "class_name": "WandbConfig",
        "module": "modal_training_gym.common.wandb",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "WandbConfig",
    },
    {
        "class_name": "ModalRayCluster",
        "module": "modal_training_gym.common.ray_cluster",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "ModalRayCluster",
    },
    {
        "class_name": "TrainResult",
        "module": "modal_training_gym.common.train_result",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "TrainResult",
    },
    {
        "class_name": "Sample",
        "module": "modal_training_gym.common.sample",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "Sample",
    },
    {
        "class_name": "MultimodalDataset",
        "module": "modal_training_gym.common.dataset",
        "group": "core",
        "class_type": "config_data",
        "sidebar_label": "MultimodalDataset",
    },
    {
        "class_name": "extract_code",
        "module": "modal_training_gym.common.sandbox_scoring",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "extract_code",
    },
    {
        "class_name": "score_in_sandbox",
        "module": "modal_training_gym.common.sandbox_scoring",
        "group": "core",
        "class_type": "behavior",
        "sidebar_label": "score_in_sandbox",
    },
    # --- Models ---
    {
        "class_name": "ToolCall",
        "module": "modal_training_gym.common.models.base",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "ToolCall",
    },
    {
        "class_name": "ParsedResponse",
        "module": "modal_training_gym.common.models.base",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "ParsedResponse",
    },
    {
        "class_name": "parse_qwen3_response",
        "module": "modal_training_gym.common.models.base",
        "group": "models",
        "class_type": "behavior",
        "sidebar_label": "parse_qwen3_response",
    },
    {
        "class_name": "Qwen3_0_6B",
        "module": "modal_training_gym.common.models.qwen3_0_6b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-0.6B",
    },
    {
        "class_name": "Qwen3_1_7B",
        "module": "modal_training_gym.common.models.qwen3_1_7b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-1.7B",
    },
    {
        "class_name": "Qwen3_4B",
        "module": "modal_training_gym.common.models.qwen3_4b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-4B",
    },
    {
        "class_name": "Qwen3_8B",
        "module": "modal_training_gym.common.models.qwen3_8b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-8B",
    },
    {
        "class_name": "Qwen3_30B",
        "module": "modal_training_gym.common.models.qwen3_30b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-30B-A3B",
    },
    {
        "class_name": "Qwen3_6_35B",
        "module": "modal_training_gym.common.models.qwen3_6_35b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3.6-35B-A3B",
    },
    {
        "class_name": "Qwen3_6_27B",
        "module": "modal_training_gym.common.models.qwen3_6_27b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3.6-27B",
    },
    {
        "class_name": "Qwen3_VL_8B",
        "module": "modal_training_gym.common.models.qwen3_vl_8b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-VL-8B",
    },
    {
        "class_name": "Qwen3_ASR_1_7B",
        "module": "modal_training_gym.common.models.qwen3_asr_1_7b",
        "group": "models",
        "class_type": "config_data",
        "sidebar_label": "Qwen3-ASR-1.7B",
    },
    # --- Training ---
    {
        "class_name": "TrainConfig",
        "module": "modal_training_gym.common.train",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "TrainConfig",
    },
    {
        "class_name": "TrainingGroup",
        "module": "modal_training_gym.common.training_group",
        "group": "training",
        "class_type": "behavior",
        "sidebar_label": "TrainingGroup",
    },
    {
        "class_name": "SlimeRecipe",
        "module": "modal_training_gym.train_recipes.slime_recipe.recipe",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "SlimeRecipe",
    },
    {
        "class_name": "MilesRecipe",
        "module": "modal_training_gym.train_recipes.miles_recipe.recipe",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "MilesRecipe",
    },
    {
        "class_name": "Qwen3_6_35b_Recipe",
        "module": "modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "Qwen3_6_35b_Recipe",
    },
    {
        "class_name": "Qwen3_6_27b_Recipe",
        "module": "modal_training_gym.train_recipes.slime_recipe.qwen3_6_27b",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "Qwen3_6_27b_Recipe",
    },
    {
        "class_name": "Qwen3_VL_8b_Recipe",
        "module": "modal_training_gym.train_recipes.slime_recipe.qwen3_vl_8b",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "Qwen3_VL_8b_Recipe",
    },
    {
        "class_name": "Qwen3_ASR_1_7b_Recipe",
        "module": "modal_training_gym.train_recipes.slime_recipe.qwen3_asr_1_7b",
        "group": "training",
        "class_type": "config_data",
        "sidebar_label": "Qwen3_ASR_1_7b_Recipe",
    },
    {
        "class_name": "list_checkpoints",
        "module": "modal_training_gym.common.checkpoint",
        "group": "training",
        "class_type": "behavior",
        "sidebar_label": "list_checkpoints",
    },
    {
        "class_name": "convert_checkpoint_to_hf",
        "module": "modal_training_gym.common.checkpoint",
        "group": "training",
        "class_type": "behavior",
        "sidebar_label": "convert_checkpoint_to_hf",
    },
    {
        "class_name": "ensure_endpoint",
        "module": "modal_training_gym.common.endpoint",
        "group": "endpoints",
        "class_type": "behavior",
        "sidebar_label": "ensure_endpoint",
    },
    {
        "class_name": "endpoint_chat",
        "module": "modal_training_gym.common.endpoint",
        "group": "endpoints",
        "class_type": "behavior",
        "sidebar_label": "endpoint_chat",
    },
    {
        "class_name": "endpoint_chat_message",
        "module": "modal_training_gym.common.endpoint",
        "group": "endpoints",
        "class_type": "behavior",
        "sidebar_label": "endpoint_chat_message",
    },
    {
        "class_name": "wait_for_server_url",
        "module": "modal_training_gym.common.endpoint",
        "group": "endpoints",
        "class_type": "behavior",
        "sidebar_label": "wait_for_server_url",
    },
]

GROUPS = {
    "core": {"label": "Core", "order": 1},
    "models": {"label": "Models", "order": 2},
    "training": {"label": "Training", "order": 3},
    "endpoints": {"label": "Endpoints", "order": 4},
}


def class_to_reference_path(class_name: str) -> str | None:
    """Return the Starlight reference path for a class, or None if not in manifest."""
    for entry in API_REFERENCE_MANIFEST:
        if entry["class_name"] == class_name:
            slug = class_name.lower()
            return f"/reference/{entry['group']}/{slug}/"
    return None


CLASS_REFERENCE_PATHS = {
    entry["class_name"]: f"/reference/{entry['group']}/{entry['class_name'].lower()}/"
    for entry in API_REFERENCE_MANIFEST
}
