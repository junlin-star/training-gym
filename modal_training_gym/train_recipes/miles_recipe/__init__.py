from modal_training_gym.train_recipes.miles_recipe.recipe import MilesConfig
from modal_training_gym.train_recipes.miles_recipe.kimi import (
    Kimi_K2_5_LoRA_Recipe,
    Kimi_K2_6_LoRA_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.qwen3_6 import (
    Qwen3_6_35B_A3B_LoRA_Recipe,
)

__all__ = [
    "MilesConfig",
    "Kimi_K2_5_LoRA_Recipe",
    "Kimi_K2_6_LoRA_Recipe",
    "Qwen3_6_35B_A3B_LoRA_Recipe",
]
