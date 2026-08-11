from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe
from modal_training_gym.train_recipes.miles_recipe.kimi import (
    Kimi_K2_5_LoRA_Recipe,
    Kimi_K2_6_LoRA_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
    Qwen3_5_4b_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.moonlight_16b_a3b import (
    Moonlight_16B_A3B_Recipe,
)

__all__ = [
    "MilesRecipe",
    "Kimi_K2_5_LoRA_Recipe",
    "Kimi_K2_6_LoRA_Recipe",
    "Moonlight_16B_A3B_Recipe",
    "Qwen3_5_4b_Recipe",
]
