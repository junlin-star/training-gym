from modal_training_gym.train_recipes.slime_recipe.blocks import (
    MultiTurn,
    SlimeRecipeBlock,
)
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.glm_4_7 import GLM_4_7_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_1_7b import Qwen3_1_7b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_8b import Qwen3_8b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_14b import Qwen3_14b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_32b import Qwen3_32b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b import (
    Qwen3_6_35b_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_asr import Qwen3ASR_Recipe

__all__ = [
    "MultiTurn",
    "SlimeRecipe",
    "SlimeRecipeBlock",
    "GLM_4_7_Recipe",
    "Qwen3_1_7b_Recipe",
    "Qwen3_4b_Recipe",
    "Qwen3_8b_Recipe",
    "Qwen3_14b_Recipe",
    "Qwen3_32b_Recipe",
    "Qwen3_6_35b_Recipe",
    "Qwen3ASR_Recipe",
]
