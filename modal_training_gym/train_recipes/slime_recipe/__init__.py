from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.glm_4_7 import GLM_4_7_Recipe
from modal_training_gym.train_recipes.slime_recipe.megagem_qwen3_4b_stage_c import (
    MegaGem_Qwen3_4B_StageC_Recipe,
    MegaGemStageCDataset,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_0_6b import Qwen3_0_6b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_1_7b import Qwen3_1_7b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_8b import Qwen3_8b_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4b_Recipe

from modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b import (
    Qwen3_6_35b_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_asr_1_7b import (
    Qwen3_ASR_1_7b_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_vl_8b import (
    Qwen3_VL_8b_Recipe,
)

__all__ = [
    "SlimeRecipe",
    "GLM_4_7_Recipe",
    "MegaGem_Qwen3_4B_StageC_Recipe",
    "MegaGemStageCDataset",
    "Qwen3_0_6b_Recipe",
    "Qwen3_1_7b_Recipe",
    "Qwen3_4b_Recipe",
    "Qwen3_8b_Recipe",
    "Qwen3_6_35b_Recipe",
    "Qwen3_ASR_1_7b_Recipe",
    "Qwen3_VL_8b_Recipe",
]
