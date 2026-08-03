from __future__ import annotations

import pytest

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import Gemma4_26B_A4B
from modal_training_gym.common.train import _merge_recipe
from modal_training_gym.train_recipes.slime_recipe import (
    Gemma4_26B_A4B_Recipe,
    SlimeRecipe,
)


def test_text_mode_trains_through_the_slime_model_script() -> None:
    model = Gemma4_26B_A4B()
    recipe = SlimeRecipe.get_base_recipe(model)

    assert model.architecture.megatron_spec is not None
    assert recipe.slime_model_script == "scripts/models/gemma4-26B-A4B.sh"
    assert recipe.megatron_to_hf_mode == "raw"
    assert recipe.use_dynamic_batch_size is True
    assert recipe.attention_backend == "flash"
    assert recipe.pipeline_model_parallel_size == 2


def test_vision_mode_trains_through_the_bridge() -> None:
    model = Gemma4_26B_A4B(vision=True)
    recipe = SlimeRecipe.get_base_recipe(model)

    assert model.requires_bshd is True
    assert model.architecture.megatron_spec is None
    # A model script would build the text-only GPTModel and ignore every image.
    assert recipe.slime_model_script == ""
    assert recipe.megatron_to_hf_mode == "bridge"
    assert recipe.use_dynamic_batch_size is False
    assert recipe.extra_config == {"qkv_format": "bshd", "micro_batch_size": 1}
    assert recipe.attention_backend is None
    assert recipe.pipeline_model_parallel_size == 1
    assert len(recipe.image_run_commands) == 4


@pytest.mark.parametrize("vision", [False, True])
def test_explicit_overrides_survive_the_model_recipe_merge(vision: bool) -> None:
    base = Gemma4_26B_A4B_Recipe(vision=vision)
    overrides = Gemma4_26B_A4B_Recipe(
        vision=vision, num_rollout=3, global_batch_size=32
    )

    # _merge_recipe reconstructs the recipe, so __post_init__ runs again and must not
    # overwrite what the caller asked for.
    merged = _merge_recipe(base, overrides)

    assert merged.num_rollout == 3
    assert merged.global_batch_size == 32
    assert merged.vision is vision


def test_a_model_recipe_mode_mismatch_is_rejected() -> None:
    recipe = Gemma4_26B_A4B_Recipe(vision=False)

    with pytest.raises(TrainingGymConfigError, match="vision"):
        recipe.validate_model_parallelism(Gemma4_26B_A4B(vision=True))


def test_the_vision_flag_is_not_a_slime_cli_flag() -> None:
    model = Gemma4_26B_A4B(vision=True)
    recipe = SlimeRecipe.get_base_recipe(model)

    assert "vision" not in recipe._fields(model=model)
    assert "--vision" not in recipe.cli_args(model=model)
