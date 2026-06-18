"""Qwen3-VL-8B gym wiring for the computer-use example: the recipe freezes the
vision tower and uses padded (bshd) batches + bridge export, the model declares
no framework-specific patch metadata, and Qwen3-VL dispatches to its recipe.
"""

from modal_training_gym import Qwen3VL_8B, Qwen3VL_Recipe
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


def _flags(args: list[str]) -> dict[str, str]:
    return {
        args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")
    }


def test_recipe_freezes_vision_tower_and_uses_bshd_bridge():
    recipe = Qwen3VL_Recipe()
    # RL only updates the language backbone; the ViT is frozen.
    assert recipe.freeze_params_name_list == ["vision_model"]
    # VL forward needs padded batches (no THD packing).
    assert recipe.use_dynamic_batch_size is False
    assert recipe.extra_config["qkv_format"] == "bshd"
    # Export goes through AutoBridge (no torch_dist pre-conversion).
    assert recipe.megatron_to_hf_mode == "bridge"

    flags = _flags(recipe.cli_args(model=Qwen3VL_8B()))
    assert flags["--freeze-params-name-list"] == "vision_model"
    assert flags["--tensor-model-parallel-size"] == "2"
    assert flags["--megatron-to-hf-mode"] == "bridge"


def test_model_carries_no_framework_patch_metadata():
    # Patch wiring is slime-specific and lives in the slime layer, not the model
    # spec (which must stay framework-agnostic).
    arch = Qwen3VL_8B().architecture
    assert not hasattr(arch, "compat_patches")


def test_model_requires_bshd():
    assert Qwen3VL_8B.requires_bshd is True


def test_qwen3_vl_dispatches_to_its_recipe():
    base = SlimeRecipe.get_base_recipe(Qwen3VL_8B())
    assert isinstance(base, Qwen3VL_Recipe)
