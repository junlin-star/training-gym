"""MilesRecipe hook stashing: the four intercepted log/megatron hooks ride
inside ``extra_config`` under ``training_gym_*`` keys, which is impossible when
``extra_config`` is a YAML file path rather than a dict.
"""

from types import SimpleNamespace

import pytest

from modal_training_gym.frameworks.miles.phase_reporting import _hook_path_from_args
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe


def test_str_hook_is_stashed_in_extra_config() -> None:
    recipe = MilesRecipe(custom_rollout_log_function="my_pkg.hooks.log_fn")
    assert isinstance(recipe.extra_config, dict)
    assert (
        recipe.extra_config["training_gym_custom_rollout_log_function_path"]
        == "my_pkg.hooks.log_fn"
    )


def test_yaml_path_extra_config_without_hooks_is_allowed() -> None:
    recipe = MilesRecipe(extra_config="configs/extra.yaml")
    assert recipe.extra_config == "configs/extra.yaml"


def test_yaml_path_extra_config_with_hook_raises() -> None:
    with pytest.raises(ValueError, match="custom_rollout_log_function"):
        MilesRecipe(
            extra_config="configs/extra.yaml",
            custom_rollout_log_function="my_pkg.hooks.log_fn",
        )


_KEY = "training_gym_custom_rollout_log_function_path"


def test_hook_lookup_prefers_training_gym_key() -> None:
    args = SimpleNamespace(
        extra_config={
            _KEY: "my_pkg.hooks.log_fn",
            "custom_rollout_log_function_path": "other.fn",
        }
    )
    assert _hook_path_from_args(args, _KEY) == "my_pkg.hooks.log_fn"


def test_hook_lookup_falls_back_to_native_key() -> None:
    args = SimpleNamespace(
        extra_config={"custom_rollout_log_function_path": "my_pkg.hooks.log_fn"}
    )
    assert _hook_path_from_args(args, _KEY) == "my_pkg.hooks.log_fn"


def test_hook_lookup_never_dispatches_to_gym_wrapper() -> None:
    args = SimpleNamespace(
        extra_config={
            "custom_rollout_log_function_path": (
                "modal_training_gym.frameworks.miles.phase_reporting.log_rollout_data"
            )
        }
    )
    assert _hook_path_from_args(args, _KEY) is None
