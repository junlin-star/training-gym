"""Caller-set field tracking and the vision-mode resolution built on it.

``_for_dataset`` overrides only the fields the caller left alone, so losing that
record turns vision mode into a blanket overwrite. The record depends on pydantic
handing the wrap validator back the same instance it initialised, which these
tests pin.
"""

import dataclasses as dc

import pydantic.dataclasses as pdc
import pytest
from pydantic import ConfigDict

from modal_training_gym.common.dataset import MultimodalDataset
from modal_training_gym.train_recipes.base import carry_explicit_fields
from modal_training_gym.train_recipes.miles_recipe import (
    Gemma4_26B_A4B_Recipe,
    MilesRecipe,
)


@pytest.fixture
def image_dataset():
    return MultimodalDataset(
        modality="image",
        rows=[{"prompt": "p", "media": ["data:image/png;base64,x"], "label": "1"}],
    )


def test_constructor_kwargs_are_recorded():
    recipe = Gemma4_26B_A4B_Recipe(num_rollout=5)
    assert "num_rollout" in recipe.explicit_fields
    assert "rollout_temperature" not in recipe.explicit_fields


def test_caller_value_survives_vision_mode(image_dataset):
    """An explicit value must win over the vision default, even when they differ."""
    resolved = Gemma4_26B_A4B_Recipe(num_rollout=5)._for_dataset(image_dataset)
    assert resolved.num_rollout == 5
    # ...while a field left alone still picks the vision default up.
    assert resolved.rollout_batch_size == 8


def test_vision_mode_applies_when_nothing_was_set(image_dataset):
    resolved = Gemma4_26B_A4B_Recipe()._for_dataset(image_dataset)
    assert resolved.num_rollout == 15
    assert resolved.rollout_top_k == 64


def test_text_dataset_leaves_recipe_alone():
    recipe = Gemma4_26B_A4B_Recipe()
    assert recipe._for_dataset(None) is recipe


def test_post_construction_assignment_counts_as_chosen(image_dataset):
    """A sweep sets values with setattr; they must not be treated as defaults."""
    recipe = Gemma4_26B_A4B_Recipe()
    recipe.rollout_temperature = 0.3
    assert "rollout_temperature" in recipe.explicit_fields
    assert recipe._for_dataset(image_dataset).rollout_temperature == 0.3


def test_swept_value_survives_a_revalidating_rebuild(image_dataset):
    """TrainingGroup mutates then rebuilds; the rebuild must keep the override."""
    recipe = Gemma4_26B_A4B_Recipe()
    recipe.rollout_temperature = 0.3
    values = {f.name: getattr(recipe, f.name) for f in dc.fields(recipe) if f.init}
    rebuilt = carry_explicit_fields(recipe, type(recipe)(**values))

    assert "rollout_temperature" in rebuilt.explicit_fields
    assert rebuilt._for_dataset(image_dataset).rollout_temperature == 0.3
    # A field nobody touched still resolves to the vision default.
    assert rebuilt._for_dataset(image_dataset).num_rollout == 15


def test_subclass_declared_fields_count_as_chosen(image_dataset):
    """Fields declared in a subclass body are that author's config, not defaults."""

    @pdc.dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
    class Custom(Gemma4_26B_A4B_Recipe):
        num_rollout: int = 5

    assert "num_rollout" in Custom().explicit_fields
    assert Custom()._for_dataset(image_dataset).num_rollout == 5


def test_recipe_own_fields_are_not_treated_as_caller_choices():
    """The preset's own declarations are the defaults vision mode overrides."""
    recipe = Gemma4_26B_A4B_Recipe()
    assert recipe.explicit_fields == frozenset()


def test_gemma4_is_wired_into_the_preset_lookup():
    from modal_training_gym.common.models import Gemma4_26B_A4B

    preset = MilesRecipe.get_base_recipe(Gemma4_26B_A4B())
    assert isinstance(preset, Gemma4_26B_A4B_Recipe)


def test_image_patches_survive_caller_supplied_run_commands():
    """The build-time patches are additive, not a default a caller can replace."""
    recipe = Gemma4_26B_A4B_Recipe(image_run_commands=["pip install foo"])
    patch_cmds = [c for c in recipe.image_run_commands if "base64 -d | python3" in c]

    assert len(patch_cmds) == 2
    assert recipe.image_run_commands[: len(patch_cmds)] == patch_cmds
    assert "pip install foo" in recipe.image_run_commands
    # Re-validating must not stack a second copy.
    again = type(recipe)(
        **{f.name: getattr(recipe, f.name) for f in dc.fields(recipe) if f.init}
    )
    assert len([c for c in again.image_run_commands if "base64 -d | python3" in c]) == 2


def test_disk_reservation_survives_caller_supplied_train_kwargs():
    """Attaching a secret must not silently drop the 1 TiB reservation."""
    secret = object()
    recipe = Gemma4_26B_A4B_Recipe(train_function_kwargs={"secrets": [secret]})

    assert recipe.train_function_kwargs["ephemeral_disk"] == 1_048_576
    assert recipe.train_function_kwargs["secrets"] == [secret]


def test_caller_can_still_choose_the_disk_size():
    recipe = Gemma4_26B_A4B_Recipe(train_function_kwargs={"ephemeral_disk": 512})
    assert recipe.train_function_kwargs == {"ephemeral_disk": 512}


def test_gemma4_only_overrides_fields_miles_declares():
    """A bare MilesRecipe must be able to override anything the preset sets.

    ``_merge_recipe`` iterates ``dataclasses.fields(MilesRecipe)``, so a field the
    preset declares on its own is unreachable from a plain ``MilesRecipe(...)``.
    """
    base_names = {f.name for f in dc.fields(MilesRecipe)}
    preset_only = {f.name for f in dc.fields(Gemma4_26B_A4B_Recipe)} - base_names

    assert preset_only == set()
