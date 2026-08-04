"""Gemma-4 is one checkpoint with two modes, selected by the dataset's modality.

`Gemma4_26B_A4B_Recipe._for_dataset` swaps in vision-mode values on an image
dataset, but only for fields the caller left alone. These tests pin both halves
of that contract, and in particular that "left alone" means *not passed to the
constructor* rather than *equal to the text default* — several vision overrides
sit on fields whose text default is a value a caller would plausibly ask for.
"""

import pytest

from modal_training_gym import (
    Gemma4_26B_A4B,
    Gemma4_26B_A4B_Recipe,
    MultimodalDataset,
)
from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.train import _resolve_slime_recipe


class _TextDataset(HuggingFaceDataset):
    hf_repo = "openai/gsm8k"
    input_key = "messages"
    label_key = "label"


def _image_dataset() -> MultimodalDataset:
    return MultimodalDataset(
        modality="image",
        rows=[{"prompt": "<image> q", "media": ["ref"], "label": "a"}],
    )


def _resolve(recipe, dataset, *, merge=False):
    return _resolve_slime_recipe(
        Gemma4_26B_A4B(), recipe, dataset, merge_model_recipe=merge
    )


def test_text_dataset_leaves_recipe_untouched():
    resolved = _resolve(Gemma4_26B_A4B_Recipe(), _TextDataset())

    assert resolved.megatron_to_hf_mode == "raw"
    assert resolved.slime_model_script == "scripts/models/gemma4-26B-A4B.sh"
    assert resolved.pipeline_model_parallel_size == 2
    assert resolved.freeze_params_name_list is None
    assert resolved.image_run_commands == []


def test_image_dataset_selects_vision_mode():
    resolved = _resolve(Gemma4_26B_A4B_Recipe(), _image_dataset())

    assert resolved.megatron_to_hf_mode == "bridge"
    assert resolved.slime_model_script == ""
    # The bridge keeps the vision tower and embedding on one pipeline stage.
    assert resolved.pipeline_model_parallel_size == 1
    assert resolved.attention_backend is None
    assert resolved.use_dynamic_batch_size is False
    assert resolved.freeze_params_name_list == ["vision_tower", "embed_vision"]
    assert resolved.extra_config["qkv_format"] == "bshd"
    # The four upstream shims a bridge-mode VL model needs at image-build time.
    assert len(resolved.image_run_commands) == 4
    # Untouched tuning fields take the vision defaults.
    assert resolved.num_rollout == 15
    assert resolved.rollout_max_response_len == 256


@pytest.mark.parametrize(
    "field, value",
    [
        # Each of these equals the text-mode default, so comparing values cannot
        # tell it from an unset field. All three have a different vision default.
        ("num_rollout", 2),
        ("rollout_max_response_len", 512),
        ("save_interval", 20),
        # And one that differs from both defaults, as a control.
        ("num_rollout", 7),
    ],
)
def test_explicit_value_survives_vision_mode(field, value):
    resolved = _resolve(Gemma4_26B_A4B_Recipe(**{field: value}), _image_dataset())

    assert getattr(resolved, field) == value
    # Still vision mode — an explicit override must not opt out of the bridge.
    assert resolved.megatron_to_hf_mode == "bridge"


def test_explicit_value_survives_the_preset_merge():
    """`merge_model_recipe=True` rebuilds the recipe from every field.

    That reconstruction must not read as the caller having set all of them,
    which would leave vision mode nothing to override.
    """
    recipe = Gemma4_26B_A4B_Recipe(num_rollout=2, tensor_model_parallel_size=4)
    resolved = _resolve(recipe, _image_dataset(), merge=True)

    assert resolved.num_rollout == 2
    assert resolved.tensor_model_parallel_size == 4
    assert resolved.megatron_to_hf_mode == "bridge"
    # Fields the caller never named still pick up their vision values.
    assert resolved.rollout_max_response_len == 256
    assert resolved.pipeline_model_parallel_size == 1


def test_resolution_is_idempotent():
    recipe = Gemma4_26B_A4B_Recipe(num_rollout=2)
    once = recipe._for_dataset(_image_dataset())
    twice = once._for_dataset(_image_dataset())

    assert twice.num_rollout == once.num_rollout == 2
    assert twice.pipeline_model_parallel_size == once.pipeline_model_parallel_size
    assert twice.image_run_commands == once.image_run_commands


def test_vision_mode_rejects_pipeline_parallelism():
    from modal_training_gym.common.errors import TrainingGymConfigError

    recipe = Gemma4_26B_A4B_Recipe(pipeline_model_parallel_size=2)
    with pytest.raises(TrainingGymConfigError, match="pipeline_model_parallel_size=1"):
        _resolve(recipe, _image_dataset())
