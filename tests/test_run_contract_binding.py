from __future__ import annotations

import pytest

from modal_training_gym.frameworks.slime.launcher import (
    _image_overlay_contract,
    _serialize_slime_params,
    _validate_committed_dataset_inputs,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


def _overlay(image):
    return image


def test_committed_overlay_contract_hashes_complete_declared_source_tree(
    tmp_path,
) -> None:
    source = tmp_path / "train"
    source.mkdir()
    runtime = source / "runtime.py"
    runtime.write_text("VALUE = 1\n")

    original = _image_overlay_contract(
        _overlay,
        [str(source)],
        required=True,
    )
    runtime.write_text("VALUE = 2\n")
    changed = _image_overlay_contract(
        _overlay,
        [str(source)],
        required=True,
    )

    assert original is not None
    assert changed is not None
    assert original["source_roots"][0]["sha256"] != changed["source_roots"][0]["sha256"]
    assert str(tmp_path) not in repr(original)


def test_overlay_source_receipt_is_independent_of_parent_checkout_path(
    tmp_path,
) -> None:
    first = tmp_path / "checkout-a" / "train"
    second = tmp_path / "checkout-b" / "train"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "runtime.py").write_text("VALUE = 1\n")
    (second / "runtime.py").write_text("VALUE = 1\n")

    first_receipt = _image_overlay_contract(
        _overlay,
        [str(first)],
        required=True,
    )
    second_receipt = _image_overlay_contract(
        _overlay,
        [str(second)],
        required=True,
    )

    assert first_receipt == second_receipt


def test_committed_overlay_requires_declared_source_inputs() -> None:
    with pytest.raises(ValueError, match="image_overlay_source_roots"):
        _image_overlay_contract(_overlay, [], required=True)


def test_committed_mode_rejects_dataset_rebuilds() -> None:
    recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=4,
        rollout_max_response_len=256,
        rollout_temperature=1.0,
        save_interval=1,
        attempt_mode="committed",
    )

    class _Dataset:
        always_prepare = True

    with pytest.raises(ValueError, match="always_prepare=False"):
        _validate_committed_dataset_inputs(recipe, _Dataset())  # type: ignore[arg-type]


def test_committed_attempt_mode_is_present_in_reporting_config() -> None:
    recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=4,
        rollout_max_response_len=256,
        rollout_temperature=1.0,
        save_interval=1,
        attempt_mode="committed",
    )

    assert _serialize_slime_params(recipe)["attempt_mode"] == "committed"
