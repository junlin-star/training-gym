from __future__ import annotations

from dataclasses import dataclass, field

from modal_training_gym.common import ids
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType


def test_create_hash_has_word_word_hash_shape() -> None:
    value = ids.create_hash("model", "ckpt", "recipe", "app", "path")

    assert value.count("-") >= 2


def test_create_hash_suffix_is_stable_for_same_parts(monkeypatch) -> None:
    monkeypatch.setattr(ids.randomname, "get_name", lambda *, sep: "brisk-river")
    first = ids.create_hash("model", "ckpt", "recipe", "app", "path")
    second = ids.create_hash("model", "ckpt", "recipe", "app", "path")

    assert first.rsplit("-", 1)[-1] == second.rsplit("-", 1)[-1]


def test_create_hash_suffix_differs_for_different_parts(monkeypatch) -> None:
    monkeypatch.setattr(ids.randomname, "get_name", lambda *, sep: "brisk-river")
    first = ids.create_hash("model-a", "ckpt", "recipe", "app", "path")
    second = ids.create_hash("model-b", "ckpt", "recipe", "app", "path")

    assert first.rsplit("-", 1)[-1] != second.rsplit("-", 1)[-1]


def test_train_config_keeps_stable_training_run_id(monkeypatch) -> None:
    class DummyDataset(DatasetConfig):
        label_key = "label"

    @dataclass
    class DummyRecipe(BaseTrainRecipe):
        recipe_type: RecipeType = field(default=RecipeType.SLIME)

    calls = 0

    def fake_create_hash(*parts: str) -> str:
        nonlocal calls
        calls += 1
        return "brisk-river-deadbeef"

    monkeypatch.setattr(
        "modal_training_gym.common.train.create_hash",
        fake_create_hash,
    )

    config = TrainConfig(
        dataset=DummyDataset(),
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=DummyRecipe(),
    )

    assert config.training_run_id == "brisk-river-deadbeef"
    assert config.training_run_id == "brisk-river-deadbeef"
    assert calls == 1
