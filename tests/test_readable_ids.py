from __future__ import annotations

from dataclasses import dataclass, field

from modal_training_gym.common import ids
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.ids import GymObjectId
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType
from modal_training_gym.utils.metadata import MetadataStore


def test_config_fingerprint_is_stable_for_dict_order() -> None:
    @dataclass
    class _Recipe:
        lr: int
        tp: int

    @dataclass
    class _Bundle:
        model: str
        recipe: _Recipe

    left = ids.config_fingerprint(_Bundle("qwen", _Recipe(1, 4)))
    right = ids.config_fingerprint(_Bundle("qwen", _Recipe(tp=4, lr=1)))

    assert left == right


def test_hash_suffix_uses_id_created_at(monkeypatch) -> None:
    monkeypatch.setattr(ids, "vol_list", lambda _store: [])
    fingerprint = ids.config_fingerprint("model", "dataset", "recipe")
    first = ids.stable_readable_id(
        MetadataStore.TRAINING_RUNS,
        fingerprint,
        id_key="training_run_id",
        id_created_at=1_700_000_000,
    )
    second = ids.stable_readable_id(
        MetadataStore.TRAINING_RUNS,
        fingerprint,
        id_key="training_run_id",
        id_created_at=1_700_000_001,
    )

    assert first.id_created_at == 1_700_000_000
    assert second.id_created_at == 1_700_000_001
    assert first.value != second.value


def test_stable_readable_id_avoids_collision(monkeypatch) -> None:
    fingerprint = ids.config_fingerprint("same-config")
    taken_suffix = (
        __import__("hashlib")
        .sha256(f"{fingerprint}:1700000000".encode())
        .hexdigest()[:5]
    )
    taken_id = f"brisk-river-{taken_suffix}"

    def fake_vol_list(store: MetadataStore) -> list[dict[str, object]]:
        assert store is MetadataStore.TRAINING_RUNS
        return [{"training_run_id": taken_id}]

    monkeypatch.setattr(ids, "vol_list", fake_vol_list)
    monkeypatch.setattr(ids.randomname, "get_name", lambda *, sep: "brisk-river")

    stable_id = ids.stable_readable_id(
        MetadataStore.TRAINING_RUNS,
        fingerprint,
        id_key="training_run_id",
        id_created_at=1_700_000_000,
    )

    assert stable_id.value != taken_id


def test_train_config_keeps_stable_training_run_id(monkeypatch) -> None:
    class DummyDataset(DatasetConfig):
        label_key = "label"

    @dataclass
    class DummyRecipe(BaseTrainRecipe):
        recipe_type: RecipeType = field(default=RecipeType.SLIME)

    calls = 0

    def fake_stable_readable_id(*args, **kwargs) -> GymObjectId:
        nonlocal calls
        calls += 1
        return GymObjectId(
            value="brisk-river-a1b2c",
            config_fingerprint="fp",
            id_created_at=1_700_000_000,
        )

    monkeypatch.setattr(
        "modal_training_gym.common.train.stable_readable_id",
        fake_stable_readable_id,
    )

    config = TrainConfig(
        dataset=DummyDataset(),
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=DummyRecipe(),
    )

    assert config.training_run_id == "brisk-river-a1b2c"
    assert config.training_run_id == "brisk-river-a1b2c"
    assert calls == 1
