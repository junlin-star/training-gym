"""Thin coverage for ``train_function_kwargs['volumes']`` on slime prepare/train."""

from __future__ import annotations

import pytest
from modal import Volume

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelArchitecture, ModelConfig
from modal_training_gym.frameworks.slime.launcher import build_slime_app
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

_SLIME_KW = dict(
    gpu_type="H100",
    colocate=True,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    rollout_num_gpus_per_engine=8,
    num_rollout=1,
    rollout_batch_size=8,
    rollout_max_response_len=128,
    rollout_temperature=1.0,
    save_interval=10,
    actor_num_gpus_per_node=8,
)


class _TinyDataset(DatasetConfig):
    label_key = "label"
    input_key = "prompt"

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        return None


def _build_app(train_function_kwargs: dict) -> object:
    recipe = SlimeRecipe(**_SLIME_KW, train_function_kwargs=train_function_kwargs)
    model = ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        architecture=ModelArchitecture(),
    )
    return build_slime_app(
        training_run_id="test-train-fn-kwargs",
        slime=recipe,
        model=model,
        dataset=_TinyDataset(),
        name="slime-train-fn-kwargs-test",
    )


def test_volumes_mount_on_prepare_and_train_not_convert() -> None:
    prompts = Volume.from_name("test-extra-volume", create_if_missing=True)
    app = _build_app({"volumes": {"/prompts": prompts}})

    prepare = app.registered_functions["prepare_dataset"]
    train = app.registered_functions["train"]
    convert = app.registered_functions["convert_checkpoint"]

    assert set(prepare.spec.volumes) == {"/data", "/prompts"}
    assert "test-extra-volume" in repr(prepare.spec.volumes["/prompts"])

    assert "/prompts" in train.spec.volumes
    assert "test-extra-volume" in repr(train.spec.volumes["/prompts"])
    assert set(train.spec.volumes) >= {
        "/data",
        "/checkpoints",
        "/metadata",
        "/prompts",
        "/root/.cache/huggingface",
    }

    assert "/prompts" not in convert.spec.volumes
    assert set(convert.spec.volumes) == {
        "/data",
        "/checkpoints",
        "/metadata",
        "/root/.cache/huggingface",
    }


def test_volumes_reject_reserved_mount() -> None:
    with pytest.raises(ValueError, match="cannot override reserved mounts: /data"):
        _build_app(
            {"volumes": {"/data": Volume.from_name("x", create_if_missing=True)}}
        )


def test_volumes_must_be_volume_instances() -> None:
    with pytest.raises(TypeError, match="values must be modal.Volume"):
        _build_app({"volumes": {"/prompts": "not-a-volume"}})
