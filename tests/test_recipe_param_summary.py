from __future__ import annotations

from modal_training_gym.common.models import ModelConfig, ModelArchitecture
from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.train import TrainConfig, _recipe_param_summary
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4b_Recipe


def _make_model(name: str = "Qwen/Qwen3-4B") -> ModelConfig:
    return ModelConfig(
        model_name=name,
        architecture=ModelArchitecture(
            num_layers=36,
            hidden_size=2560,
            ffn_hidden_size=6912,
            num_attention_heads=32,
            group_query_attention=True,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=151936,
        ),
    )


def _make_dataset() -> HuggingFaceDataset:
    return HuggingFaceDataset(
        hf_repo="some/dataset",
        input_key="messages",
        label_key="label",
    )


def test_preset_fields_tagged_as_preset():
    """When using a known model with a preset, preset-provided fields are tagged 'preset'."""
    user_recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=16,
        rollout_max_response_len=4096,
        rollout_temperature=1.0,
        save_interval=10,
    )
    base_recipe = Qwen3_4b_Recipe()
    from modal_training_gym.common.train import _merge_recipe

    combined = _merge_recipe(base_recipe, user_recipe)

    params = _recipe_param_summary(user_recipe, combined, base_recipe)

    assert params["n_samples_per_prompt"]["value"] == 8
    assert params["n_samples_per_prompt"]["source"] == "preset"

    assert params["lr"]["value"] == 5e-7
    assert params["lr"]["source"] == "preset"

    assert params["gpu_type"]["value"] == "H100"


def test_user_overrides_tagged_as_user():
    """When the user overrides a preset field, it's tagged 'user'."""
    user_recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=16,
        rollout_max_response_len=4096,
        rollout_temperature=1.0,
        save_interval=10,
        lr=1e-5,
    )
    base_recipe = Qwen3_4b_Recipe()
    from modal_training_gym.common.train import _merge_recipe

    combined = _merge_recipe(base_recipe, user_recipe)

    params = _recipe_param_summary(user_recipe, combined, base_recipe)

    assert params["lr"]["value"] == 1e-5
    assert params["lr"]["source"] == "user"


def test_default_fields_tagged_as_default():
    """Fields not set by user or preset are tagged 'default'."""
    user_recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=16,
        rollout_max_response_len=4096,
        rollout_temperature=1.0,
        save_interval=10,
    )
    base_recipe = Qwen3_4b_Recipe()
    from modal_training_gym.common.train import _merge_recipe

    combined = _merge_recipe(base_recipe, user_recipe)

    params = _recipe_param_summary(user_recipe, combined, base_recipe)

    assert params["weight_decay"]["value"] == 0.1
    assert params["weight_decay"]["source"] == "default"

    assert params["adam_beta1"]["value"] == 0.9
    assert params["adam_beta1"]["source"] == "default"


def test_no_preset_all_user_or_default():
    """When there's no model preset, fields are either 'user' or 'default'."""
    user_recipe = SlimeRecipe(
        gpu_type="A100",
        colocate=False,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        rollout_num_gpus_per_engine=2,
        num_rollout=2,
        rollout_batch_size=32,
        rollout_max_response_len=8192,
        rollout_temperature=0.7,
        save_interval=5,
        lr=3e-6,
    )

    params = _recipe_param_summary(user_recipe, user_recipe, None)

    assert params["gpu_type"]["value"] == "A100"
    assert params["gpu_type"]["source"] == "user"

    assert params["weight_decay"]["value"] == 0.1
    assert params["weight_decay"]["source"] == "default"

    for info in params.values():
        assert info["source"] in ("user", "default")


def test_train_config_recipe_param_summary(monkeypatch):
    """TrainConfig.recipe_param_summary() integrates preset resolution end-to-end."""
    monkeypatch.setattr(
        "modal_training_gym.common.train.create_hash",
        lambda *_: "test-hash",
    )

    config = TrainConfig(
        model=_make_model("Qwen/Qwen3-4B"),
        dataset=_make_dataset(),
        recipe=SlimeRecipe(
            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            num_rollout=1,
            rollout_batch_size=16,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,
            save_interval=10,
        ),
    )

    params = config.recipe_param_summary()

    assert params["n_samples_per_prompt"]["source"] == "preset"
    assert params["n_samples_per_prompt"]["value"] == 8

    assert params["lr"]["source"] == "preset"
    assert params["lr"]["value"] == 5e-7

    assert params["weight_decay"]["source"] == "default"


def test_serialized_recipe_fields():
    """_serialize_recipe_fields in the launcher produces JSON-safe values."""
    from modal_training_gym.frameworks.slime.launcher import _serialize_recipe_fields

    recipe = Qwen3_4b_Recipe()
    fields = _serialize_recipe_fields(recipe)

    assert fields["gpu_type"] == "H100"
    assert fields["colocate"] is True
    assert fields["lr"] == 5e-7
    assert fields["n_samples_per_prompt"] == 8
    assert fields["tensor_model_parallel_size"] == 1

    import json

    json.dumps(fields)


def test_serialized_slime_params_include_full_debug_fields():
    """Dashboard debug payload includes full Slime CLI params, not just a summary."""
    from modal_training_gym.frameworks.slime.launcher import _serialize_slime_params

    recipe = Qwen3_4b_Recipe(
        rollout_stop_token_ids=[151643, 151645],
        train_env_vars={"WANDB_API_KEY": "secret-value", "SAFE_FLAG": "1"},
    )
    fields = _serialize_slime_params(
        recipe, dataset=_make_dataset(), model=_make_model()
    )

    assert fields["gpu_type"] == "H100"
    assert fields["n_samples_per_prompt"] == 8
    assert fields["train_env_vars"]["WANDB_API_KEY"] == "[redacted]"
    assert fields["train_env_vars"]["SAFE_FLAG"] == "1"
    assert fields["rollout_stop_token_ids"] == [151643, 151645]
    assert fields["prompt_data"] == "/data/some_dataset/train.parquet"
    assert fields["num_layers"] == 36

    import json

    json.dumps(fields)


def test_param_summary_values_are_json_serializable():
    """All values in the param summary must be JSON-serializable."""
    import json

    user_recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=16,
        rollout_max_response_len=4096,
        rollout_temperature=1.0,
        save_interval=10,
    )
    base_recipe = Qwen3_4b_Recipe()
    from modal_training_gym.common.train import _merge_recipe

    combined = _merge_recipe(base_recipe, user_recipe)

    params = _recipe_param_summary(user_recipe, combined, base_recipe)

    serialized = json.dumps(params)
    assert len(serialized) > 0
