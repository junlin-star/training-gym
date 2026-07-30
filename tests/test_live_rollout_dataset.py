import json

import pytest

from modal_training_gym import (
    HuggingFaceDataset,
    LiveRolloutDataset,
    MultimodalDataset,
    Qwen3_0_6B,
    Qwen3_VL_8B,
    SlimeRecipe,
    TrainConfig,
    TrainingGroup,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.train import _resolve_slime_recipe
from modal_training_gym.common.training_group import TrainingGroupError
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesConfig


async def _fake_generate(*_args, **_kwargs):
    return None


_RECIPE_KW = dict(
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
    custom_generate_function=_fake_generate,
)


def _live_recipe(**overrides):
    return SlimeRecipe(**{**_RECIPE_KW, **overrides})


def _omit_cfg(**recipe_overrides):
    return TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=_live_recipe(**recipe_overrides),
        merge_model_recipe=False,
    )


def test_prepare_writes_n_prompt_label_rows(tmp_path):
    ds = LiveRolloutDataset(n_rows=3)
    out = str(tmp_path / "train.jsonl")
    ds.prepare(out)
    lines = open(out).read().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert set(row) == {"prompt", "label"}
        assert row["label"] == str(i)
        assert "Placeholder prompt" in row["prompt"]
    ds.validate_prepared(out)


def test_load_honors_input_and_label_keys():
    ds = LiveRolloutDataset(n_rows=2, input_key="q", label_key="a")
    rows = ds.load(split="train")
    assert rows == [{"q": ds._prompt, "a": "0"}, {"q": ds._prompt, "a": "1"}]


def test_train_config_omits_dataset_interpolates_live_rollout():
    cfg = _omit_cfg()
    assert isinstance(cfg.dataset, LiveRolloutDataset)
    assert cfg.dataset.n_rows == _RECIPE_KW["rollout_batch_size"]
    assert cfg.dataset.auto_sized is True
    assert cfg.dataset.writes_eval_paths is False


def test_omit_without_custom_generate_raises():
    cfg = TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=SlimeRecipe(
            **{k: v for k, v in _RECIPE_KW.items() if k != "custom_generate_function"}
        ),
        merge_model_recipe=False,
    )
    assert cfg.dataset is None
    with pytest.raises(TrainingGymConfigError, match="custom_generate_function"):
        cfg._build_app()


def test_omit_with_miles_raises():
    cfg = TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=MilesConfig(
            colocate=False,
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=4,
        ),
    )
    assert cfg.dataset is None
    with pytest.raises(TrainingGymConfigError, match="SlimeRecipe"):
        cfg._build_app()


def test_explicit_hf_and_multimodal_unchanged():
    hf = HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
    )
    cfg_hf = TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=_live_recipe(),
        dataset=hf,
        merge_model_recipe=False,
    )
    assert cfg_hf.dataset is hf

    mm = MultimodalDataset(
        rows=[{"prompt": "x", "media": ["http://example.com/a.wav"], "label": "0"}],
    )
    cfg_mm = TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=_live_recipe(),
        dataset=mm,
        merge_model_recipe=False,
    )
    assert cfg_mm.dataset is mm


def test_relaunch_resize_auto_sized_stub():
    cfg = _omit_cfg(rollout_batch_size=4)
    assert cfg.dataset.n_rows == 4
    object.__setattr__(cfg, "recipe", _live_recipe(rollout_batch_size=8))
    assert isinstance(cfg.dataset, LiveRolloutDataset)
    assert cfg.dataset.n_rows == 8


def test_auto_stub_never_writes_eval_paths():
    no_eval = _omit_cfg(eval_interval=None)
    assert no_eval.dataset.writes_eval_paths is False

    with_eval = _omit_cfg(eval_interval=10)
    assert with_eval.dataset.writes_eval_paths is False

    merged = TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=_live_recipe(),
        merge_model_recipe=True,
    )
    assert merged.dataset.writes_eval_paths is False


def test_omit_merge_clears_preset_eval_interval():
    cfg = TrainConfig(
        model=Qwen3_0_6B(),
        recipe=_live_recipe(eval_interval=None),
        merge_model_recipe=True,
    )
    assert isinstance(cfg.dataset, LiveRolloutDataset)
    assert cfg.dataset.writes_eval_paths is False

    resolved = _resolve_slime_recipe(
        cfg.model,
        cfg.recipe,
        merge_model_recipe=True,
        dataset=cfg.dataset,
    )
    assert resolved.eval_interval is None
    _, eval_paths = SlimeRecipe._resolve_data_paths(cfg.dataset)
    assert eval_paths is None
    args = resolved.cli_args(dataset=cfg.dataset)
    assert "--eval-interval" not in args
    assert "--eval-prompt-data" not in args


def test_hf_dataset_keeps_merged_eval_interval():
    hf = HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
    )
    cfg = TrainConfig(
        model=Qwen3_0_6B(),
        recipe=SlimeRecipe(
            **{k: v for k, v in _RECIPE_KW.items() if k != "custom_generate_function"}
        ),
        dataset=hf,
        merge_model_recipe=True,
    )
    resolved = _resolve_slime_recipe(
        cfg.model,
        cfg.recipe,
        merge_model_recipe=True,
        dataset=cfg.dataset,
    )
    assert resolved.eval_interval == 10
    fields = resolved._fields(dataset=cfg.dataset)
    assert fields["eval_prompt_data"] is not None


def test_writes_eval_paths_false_non_live_keeps_eval_interval():
    """BFCL/Toolathlon set writes_eval_paths=False for a separate eval DatasetConfig."""
    hf = HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
    )
    hf.writes_eval_paths = False
    cfg = TrainConfig(
        model=Qwen3_0_6B(),
        recipe=SlimeRecipe(
            **{k: v for k, v in _RECIPE_KW.items() if k != "custom_generate_function"}
        ),
        dataset=hf,
        merge_model_recipe=True,
    )
    resolved = _resolve_slime_recipe(
        cfg.model,
        cfg.recipe,
        merge_model_recipe=True,
        dataset=cfg.dataset,
    )
    assert resolved.eval_interval == 10


def test_resolve_data_paths_includes_n_rows_slug():
    cfg = _omit_cfg(rollout_batch_size=4)
    prompt_data, eval_paths = SlimeRecipe._resolve_data_paths(cfg.dataset)
    assert prompt_data.startswith("/data/live-rollout-4-")
    assert prompt_data.endswith("/train.jsonl")
    assert eval_paths is None

    cfg8 = _omit_cfg(rollout_batch_size=8)
    prompt8, _ = SlimeRecipe._resolve_data_paths(cfg8.dataset)
    assert prompt8.startswith("/data/live-rollout-8-")
    assert prompt_data != prompt8


def test_dataset_to_fields_has_live_rollout_path_no_multimodal():
    cfg = _omit_cfg()
    fields = SlimeRecipe._dataset_to_fields(cfg.dataset)
    assert "live-rollout-4" in fields["prompt_data"]
    assert "multimodal_keys" not in fields
    args = _live_recipe().cli_args(dataset=cfg.dataset)
    assert "--multimodal-keys" not in args


def test_omit_vl_interpolates_multimodal_keys():
    cfg = TrainConfig(
        model=Qwen3_VL_8B(),
        recipe=_live_recipe(),
        merge_model_recipe=False,
    )
    assert isinstance(cfg.dataset, LiveRolloutDataset)
    assert cfg.dataset.modality == "image"
    assert cfg.dataset.multimodal_keys == {"image": "images"}
    assert cfg.dataset.writes_eval_paths is False
    fields = SlimeRecipe._dataset_to_fields(cfg.dataset)
    assert fields["multimodal_keys"] == {"image": "images"}
    assert "live-rollout-4-image" in fields["prompt_data"]
    args = _live_recipe().cli_args(dataset=cfg.dataset)
    assert "--multimodal-keys" in args
    assert json.dumps({"image": "images"}) in args


def test_omit_vl_paths_size_distinct():
    cfg4 = TrainConfig(
        model=Qwen3_VL_8B(),
        recipe=_live_recipe(rollout_batch_size=4),
        merge_model_recipe=False,
    )
    cfg8 = TrainConfig(
        model=Qwen3_VL_8B(),
        recipe=_live_recipe(rollout_batch_size=8),
        merge_model_recipe=False,
    )
    p4, _ = SlimeRecipe._resolve_data_paths(cfg4.dataset)
    p8, _ = SlimeRecipe._resolve_data_paths(cfg8.dataset)
    assert p4.startswith("/data/live-rollout-4-image-")
    assert p8.startswith("/data/live-rollout-8-image-")
    assert p4.endswith("/train.jsonl")
    assert p8.endswith("/train.jsonl")
    assert p4 != p8


def test_live_rollout_modality_writes_empty_media(tmp_path):
    ds = LiveRolloutDataset(n_rows=2, modality="image")
    out = str(tmp_path / "train.jsonl")
    ds.prepare(out)
    for line in open(out):
        row = json.loads(line)
        assert row["images"] == []
        assert set(row) == {"prompt", "label", "images"}


def test_live_rollout_n_rows_zero_raises():
    with pytest.raises(TrainingGymConfigError):
        LiveRolloutDataset(n_rows=0)


def test_training_group_updates_public_dataset_n_rows():
    base = _omit_cfg(rollout_batch_size=4)
    assert isinstance(base.dataset, LiveRolloutDataset)
    assert base.dataset.n_rows == 4
    assert base.dataset.auto_sized is True

    group = TrainingGroup(
        base=base,
        grid={"recipe.rollout_batch_size": [4, 8, 16]},
    )
    configs = group.get_train_configs()
    sizes = sorted(cfg.dataset.n_rows for cfg in configs)
    assert sizes == [4, 8, 16]
    slugs = {cfg.dataset.data_dir_name for cfg in configs}
    assert len(slugs) == 3
    assert {s.rsplit("-", 1)[0] for s in slugs} == {
        "live-rollout-4",
        "live-rollout-8",
        "live-rollout-16",
    }


def test_training_group_accepts_dataset_grid_path():
    hf = HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
    )
    base = TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=_live_recipe(),
        dataset=hf,
        merge_model_recipe=False,
    )
    group = TrainingGroup(base=base, grid={"dataset.n_rows": [10, 20]})
    configs = group.get_train_configs()
    assert sorted(c.dataset.n_rows for c in configs) == [10, 20]


def test_dataset_grid_path_on_auto_sized_stub_raises():
    with pytest.raises(TrainingGroupError, match="recipe.rollout_batch_size"):
        TrainingGroup(base=_omit_cfg(), grid={"dataset.n_rows": [10, 20]})


def test_dataset_grid_path_without_dataset_raises():
    base = TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=SlimeRecipe(
            **{k: v for k, v in _RECIPE_KW.items() if k != "custom_generate_function"}
        ),
        merge_model_recipe=False,
    )
    with pytest.raises(TrainingGroupError, match="is None"):
        TrainingGroup(base=base, grid={"dataset.n_rows": [10]})


def test_subclass_class_attrs_survive_init():
    class AudioLive(LiveRolloutDataset):
        modality = "audio"
        input_key = "q"
        writes_eval_paths = True

    ds = AudioLive(n_rows=2)
    assert ds.modality == "audio"
    assert ds.writes_eval_paths is True
    assert ds.media_column == "audios"
    assert ds.multimodal_keys == {"audio": "audios"}
    assert ds.load()[0] == {"q": ds._prompt, "label": "0", "audios": []}


def test_auto_sized_stub_dropped_when_recipe_loses_custom_generate():
    cfg = _omit_cfg()
    assert isinstance(cfg.dataset, LiveRolloutDataset)
    cfg.recipe = SlimeRecipe(
        **{k: v for k, v in _RECIPE_KW.items() if k != "custom_generate_function"}
    )
    assert cfg.dataset is None
    with pytest.raises(TrainingGymConfigError, match="custom_generate_function"):
        cfg._build_app()


def test_live_rollout_data_dir_is_content_addressed():
    assert LiveRolloutDataset(n_rows=2).always_prepare is False
    assert _omit_cfg().dataset.always_prepare is False

    base = LiveRolloutDataset(n_rows=2)
    assert LiveRolloutDataset(n_rows=2).data_dir_name == base.data_dir_name
    variants = [
        LiveRolloutDataset(n_rows=2, prompt="a different prompt"),
        LiveRolloutDataset(n_rows=2, input_key="question"),
        LiveRolloutDataset(n_rows=2, label_key="target"),
        LiveRolloutDataset(n_rows=2, modality="image"),
        LiveRolloutDataset(n_rows=3),
    ]
    names = {v.data_dir_name for v in variants}
    assert base.data_dir_name not in names
    assert len(names) == len(variants)


def test_explicit_live_rollout_not_resized_by_resolve():
    ds = LiveRolloutDataset(n_rows=2)
    cfg = TrainConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-4B"),
        recipe=_live_recipe(rollout_batch_size=8),
        dataset=ds,
        merge_model_recipe=False,
    )
    assert cfg._require_dataset() is cfg.dataset
    assert cfg.dataset.n_rows == 2
