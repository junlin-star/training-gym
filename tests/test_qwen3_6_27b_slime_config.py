import importlib.util
import json
from pathlib import Path

from modal_training_gym import Qwen3_6_27B, TrainConfig
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.frameworks.slime.modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
)
from modal_training_gym.train_recipes.slime_recipe import (
    Qwen3_6_27b_Recipe,
    SlimeRecipe,
)


class TinyMathDataset(DatasetConfig):
    dataset_id = "tiny-qwen36-27b-smoke"
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        row = {
            "messages": [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "2"},
            ],
            "label": "2",
        }
        Path(path).write_text(json.dumps(row) + "\n")

    def load(self, split="all"):
        return []


def test_qwen3_6_27b_model_architecture_matches_hf_and_slime_script() -> None:
    model = Qwen3_6_27B()
    arch = model.architecture

    assert model.model_name == "Qwen/Qwen3.6-27B"
    assert arch is not None
    assert arch.num_layers == 64
    assert arch.hidden_size == 5120
    assert arch.ffn_hidden_size == 17408
    assert arch.num_attention_heads == 24
    assert arch.num_query_groups == 4
    assert arch.kv_channels == 256
    assert arch.vocab_size == 248320
    assert arch.megatron_spec == ["slime_plugins.models.qwen3_5", "get_qwen3_5_spec"]
    assert arch.megatron_model_type == "qwen3.5-27B"
    assert arch.use_gated_attention is True
    assert arch.attention_output_gate is True
    assert arch.rotary_base == 10000000
    assert arch.rotary_percent == 0.25


def test_qwen3_6_27b_recipe_conversion_policy_and_train_command() -> None:
    model = Qwen3_6_27B()
    dataset = TinyMathDataset()
    recipe = Qwen3_6_27b_Recipe(rm_type="deepscaler")

    assert isinstance(SlimeRecipe.get_base_recipe(model), Qwen3_6_27b_Recipe)
    assert recipe.total_nodes == 1

    nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(recipe, model)
    assert (nodes, nproc_per_node) == (1, 4)
    for expected in [
        "--tensor-model-parallel-size 4",
        "--pipeline-model-parallel-size 1",
        "--num-layers 64",
        "--hidden-size 5120",
        "--ffn-hidden-size 17408",
        "--num-attention-heads 24",
        "--num-query-groups 4",
        "--kv-channels 256",
        "--vocab-size 248320",
        "--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec",
        "--apply-layernorm-1p",
        "--use-gated-attention",
        "--attention-output-gate",
        "--rotary-base 10000000",
        "--rotary-percent 0.25",
    ]:
        assert expected in extra_args

    cmd = build_train_cmd(recipe, "/root/slime", model=model, dataset=dataset)
    for expected in [
        "python3 /root/slime/train.py",
        "--hf-checkpoint Qwen/Qwen3.6-27B",
        "--tensor-model-parallel-size 4",
        "--sequence-parallel",
        "--rollout-num-gpus-per-engine 4",
        "--rollout-batch-size 32",
        "--n-samples-per-prompt 8",
        "--rollout-max-response-len 8192",
        "--global-batch-size 256",
        "--max-tokens-per-gpu 8192",
        "--sglang-mem-fraction-static 0.75",
        "--lr 1e-06",
        "--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec",
        "--attention-output-gate",
        "--rotary-base 10000000",
        "--rotary-percent 0.25",
        "--prompt-data /data/TinyMathDataset/train.parquet",
        "--input-key messages",
        "--label-key label",
    ]:
        assert expected in cmd


def test_qwen3_6_27b_training_config_builds_slime_app() -> None:
    config = TrainConfig(
        model=Qwen3_6_27B(),
        dataset=TinyMathDataset(),
        recipe=Qwen3_6_27b_Recipe(rm_type="deepscaler"),
    )

    app = config._build_app()

    assert app.name == "slime-qwen3_6_27b_recipe"
    assert callable(app.train.get_raw_f())
    assert callable(app.download.get_raw_f())
    assert callable(app.convert_checkpoint.get_raw_f())


def test_qwen3_6_27b_singlenode_tutorial_uses_recipe_defaults() -> None:
    path = (
        Path(__file__).parents[1]
        / "tutorials"
        / "singlenode"
        / "001_qwen27b"
        / "001_qwen27b.py"
    )
    spec = importlib.util.spec_from_file_location("qwen3_6_27b_singlenode", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    recipe = Qwen3_6_27b_Recipe(rm_type="deepscaler")
    cli_args = recipe.cli_args(
        module.MathDataset(n_rows=120),
        Qwen3_6_27B(),
    )
    assert "--no-save-optim" not in cli_args
    assert "--no-save-rng" not in cli_args
    assert "--sglang-disable-custom-all-reduce" in cli_args
    source = path.read_text()
    assert 'Qwen3_6_27b_Recipe(rm_type="deepscaler")' in source
    assert "n_samples_per_prompt=4" not in source
    assert "eval_max_response_len=4096" not in source
    assert "tutorials/rl/005_qwen27b" not in source
