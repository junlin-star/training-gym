"""Unit tests for the Vime (slime + vLLM) recipe.

Kept deliberately small — the other framework recipes carry no dedicated recipe
tests, so this only pins the behavior that is *new* to vime and easy to get
wrong: the vLLM/router CLI surface (not SGLang) and the fresh-bridge load
contract.
"""

from __future__ import annotations

from modal_training_gym.common.models import ModelArchitecture, ModelConfig
from modal_training_gym.train_recipes.base import RecipeType
from modal_training_gym.train_recipes.vime_recipe import Qwen3_4b_Vime_Recipe


def _model() -> ModelConfig:
    return ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        architecture=ModelArchitecture(
            num_layers=28,
            hidden_size=1024,
            ffn_hidden_size=3072,
            num_attention_heads=16,
            group_query_attention=True,
            num_query_groups=8,
            kv_channels=128,
            vocab_size=151936,
        ),
    )


def test_recipe_type_is_vime():
    assert Qwen3_4b_Vime_Recipe().recipe_type is RecipeType.VIME


def test_cli_args_use_vllm_router_not_sglang():
    """The whole point of vime: vLLM/router flags, never SGLang flags."""
    args = Qwen3_4b_Vime_Recipe().cli_args()
    assert "--vllm-gpu-memory-utilization" in args
    assert "--vllm-router-policy" in args
    assert not any("sglang" in a for a in args)


def test_fresh_bridge_run_omits_load_flags():
    """Bridge-mode contract: with empty load/ref_load, no --load/--ref-load is
    emitted, so vime falls back to ``--hf-checkpoint`` and bridge-loads the HF
    snapshot. Emitting an empty --load would break that fallback.
    """
    args = Qwen3_4b_Vime_Recipe().cli_args(model=_model())
    assert "--load" not in args
    assert "--ref-load" not in args
    assert "--hf-checkpoint" in args
    assert args[args.index("--megatron-to-hf-mode") + 1] == "bridge"
