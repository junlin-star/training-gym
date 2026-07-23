"""Gemma-4-26B-A4B model spec as a concrete HFModelConfiguration subclass."""

from __future__ import annotations

from .base import HFModelConfiguration, ModelArchitecture


class Gemma4_26B_A4B(HFModelConfiguration):
    """Gemma-4-26B-A4B-it (25.2B total, ~3.8B active) MoE model from Google.

    Mixture-of-Experts with 128 experts, 8 active per token plus 1 shared.
    Downloads from ``google/gemma-4-26B-A4B-it`` on HuggingFace.
    """

    model_name = "google/gemma-4-26B-A4B-it"

    # text_config from scripts/models/gemma4-26B-A4B.sh (informational — the
    # model script drives training flags; see module docstring).
    architecture = ModelArchitecture(
        num_layers=30,
        hidden_size=2816,
        ffn_hidden_size=2112,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
        kv_channels=256,
        vocab_size=262144,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=False,
        num_experts=128,
        moe_ffn_hidden_size=704,
        moe_grouped_gemm=True,
        moe_router_topk=8,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.0,
        megatron_spec=["slime_plugins.models.gemma4", "get_gemma4_spec"],
        use_rotary_position_embeddings=True,
        rotary_base=10000,
        rotary_percent=1.0,
    )
