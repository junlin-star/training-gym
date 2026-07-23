"""Gemma-4-26B-A4B model spec as a concrete HFModelConfiguration subclass.

Gemma 4 26B-A4B is a natively multimodal (text + image) Mixture-of-Experts model
from Google: 25.2B total parameters with ~3.8B active per token (128 experts, 8
routed + 1 shared). The text backbone is a 30-layer decoder with a 5-local /
1-global sliding-attention pattern; a ~550M ViT vision tower is loaded by SGLang
straight from the HF checkpoint for rollouts.

The architecture below mirrors the *text_config* used by slime's upstream
``scripts/models/gemma4-26B-A4B.sh``. Because that model script relies on a
custom Megatron spec + model provider (``--spec ...`` and
``--custom-model-provider-path ...``) that ``ModelArchitecture`` can't express,
``Gemma4_26B_A4B_Recipe`` selects it via ``slime_model_script``; the arch below
is therefore informational (and drives the num_experts ÷ EP validator) rather
than the source of the training CLI flags.
"""

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
