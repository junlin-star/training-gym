"""Gemma-4-26B-A4B model spec as a concrete HFModelConfiguration subclass."""

from __future__ import annotations

from .base import HFModelConfiguration, ModelArchitecture, parse_gemma4_response


class Gemma4_26B_A4B(HFModelConfiguration):
    """Gemma-4-26B-A4B-it (25.2B total, ~3.8B active) MoE model from Google.

    Mixture-of-Experts with 128 experts, 8 active per token plus 1 shared.
    Downloads from ``google/gemma-4-26B-A4B-it`` on HuggingFace.

    The checkpoint is a ``Gemma4ForConditionalGeneration``: the MoE decoder described
    below plus a 27-layer vision tower. ``Gemma4_26B_A4B_Recipe`` trains it through
    the HF<->Megatron bridge on either text or image data; the architecture here
    describes the decoder either way.
    """

    model_name = "google/gemma-4-26B-A4B-it"
    response_parser = staticmethod(parse_gemma4_response)

    architecture = ModelArchitecture(
        # text_config from config.json. Informational: the recipe sets
        # ``miles_model_script``, so miles takes the real values from
        # ``scripts/models/gemma-4-26b-a4b-it.sh`` and none of these are emitted as
        # flags. They still drive the num_experts/expert-parallel validator and
        # FLOPs accounting, so keep them in step with that script.
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
        swiglu=False,  # GeGLU, set by the layer spec; see ModelArchitecture.swiglu
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
        use_rotary_position_embeddings=True,
        # Gemma-4 nests rope_theta per attention type; the miles model script takes
        # the global-attention value.
        rotary_base=1000000,
        rotary_percent=1.0,
    )
