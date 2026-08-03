"""Gemma-4-26B-A4B-it as a vision-language model spec.

``google/gemma-4-26B-A4B-it`` is a ``Gemma4ForConditionalGeneration``: a 26B-A4B MoE
text decoder plus a 27-layer vision tower. This config builds the whole thing; the
sibling :class:`~.gemma4_26b_a4b.Gemma4_26B_A4B` hands the same checkpoint to slime's
text-only model script, which builds a bare ``GPTModel`` and no vision tower.

The class holds only the model's specs. Framework wiring (bridge mode, the frozen
vision tower, the packed-sequence shim) lives on ``Gemma4_26B_A4B_VL_Recipe``, since
it's meaningless for other backends.
"""

from __future__ import annotations

from .base import HFModelConfiguration, ModelArchitecture, parse_gemma4_response


class Gemma4_26B_A4B_VL(HFModelConfiguration):
    """Gemma-4-26B-A4B-it driven as a vision-language model (image + text → text).

    The vision tower is frozen during RL training
    (``Gemma4_26B_A4B_VL_Recipe.freeze_params_name_list``) and served by SGLang's
    ``gemma4_mm`` for rollouts.
    """

    model_name = "google/gemma-4-26B-A4B-it"
    # Same HF repo as the text-only Gemma4_26B_A4B, so this variant needs its own key.
    catalog_name = "google/gemma-4-26B-A4B-it-vl"
    response_parser = staticmethod(parse_gemma4_response)

    # ``Gemma4VLModel.forward`` builds a dense mask (bidirectional within each image
    # block) that is only valid on a single sequence, and image-token counts vary with
    # aspect ratio so a ragged micro-batch cannot be reconciled.
    requires_bshd = True

    architecture = ModelArchitecture(
        # text_config from config.json. Bridge mode builds the provider from the HF
        # config, so these drive slime's arg validation and FLOPs accounting rather
        # than the model. megatron_spec is omitted: --spec is the text-only path.
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
        swiglu=False,  # GeGLU, set by the bridge provider; see ModelArchitecture.swiglu
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
        rotary_base=10000,
        rotary_percent=1.0,
    )
