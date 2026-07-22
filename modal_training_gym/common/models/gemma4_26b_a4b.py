"""Gemma-4-26B-A4B (MoE) model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture


class Gemma4_26B_A4B(HFModelConfiguration):
    """Gemma-4-26B-A4B (26B total, ~4B active) MoE model from Google.

    Mixture-of-Experts with 128 routed experts, 8 active per token.
    slime consumes this model through its upstream model script
    (``scripts/models/gemma4-26B-A4B.sh``), which sets a custom
    ``--spec``/``--custom-model-provider-path`` that cannot be expressed as
    plain ``ModelArchitecture`` CLI flags. The ``architecture`` below is
    therefore *informational*: the recipe pins ``slime_model_script`` so the
    launcher sources the upstream args verbatim and skips
    ``_model_to_fields``. The fields are still read by the
    ``num_experts % expert_model_parallel_size`` validator.

    Downloads from ``google/gemma-4-26b-a4b`` on HuggingFace (gated — needs
    ``HF_TOKEN`` with license acceptance). The default snapshot download is
    sufficient; no post-processing required.
    """

    model_name = "google/gemma-4-26b-a4b"
    architecture = ModelArchitecture(
        # Mirrors scripts/models/gemma4-26B-A4B.sh (informational under
        # slime_model_script mode).
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
        # Gemma uses GeGLU (gelu-tanh), not SwiGLU, and ties input/output
        # embeddings.
        swiglu=False,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=False,
        use_rotary_position_embeddings=True,
        rotary_base=10000,
        rotary_percent=1.0,
        num_experts=128,
        moe_ffn_hidden_size=704,
        moe_grouped_gemm=True,
        moe_router_topk=8,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.0,
    )
