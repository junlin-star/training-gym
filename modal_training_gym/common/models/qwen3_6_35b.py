"""Qwen3.6-35B-A3B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture


class Qwen3_6_35B(HFModelConfiguration):
    """Qwen3.6-35B-A3B (35B total, ~3B active) MoE model from Alibaba.

    Mixture-of-Experts with 256 experts, 8 active per token.
    Pre-configured with base ``ModelArchitecture`` for Megatron-based
    frameworks (slime). Downloads from ``Qwen/Qwen3.6-35B-A3B`` on HuggingFace.
    """

    model_name = "Qwen/Qwen3.6-35B-A3B"
    architecture = ModelArchitecture(
        num_layers=40,
        hidden_size=2048,
        ffn_hidden_size=512,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=2,
        kv_channels=256,
        vocab_size=248320,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        use_rotary_position_embeddings=True,
        rotary_base=10000000,
    )
