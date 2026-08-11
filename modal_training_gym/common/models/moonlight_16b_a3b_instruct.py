"""Moonlight-16B-A3B-Instruct model configuration."""

from .base import HFModelConfiguration, ModelArchitecture


class Moonlight_16B_A3B_Instruct(HFModelConfiguration):
    """Moonshot AI's 16B-total, 3B-active Moonlight MoE instruct model."""

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    architecture = ModelArchitecture(
        num_layers=27,
        hidden_size=2048,
        ffn_hidden_size=11264,
        num_attention_heads=16,
        group_query_attention=False,
        kv_channels=128,
        vocab_size=163840,
        normalization="RMSNorm",
        norm_epsilon=1e-5,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        num_experts=64,
        moe_ffn_hidden_size=1408,
        moe_shared_expert_intermediate_size=2816,
        moe_grouped_gemm=True,
        moe_router_topk=6,
        moe_router_score_function="sigmoid",
        moe_token_drop_policy="probs",
        moe_router_dtype="fp32",
        moe_permute_fusion=True,
        moe_aux_loss_coeff=0.0,
        megatron_model_type="moonlight",
        use_rotary_position_embeddings=True,
        rotary_base=50000,
    )
