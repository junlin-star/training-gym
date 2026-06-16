"""Qwen3-Omni-30B-A3B-Instruct as a gym model.

Qwen3-Omni is a Thinker-Talker omni-modal MoE: a Qwen3-30B-A3B text MoE backbone
with SigLIP-2 vision + AuT audio encoders (the Thinker, which perceives
audio/image/video/text and emits text — what GRPO trains; the Talker/code2wav
speech path is unused here). We GRPO-train the Thinker.

``architecture`` is the Thinker's **text backbone**, which vime's
``hf_validate_args`` checks field-by-field against the HF config; the MoE expert
config and the vision/audio towers are read from the HF checkpoint by megatron's
``qwen_omni`` bridge (bridge mode). Note this backbone differs from the plain
Qwen3-30B-A3B (``vocab_size=152064``, ``intermediate_size=768``).
"""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response


class Qwen3_Omni_30B(HFModelConfiguration):
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    response_parser = staticmethod(parse_qwen3_response)

    # Thinker text backbone (Qwen3-Omni-30B-A3B), verified against the HF config.
    architecture = ModelArchitecture(
        num_layers=48,
        hidden_size=2048,
        ffn_hidden_size=768,  # == intermediate_size (all-MoE)
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=4,
        kv_channels=128,
        vocab_size=152064,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        use_rotary_position_embeddings=True,
        rotary_base=1000000,
        # MoE (128 experts, top-8). Required on the CLI for expert parallelism —
        # not auto-read from the HF config. Values from the omni config + vime's
        # proven Qwen3-30B-A3B model script.
        num_experts=128,
        moe_ffn_hidden_size=768,
        moe_router_topk=8,
        moe_router_score_function="softmax",
        moe_token_drop_policy="probs",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
    )
