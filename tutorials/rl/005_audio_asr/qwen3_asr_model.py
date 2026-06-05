"""Qwen3-ASR-1.7B as a gym ModelConfig.

Qwen3-ASR is an audio-only ASR model (served by SGLang on
/v1/audio/transcriptions). Its text backbone is a dense Qwen3-1.7B; the audio
tower (a Qwen3-Omni-style Whisper encoder, 24 layers) is loaded by
Megatron-Bridge straight from the HF checkpoint and isn't expressible in the
gym's ModelArchitecture (LLM-backbone-only) — same situation as Qwen2.5-Omni.

For SERVING (rollouts/eval) only `model_name` matters — SGLang reads the HF
config directly. The architecture below (verbatim from the HF config.json
`thinker_config.text_config`) is for Megatron *training*.
"""

from __future__ import annotations

from modal_training_gym import HFModelConfiguration, ModelArchitecture


class Qwen3ASR(HFModelConfiguration):
    model_name = "Qwen/Qwen3-ASR-1.7B"

    # thinker_config.text_config (Qwen3 dense backbone), verbatim from config.json.
    architecture = ModelArchitecture(
        num_layers=28,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,         # num_key_value_heads
        kv_channels=128,            # head_dim (explicit; != hidden/heads in general)
        vocab_size=151936,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,                # hidden_act = silu
        disable_bias_linear=True,   # Qwen3 dropped qkv bias (no add_qkv_bias needed)
        qk_layernorm=True,          # Qwen3 family adds qk-layernorm
        use_rotary_position_embeddings=True,
        rotary_base=1000000,        # rope_theta
    )
