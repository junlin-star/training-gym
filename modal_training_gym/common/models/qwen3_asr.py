"""Qwen3-ASR-1.7B as a gym model.

Qwen3-ASR is an audio-only ASR model served by SGLang on
``/v1/audio/transcriptions``. Its text backbone is a dense Qwen3-1.7B; the audio
tower (a Qwen3-Omni-style Whisper encoder) is loaded by Megatron-Bridge straight
from the HF checkpoint and isn't expressible in ``ModelArchitecture`` (which is
LLM-backbone-only) — same situation as Qwen2.5-Omni.

For serving (rollouts/eval) only ``model_name`` matters — SGLang reads the HF
config directly. The architecture below (verbatim from the HF config.json
``thinker_config.text_config``) drives Megatron *training*.

``compat_patches`` lists the image-build shims the gym applies for this model:
upstream gaps on the native stack (bridge config validate-order, slime processor
loading, bridge pg_collection, MB→HF export of the audio tower). Each should be
reported upstream; once fixed there, the corresponding patch can be dropped.
"""

from __future__ import annotations

from .base import HFModelConfiguration, ModelArchitecture


class Qwen3ASR(HFModelConfiguration):
    model_name = "Qwen/Qwen3-ASR-1.7B"

    # The native megatron-bridge Qwen3-ASR forward doesn't implement THD sequence
    # packing, so training must use padded (bshd) batches. The launcher enforces
    # this (raises if the recipe leaves slime's default thd packing on).
    requires_bshd = True

    # Runtime deps the gym installs into the image when this model is used, so the
    # example stays reward-only: soundfile/librosa decode audio (dataset prep +
    # transcription rollout); jiwer powers the standard −WER reward. The base slime
    # image already ships `datasets`.
    pip_packages = ["jiwer", "librosa", "soundfile"]

    # thinker_config.text_config (Qwen3 dense backbone), verbatim from config.json.
    architecture = ModelArchitecture(
        num_layers=28,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,          # num_key_value_heads
        kv_channels=128,             # head_dim (explicit; != hidden/heads in general)
        vocab_size=151936,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,                 # hidden_act = silu
        disable_bias_linear=True,    # Qwen3 dropped qkv bias
        qk_layernorm=True,           # Qwen3 family adds qk-layernorm
        use_rotary_position_embeddings=True,
        rotary_base=1000000,         # rope_theta
        compat_patches=[
            "patch_qwen3asr_bridge_config",
            "patch_qwen3asr_processor",
            "patch_qwen3asr_pg_collection",
            "patch_qwen3asr_export",
        ],
    )
