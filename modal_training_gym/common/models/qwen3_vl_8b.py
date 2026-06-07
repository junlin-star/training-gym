"""Qwen3-VL-8B-Instruct model spec as a concrete HFModelConfiguration subclass.

Qwen3-VL is a vision-language model served by SGLang on ``/v1/chat/completions``
with image content. Its text backbone is a dense Qwen3-8B decoder; the vision
tower (a ViT with patch size 16, depth 27) is loaded by SGLang straight from the
HF checkpoint. The architecture below (from ``config.json`` → ``text_config``)
drives Megatron *training*.

Note: compat patches for the vision tower (bridge config, export converter, etc.)
will likely be needed once megatron-bridge support is verified — similar to the
Qwen3-ASR audio tower patches.
"""

from __future__ import annotations

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response


class Qwen3VL_8B(HFModelConfiguration):
    """Qwen3-VL-8B-Instruct (vision-language, 8B parameters) from Alibaba.

    Pre-configured with ``ModelArchitecture`` for the text backbone. The vision
    tower is frozen during RL training and handled by SGLang for rollouts.
    """

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    # The vision encoder makes prompts much longer (image patches expand into
    # many tokens), so padded (bshd) batches avoid the THD packing path that
    # VL models may not support in megatron-bridge.
    requires_bshd = True

    architecture = ModelArchitecture(
        # text_config from config.json
        num_layers=36,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=151936,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        use_rotary_position_embeddings=True,
        rotary_base=5000000,
    )
