"""Qwen3-VL-8B-Instruct model spec as a concrete HFModelConfiguration subclass.

Qwen3-VL is a vision-language model served by SGLang on ``/v1/chat/completions``
with image content. Its text backbone is a dense Qwen3-8B decoder; the vision
tower (a ViT with patch size 16, depth 27) is loaded by SGLang straight from the
HF checkpoint. The architecture below (from ``config.json`` → ``text_config``)
drives Megatron *training*.

Megatron-bridge support: AutoBridge loads/trains the full VL model natively (no
ViT shim needed — verified empirically). The one gap is the MB→HF *export*:
slime's stock qwen3 converter can't map the vision tower. The
``patch_qwen3_vl_export`` shim ships a qwen3_vl MB→HF converter that converts the
trained language backbone and identity-passes the frozen ViT
(``vision_model.*`` → ``model.visual.*``), mirroring the qwen3_asr audio-tower
converter. Report upstream; drop once fixed there.
"""

from __future__ import annotations

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response


class Qwen3VL_8B(HFModelConfiguration):
    """Qwen3-VL-8B-Instruct (vision-language, 8B parameters) from Alibaba.

    Pre-configured with ``ModelArchitecture`` for the text backbone. The vision
    tower is frozen during RL training (``Qwen3VL_Recipe.freeze_params_name_list``)
    and handled by SGLang for rollouts.
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
        # slime's stock qwen3 export can't map the vision tower; ship a qwen3_vl
        # MB->HF converter that converts the language stack and identity-passes the
        # frozen ViT (vision_model.* -> model.visual.*), mirroring the qwen3_asr
        # audio-tower converter. The torch_dist patch additionally teaches the
        # deploy/eval torch_dist->HF tool to skip the frozen ViT's stacked layers
        # (whose depth != the LLM's num_layers) and refill them from the base HF
        # checkpoint, since slime's converter otherwise asserts on them.
        compat_patches=["patch_qwen3_vl_export", "patch_qwen3_vl_torch_dist"],
    )
