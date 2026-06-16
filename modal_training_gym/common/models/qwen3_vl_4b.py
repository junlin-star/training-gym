"""Qwen3-VL-4B-Instruct as a gym model.

Qwen3-VL pairs a Qwen3 text backbone with a SigLIP-2 vision encoder. The
``architecture`` is the **text backbone**, which vime's ``hf_validate_args``
checks field-by-field against the HF config (it requires these flags and rejects
mismatches). Qwen3-VL-4B's backbone equals Qwen3-4B *except* ``rope_theta``
(5e6, not 1e6). The vision tower is read from the HF checkpoint by vime's
Qwen3-VL bridge. Images reach the rollout via a
``MultimodalDataset(modality="image")``'s ``multimodal_keys``; vime's default
vLLM rollout forwards them as image content.

De-risk stepping stone for Qwen3-Omni (this + an audio tower).
"""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response


class Qwen3_VL_4B(HFModelConfiguration):
    model_name = "Qwen/Qwen3-VL-4B-Instruct"

    response_parser = staticmethod(parse_qwen3_response)

    # Text backbone; values verified against vime's hf_validate_args (== Qwen3-4B
    # but rope_theta=5e6).
    architecture = ModelArchitecture(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
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
        use_rotary_position_embeddings=True,
        rotary_base=5000000,
    )
