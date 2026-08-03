"""Gemma-4-26B-A4B model spec as a concrete HFModelConfiguration subclass."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .base import HFModelConfiguration, ModelArchitecture, parse_gemma4_response


class Gemma4_26B_A4B(HFModelConfiguration):
    """Gemma-4-26B-A4B-it (25.2B total, ~3.8B active) MoE model from Google.

    Mixture-of-Experts with 128 experts, 8 active per token plus 1 shared.
    Downloads from ``google/gemma-4-26B-A4B-it`` on HuggingFace.

    The checkpoint is a ``Gemma4ForConditionalGeneration``: the MoE decoder below plus a
    27-layer vision tower. ``vision=True`` drives the whole thing as a vision-language
    model (tower frozen by ``Gemma4_26B_A4B_Recipe``, rollouts served by SGLang's
    ``gemma4_mm``); the default builds a bare text-only ``GPTModel``.
    """

    response_parser = staticmethod(parse_gemma4_response)

    model_name = "google/gemma-4-26B-A4B-it"
    vision = False
    architecture = ModelArchitecture(
        # text_config from config.json
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
        swiglu=False,  # GeGLU, set by megatron_spec; see ModelArchitecture.swiglu
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
        megatron_spec=["slime_plugins.models.gemma4", "get_gemma4_spec"],
        use_rotary_position_embeddings=True,
        rotary_base=10000,
        rotary_percent=1.0,
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.vision:
            return
        if "architecture" not in kwargs:
            # ``--spec`` is the text-only path; bridge mode builds the vision tower +
            # projector + decoder from the HF config. The dimensions still drive slime's
            # arg validation and FLOPs accounting.
            self.architecture = replace(self.architecture, megatron_spec=None)
        # Same HF repo as text-only mode, so this variant needs its own catalog key.
        self.catalog_name = "google/gemma-4-26B-A4B-it-vl"
        # Gemma4VLModel.forward's dense mask is only valid on a single sequence.
        self.requires_bshd = True
