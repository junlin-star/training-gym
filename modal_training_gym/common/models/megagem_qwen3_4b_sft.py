"""MegaGem SFT checkpoint on the Qwen3-4B architecture."""

from .base import ModelArchitecture, parse_qwen3_response
from .qwen3_4b import Qwen3_4B


MEGAGEM_QWEN3_4B_SFT_MODEL = "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2"


class MegaGem_Qwen3_4B_SFT(Qwen3_4B):
    """Private MegaGem SFT checkpoint used as the Phase-3 RL starting policy."""

    response_parser = staticmethod(parse_qwen3_response)

    model_name = MEGAGEM_QWEN3_4B_SFT_MODEL
    architecture = ModelArchitecture(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
        num_attention_heads=32,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=151936,
        rotary_base=5000000,
    )
