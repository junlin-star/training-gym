from __future__ import annotations

from .kimi_k2_5 import Kimi_K2_5


class Kimi_K2_6(Kimi_K2_5):
    """Kimi-K2.6 model preset using the Kimi-K2.5 architecture path."""

    model_name = "moonshotai/Kimi-K2.6"
    model_path = "/checkpoints/Kimi-K2.6-bf16"
    int4_model_path = "/checkpoints/Kimi-K2.6-int4"
