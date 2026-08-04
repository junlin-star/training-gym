"""Model configs supported by the CI validation run."""

from .base import ModelConfig
from .gemma4_26b_a4b import Gemma4_26B_A4B
from .qwen3_0_6b import Qwen3_0_6B
from .qwen3_1_7b import Qwen3_1_7B
from .qwen3_4b import Qwen3_4B
from .qwen3_6_35b import Qwen3_6_35B
from .qwen3_8b import Qwen3_8B
from .qwen3_asr_1_7b import Qwen3_ASR_1_7B
from .qwen3_vl_8b import Qwen3_VL_8B

VALIDATABLE_MODELS: tuple[tuple[str, type[ModelConfig]], ...] = (
    ("gemma-4-26B-A4B-it", Gemma4_26B_A4B),
    ("Qwen3-0.6B", Qwen3_0_6B),
    ("Qwen3-1.7B", Qwen3_1_7B),
    ("Qwen3-4B", Qwen3_4B),
    ("Qwen3-8B", Qwen3_8B),
    ("Qwen3-ASR-1.7B", Qwen3_ASR_1_7B),
    ("Qwen3-VL-8B-Instruct", Qwen3_VL_8B),
    ("Qwen3.6-35B-A3B", Qwen3_6_35B),
)
