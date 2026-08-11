"""Model configs supported by the CI validation run."""

from .base import ModelConfig
from .qwen3_0_6b import Qwen3_0_6B
from .qwen3_1_7b import Qwen3_1_7B
from .qwen3_4b import Qwen3_4B
from .qwen3_5_0_8b import Qwen3_5_0_8B
from .qwen3_5_2b import Qwen3_5_2B
from .qwen3_5_4b import Qwen3_5_4B
from .qwen3_5_9b import Qwen3_5_9B
from .qwen3_6_35b import Qwen3_6_35B
from .qwen3_8b import Qwen3_8B
from .qwen3_asr_1_7b import Qwen3_ASR_1_7B
from .qwen3_vl_8b import Qwen3_VL_8B

VALIDATABLE_MODELS: tuple[tuple[str, type[ModelConfig]], ...] = (
    ("Qwen3-0.6B", Qwen3_0_6B),
    ("Qwen3-1.7B", Qwen3_1_7B),
    ("Qwen3-4B", Qwen3_4B),
    ("Qwen3-8B", Qwen3_8B),
    ("Qwen3-ASR-1.7B", Qwen3_ASR_1_7B),
    ("Qwen3-VL-8B-Instruct", Qwen3_VL_8B),
    ("Qwen3.5-0.8B", Qwen3_5_0_8B),
    ("Qwen3.5-2B", Qwen3_5_2B),
    ("Qwen3.5-4B", Qwen3_5_4B),
    ("Qwen3.5-9B", Qwen3_5_9B),
    ("Qwen3.6-35B-A3B", Qwen3_6_35B),
)
