from __future__ import annotations

from enum import Enum
from typing import TypeAlias


class SlimeStatus(str, Enum):
    INITIALIZING = "initializing"
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL = "convert_model"
    PREPARE_DATASET = "prepare_dataset"
    TRAINING = "training"


class MilesStatus(str, Enum):
    INITIALIZING = "initializing"
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL = "convert_model"
    PREPARE_DATASET = "prepare_dataset"
    TRAINING = "training"


FrameworkStatus: TypeAlias = SlimeStatus | MilesStatus
