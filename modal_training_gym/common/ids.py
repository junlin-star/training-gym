from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass
from enum import Enum

import randomname

from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.deploy_recipes.base import BaseDeployRecipe
from modal_training_gym.train_recipes.base import BaseTrainRecipe
from modal_training_gym.utils.metadata import MetadataStore, vol_list

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

type FingerprintJson = (
    str | int | float | bool | None | list[FingerprintJson] | dict[str, FingerprintJson]
)

type RunConfigPart = (
    str
    | int
    | float
    | bool
    | None
    | Enum
    | ModelConfig
    | DatasetConfig
    | Checkpoint
    | BaseTrainRecipe
    | BaseDeployRecipe
)


@dataclass(frozen=True, slots=True)
class GymObjectId:
    value: str
    config_fingerprint: str
    id_created_at: int


def config_fingerprint(*parts: RunConfigPart) -> str:
    """12-char hash of train/deploy/eval setup (model, dataset, recipe, tags, …)."""
    payload = json.dumps(
        [_serialize_config_part(part) for part in parts],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def stable_readable_id(
    store: MetadataStore,
    fingerprint: str,
    *,
    id_key: str,
    id_created_at: int | None = None,
) -> GymObjectId:
    """Mint a unique ``word-word-hash`` id for one run record in the metadata store."""
    existing_ids = {str(record.get(id_key, "")) for record in vol_list(store)}

    for _ in range(64):
        run_created_at = (
            id_created_at if id_created_at is not None else int(time.time())
        )
        suffix = hashlib.sha256(f"{fingerprint}:{run_created_at}".encode()).hexdigest()[
            :5
        ]
        identifier = f"{_random_word_slug()}-{suffix}"
        if identifier not in existing_ids:
            return GymObjectId(
                value=identifier,
                config_fingerprint=fingerprint,
                id_created_at=run_created_at,
            )
        id_created_at = None

    run_created_at = int(time.time())
    suffix = hashlib.sha256(f"{fingerprint}:{run_created_at}".encode()).hexdigest()[:5]
    return GymObjectId(
        value=f"{_random_word_slug()}-{suffix}-{secrets.token_hex(2)}",
        config_fingerprint=fingerprint,
        id_created_at=run_created_at,
    )


def _random_word_slug() -> str:
    slug = _NON_ALNUM_RE.sub("-", randomname.get_name(sep="-").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if slug.count("-") < 1:
        return f"run-{secrets.token_hex(3)}"
    return slug


def _serialize_config_part(part: RunConfigPart) -> FingerprintJson:
    if part is None or isinstance(part, str | int | float | bool):
        return part
    if isinstance(part, Enum):
        return part.value
    if isinstance(part, ModelConfig | DatasetConfig):
        return _config_dict(type(part).__name__, _public_vars(part))
    if dataclasses.is_dataclass(part) and not isinstance(part, type):
        return _config_dict(
            type(part).__name__,
            {
                field.name: _serialize_nested(getattr(part, field.name))
                for field in dataclasses.fields(part)
                if not field.name.startswith("_")
                and not callable(getattr(part, field.name))
            },
        )
    raise TypeError(
        f"Unsupported config part for fingerprinting: {type(part).__name__}"
    )


def _config_dict(
    class_name: str, fields: dict[str, FingerprintJson]
) -> dict[str, FingerprintJson]:
    return {"class": class_name, **fields}


def _public_vars(instance: ModelConfig | DatasetConfig) -> dict[str, FingerprintJson]:
    return {
        str(key): _serialize_nested(value)
        for key, value in sorted(vars(instance).items(), key=lambda item: str(item[0]))
        if not str(key).startswith("_") and not callable(value)
    }


def _serialize_nested(value: object) -> FingerprintJson:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _config_dict(
            type(value).__name__,
            {
                field.name: _serialize_nested(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if not field.name.startswith("_")
                and not callable(getattr(value, field.name))
            },
        )
    if isinstance(value, dict):
        return {
            str(key): _serialize_nested(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if not callable(item)
        }
    if isinstance(value, list | tuple | set):
        return [_serialize_nested(item) for item in value]
    raise TypeError(
        f"Unsupported nested config field type for fingerprinting: {type(value).__name__}"
    )
