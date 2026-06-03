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
from pydantic import BaseModel

from modal_training_gym.utils.metadata import MetadataStore, vol_list

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

type FingerprintJson = (
    str | int | float | bool | None | list[FingerprintJson] | dict[str, FingerprintJson]
)


@dataclass(frozen=True, slots=True)
class GymObjectId:
    value: str
    config_fingerprint: str
    id_created_at: int


def to_fingerprint_value(value: object) -> FingerprintJson:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "class": type(value).__name__,
            **{
                field.name: to_fingerprint_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if not field.name.startswith("_")
            },
        }
    if isinstance(value, BaseModel):
        return {
            "class": type(value).__name__,
            **to_fingerprint_value(value.model_dump(mode="json")),
        }
    if isinstance(value, dict):
        return {
            str(k): to_fingerprint_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if not str(k).startswith("_") and not callable(v)
        }
    if isinstance(value, list | tuple | set):
        return [to_fingerprint_value(v) for v in value]
    if callable(value):
        return getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if hasattr(value, "__dict__"):
        return {
            "class": type(value).__name__,
            **{
                str(k): to_fingerprint_value(v)
                for k, v in sorted(vars(value).items(), key=lambda item: str(item[0]))
                if not str(k).startswith("_") and not callable(v)
            },
        }
    return repr(value)


def config_fingerprint(*parts: object) -> str:
    payload = json.dumps(
        to_fingerprint_value(parts),
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
    existing_ids = {
        str(record.get(id_key, "")) for record in vol_list(store)
    }

    for _ in range(64):
        run_created_at = id_created_at if id_created_at is not None else int(time.time())
        suffix = hashlib.sha256(f"{fingerprint}:{run_created_at}".encode()).hexdigest()[:5]
        word_slug = _NON_ALNUM_RE.sub("-", randomname.get_name(sep="-").lower()).strip("-")
        word_slug = re.sub(r"-{2,}", "-", word_slug)
        if word_slug.count("-") < 1:
            word_slug = f"run-{secrets.token_hex(3)}"
        identifier = f"{word_slug}-{suffix}"
        if identifier not in existing_ids:
            return GymObjectId(
                value=identifier,
                config_fingerprint=fingerprint,
                id_created_at=run_created_at,
            )
        id_created_at = None

    run_created_at = int(time.time())
    suffix = hashlib.sha256(f"{fingerprint}:{run_created_at}".encode()).hexdigest()[:5]
    word_slug = re.sub(
        r"-{2,}",
        "-",
        _NON_ALNUM_RE.sub("-", randomname.get_name(sep="-").lower()).strip("-"),
    )
    return GymObjectId(
        value=f"{word_slug}-{suffix}-{secrets.token_hex(2)}",
        config_fingerprint=fingerprint,
        id_created_at=run_created_at,
    )
