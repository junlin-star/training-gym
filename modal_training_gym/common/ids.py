from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import secrets
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from modal_training_gym.utils.metadata import MetadataStore, vol_list

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    slug = _NON_ALNUM_RE.sub("-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "class": type(value).__name__,
            **{
                field.name: canonicalize(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if not field.name.startswith("_")
            },
        }
    if isinstance(value, BaseModel):
        return {
            "class": type(value).__name__,
            **canonicalize(value.model_dump(mode="json")),
        }
    if isinstance(value, dict):
        return {
            str(k): canonicalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if not str(k).startswith("_") and not callable(v)
        }
    if isinstance(value, list | tuple | set):
        return [canonicalize(v) for v in value]
    if callable(value):
        return getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if hasattr(value, "__dict__"):
        return {
            "class": type(value).__name__,
            **{
                str(k): canonicalize(v)
                for k, v in sorted(vars(value).items(), key=lambda item: str(item[0]))
                if not str(k).startswith("_") and not callable(v)
            },
        }
    return repr(value)


def config_fingerprint(*parts: Any) -> str:
    try:
        payload = json.dumps(canonicalize(parts), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = repr(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def id_hash(fingerprint: str, attempt_index: int, *, length: int = 5) -> str:
    payload = f"{fingerprint}:{attempt_index}"
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def random_word_slug() -> str:
    import randomname

    slug = slugify(randomname.get_name(sep="-"))
    if slug.count("-") >= 1:
        return slug
    return f"run-{secrets.token_hex(3)}"


def _metadata_records(store: MetadataStore) -> list[dict[str, Any]]:
    try:
        return vol_list(store)
    except Exception:
        return []


def _metadata_value(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    for nested_key in ("metadata", "extra"):
        nested = record.get(nested_key)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return None


def next_attempt_index(store: MetadataStore, fingerprint: str) -> int:
    attempts: list[int] = []
    for record in _metadata_records(store):
        if _metadata_value(record, "config_fingerprint") != fingerprint:
            continue
        attempt = _metadata_value(record, "attempt_index")
        if isinstance(attempt, int):
            attempts.append(attempt)
        elif isinstance(attempt, str) and attempt.isdigit():
            attempts.append(int(attempt))
    return (max(attempts) + 1) if attempts else 1


def unique_readable_id(
    store: MetadataStore,
    fingerprint: str,
    *,
    id_key: str,
    attempt_index: int | None = None,
) -> tuple[str, int]:
    attempt = attempt_index or next_attempt_index(store, fingerprint)
    suffix = id_hash(fingerprint, attempt)
    existing_ids = {str(record.get(id_key, "")) for record in _metadata_records(store)}

    for _ in range(64):
        identifier = f"{random_word_slug()}-{suffix}"
        if identifier not in existing_ids:
            return identifier, attempt

    return f"{random_word_slug()}-{suffix}-{secrets.token_hex(2)}", attempt
