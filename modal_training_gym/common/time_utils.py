"""Shared parsing helpers for user-supplied time bounds."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime


_RELATIVE_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
}


def parse_time(value: str, now: float) -> float | None:
    """Parse relative age, epoch seconds, or ISO 8601 into epoch seconds."""
    text = (value or "").strip()
    if not text:
        return None

    relative = re.fullmatch(r"(\d+)\s*([smhdw])", text, re.IGNORECASE)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).lower()
        return now - amount * _RELATIVE_SECONDS[unit]

    try:
        parsed_number = float(text)
    except ValueError:
        pass
    else:
        return parsed_number if math.isfinite(parsed_number) else None

    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed_datetime = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(tzinfo=UTC)
    return parsed_datetime.timestamp()
