"""Regression tests for the dashboard's log-time-bound parser.

``_parse_log_time`` turns the ``since``/``until`` query params the log-stream
endpoint receives into epoch seconds. These pin down each accepted input
format against a fixed ``now`` so a refactor can't silently change behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modal_training_gym._dashboard import _parse_log_time

# Fixed reference point: 2026-07-09T18:00:00Z.
NOW = datetime(2026, 7, 9, 18, 0, 0, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_empty_or_blank_returns_none(value: str) -> None:
    assert _parse_log_time(value, NOW) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("45s", NOW - 45),
        ("30m", NOW - 30 * 60),
        ("2h", NOW - 2 * 3600),
        ("1d", NOW - 86400),
        ("0m", NOW),
        ("30 m", NOW - 30 * 60), 
    ],
)
def test_relative_age_subtracts_from_now(value: str, expected: float) -> None:
    assert _parse_log_time(value, NOW) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1720557600", 1720557600.0),
        ("1720557600.5", 1720557600.5),
        ("  1720557600  ", 1720557600.0),
    ],
)
def test_epoch_seconds(value: str, expected: float) -> None:
    assert _parse_log_time(value, NOW) == expected


def test_iso_with_trailing_z_is_utc() -> None:
    expected = datetime(2026, 7, 9, 18, 0, 0, tzinfo=timezone.utc).timestamp()
    assert _parse_log_time("2026-07-09T18:00:00Z", NOW) == expected


def test_iso_naive_is_read_as_utc() -> None:
    expected = datetime(2026, 7, 9, 18, 0, 0, tzinfo=timezone.utc).timestamp()
    assert _parse_log_time("2026-07-09T18:00:00", NOW) == expected


def test_iso_with_explicit_offset_is_honored() -> None:
    # 18:00 at -04:00 is 22:00 UTC.
    expected = datetime(2026, 7, 9, 22, 0, 0, tzinfo=timezone.utc).timestamp()
    assert _parse_log_time("2026-07-09T18:00:00-04:00", NOW) == expected


@pytest.mark.parametrize(
    "value",
    [
        "not-a-time",
        "30x",  # unknown unit
        "2026-13-01T00:00:00Z",  # invalid month
        "yesterday",
        "-5m",  # leading sign isn't part of the relative grammar
    ],
)
def test_unparseable_returns_none(value: str) -> None:
    assert _parse_log_time(value, NOW) is None
