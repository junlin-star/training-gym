"""Unit tests for the dashboard's ``get_run_logs`` helper functions.

These check the pure logic extracted from the historical-log route.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from modal_training_gym._dashboard import (
    _compute_next_page,
    _parse_log_batches,
    _parse_log_time,
    _resolve_log_window,
    _to_timestamp,
)

NOW = 1_720_557_600.0


def _batch(task_id: str, items: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, items=items)


def _item(
    data: str,
    *,
    file_descriptor: int = 0,
    timestamp: float = 0.0,
    timestamp_ns: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        file_descriptor=file_descriptor,
        timestamp=timestamp,
        timestamp_ns=timestamp_ns,
    )


# ── _resolve_log_window ──────────────────────────────────────────────────


def test_window_defaults_when_bounds_unset() -> None:
    since_ts, until_ts = _resolve_log_window(
        None,
        None,
        default_since=100.0,
        default_until=900.0,
        now=NOW,
    )
    assert since_ts == 100.0
    assert until_ts == 900.0


def test_window_until_defaults_to_now_when_default_nonpositive() -> None:
    _, until_ts = _resolve_log_window(
        None,
        None,
        default_since=100.0,
        default_until=0.0,
        now=NOW,
    )
    assert until_ts == NOW


def test_window_nonpositive_explicit_until_becomes_now() -> None:
    _, until_ts = _resolve_log_window(
        1000.0,
        -5.0,
        default_since=100.0,
        default_until=900.0,
        now=NOW,
    )
    assert until_ts == NOW


def test_window_negative_since_is_floored_at_zero() -> None:
    since_ts, _ = _resolve_log_window(
        -30.0,
        1000.0,
        default_since=100.0,
        default_until=900.0,
        now=NOW,
    )
    assert since_ts == 0.0


def test_window_explicit_bounds_pass_through() -> None:
    since_ts, until_ts = _resolve_log_window(
        200.0,
        1000.0,
        default_since=100.0,
        default_until=900.0,
        now=NOW,
    )
    assert since_ts == 200.0
    assert until_ts == 1000.0


# ── _to_timestamp ────────────────────────────────────────────────────────


def test_to_timestamp_whole_seconds() -> None:
    ts = _to_timestamp(1_720_557_600.0)
    assert ts.seconds == 1_720_557_600
    assert ts.nanos == 0


def test_to_timestamp_fractional_seconds() -> None:
    ts = _to_timestamp(1_720_557_600.5)
    assert ts.seconds == 1_720_557_600
    assert ts.nanos == 500_000_000


# ── _parse_log_batches ───────────────────────────────────────────────────


def test_parse_skips_empty_data_and_sets_core_fields() -> None:
    batches = [
        _batch(
            "task-a",
            [
                _item("hello", file_descriptor=1),
                _item(""),  # dropped
            ],
        )
    ]
    logs = _parse_log_batches(batches)
    assert logs == [{"task_id": "task-a", "line": "hello", "fd": 1}]


def test_parse_attaches_timestamps_only_when_present() -> None:
    batches = [
        _batch(
            "task-a",
            [
                _item("with-ts", timestamp=12.5, timestamp_ns=12_500_000_000),
                _item("no-ts"),
            ],
        )
    ]
    logs = _parse_log_batches(batches)
    assert logs[0] == {
        "task_id": "task-a",
        "line": "with-ts",
        "fd": 0,
        "ts": 12.5,
        "ts_ns": 12_500_000_000,
    }
    assert logs[1] == {"task_id": "task-a", "line": "no-ts", "fd": 0}
    assert "ts" not in logs[1]
    assert "ts_ns" not in logs[1]


def test_parse_flattens_multiple_batches_in_order() -> None:
    batches = [
        _batch("task-a", [_item("a1"), _item("a2")]),
        _batch("task-b", [_item("b1")]),
    ]
    logs = _parse_log_batches(batches)
    assert [e["line"] for e in logs] == ["a1", "a2", "b1"]
    assert [e["task_id"] for e in logs] == ["task-a", "task-a", "task-b"]


# ── _compute_next_page ───────────────────────────────────────────────────


def test_next_page_partial_page_has_no_more() -> None:
    logs = [{"task_id": "t", "line": "x", "fd": 0, "ts_ns": 5_000_000_000}]
    assert _compute_next_page(logs, limit=100) == (False, None)


def test_next_page_empty_logs() -> None:
    assert _compute_next_page([], limit=100) == (False, None)


def test_next_page_full_page_uses_ts_ns_of_oldest() -> None:
    logs = [
        {"task_id": "t", "line": "x", "fd": 0, "ts_ns": 5_000_000_000},
        {"task_id": "t", "line": "y", "fd": 0, "ts_ns": 6_000_000_000},
    ]
    has_more, next_until = _compute_next_page(logs, limit=2)
    assert has_more is True
    assert next_until == (5_000_000_000 - 1) / 1_000_000_000


def test_next_page_falls_back_to_scaled_ts() -> None:
    logs = [
        {"task_id": "t", "line": "x", "fd": 0, "ts": 5.0},
        {"task_id": "t", "line": "y", "fd": 0, "ts": 6.0},
    ]
    has_more, next_until = _compute_next_page(logs, limit=2)
    assert has_more is True
    assert next_until == (5_000_000_000 - 1) / 1_000_000_000


# ── _parse_log_time ──────────────────────────────────────────────────────

# Fixed reference point: 2026-07-09T18:00:00Z.
PARSE_NOW = datetime(2026, 7, 9, 18, 0, 0, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_parse_time_empty_or_blank_returns_none(value: str) -> None:
    assert _parse_log_time(value, PARSE_NOW) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("45s", PARSE_NOW - 45),
        ("30m", PARSE_NOW - 30 * 60),
        ("2h", PARSE_NOW - 2 * 3600),
        ("1d", PARSE_NOW - 86400),
        ("0m", PARSE_NOW),
        ("30 m", PARSE_NOW - 30 * 60),
    ],
)
def test_parse_time_relative_age_subtracts_from_now(
    value: str, expected: float
) -> None:
    assert _parse_log_time(value, PARSE_NOW) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1720557600", 1720557600.0),
        ("1720557600.5", 1720557600.5),
        ("  1720557600  ", 1720557600.0),
    ],
)
def test_parse_time_epoch_seconds(value: str, expected: float) -> None:
    assert _parse_log_time(value, PARSE_NOW) == expected


def test_parse_time_iso_with_trailing_z_is_utc() -> None:
    expected = datetime(2026, 7, 9, 18, 0, 0, tzinfo=timezone.utc).timestamp()
    assert _parse_log_time("2026-07-09T18:00:00Z", PARSE_NOW) == expected


def test_parse_time_iso_naive_is_read_as_utc() -> None:
    expected = datetime(2026, 7, 9, 18, 0, 0, tzinfo=timezone.utc).timestamp()
    assert _parse_log_time("2026-07-09T18:00:00", PARSE_NOW) == expected


def test_parse_time_iso_with_explicit_offset_is_honored() -> None:
    # 18:00 at -04:00 is 22:00 UTC.
    expected = datetime(2026, 7, 9, 22, 0, 0, tzinfo=timezone.utc).timestamp()
    assert _parse_log_time("2026-07-09T18:00:00-04:00", PARSE_NOW) == expected


@pytest.mark.parametrize(
    "value",
    [
        "not-a-time",
        "30x",  # unknown unit
        "2026-13-01T00:00:00Z",  # invalid month
        "yesterday",
        "-5m",  # leading sign isn't part of the relative grammar
        "inf",
        "nan",
        "Infinity",
        "1e999",  # overflows to inf
    ],
)
def test_parse_time_unparseable_returns_none(value: str) -> None:
    assert _parse_log_time(value, PARSE_NOW) is None
