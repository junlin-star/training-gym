"""Append-only authority for retry-safe ``TrainingRun`` state.

The historical ``training-runs/<id>.json`` record remains a disposable cache.
It is necessarily vulnerable to read-modify-write races because Modal Volumes
do not provide compare-and-swap.  Claim-grade runs additionally publish
immutable, content-addressed events.  Reads deterministically materialize the
active attempt from those events, so:

* a terminal event dominates delayed status writes from the same attempt;
* every event from an older attempt is provenance, never active state;
* two attempt IDs claiming the same attempt number are a detected fork; and
* conflicting terminal outcomes fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Literal

from modal_training_gym.utils.metadata import (
    vol_list,
    vol_put_immutable,
    vol_remove_store,
)

RUN_EVENT_SCHEMA_VERSION = 1
RUN_EVENT_STORE_PREFIX = "training-run-events"
RUN_EVENT_KINDS = frozenset({"started", "snapshot", "failure", "terminal"})
_TERMINAL_STATUSES = frozenset({"stopped", "cancelled", "completed", "failed"})

RunEventKind = Literal["started", "snapshot", "failure", "terminal"]


class RunEventConflict(RuntimeError):
    """The immutable run journal is internally inconsistent."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _event_store(training_run_id: str) -> str:
    if not isinstance(training_run_id, str) or not training_run_id.strip():
        raise ValueError("training_run_id must be a nonempty string")
    digest = hashlib.sha256(training_run_id.encode()).hexdigest()
    return f"{RUN_EVENT_STORE_PREFIX}/{digest}"


def _attempt_identity(snapshot: Mapping[str, Any]) -> tuple[int, str] | None:
    metadata = snapshot.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    attempt_id = metadata.get("active_attempt_id")
    attempt_count = metadata.get("attempt_count")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
        or not isinstance(attempt_id, str)
        or not attempt_id.strip()
    ):
        return None
    return attempt_count, attempt_id


def build_training_run_event(
    snapshot: Mapping[str, Any],
    *,
    kind: RunEventKind,
    observed_at_ns: int | None = None,
) -> dict[str, Any]:
    """Build one self-authenticating event from a JSON-compatible run view."""
    if kind not in RUN_EVENT_KINDS:
        raise ValueError(f"unsupported training-run event kind: {kind!r}")
    identity = _attempt_identity(snapshot)
    if identity is None:
        raise ValueError("training-run events require an active attempt identity")
    attempt_count, attempt_id = identity
    training_run_id = snapshot.get("training_run_id")
    if not isinstance(training_run_id, str) or not training_run_id.strip():
        raise ValueError("training-run event snapshot has no run ID")
    timestamp = time.time_ns() if observed_at_ns is None else observed_at_ns
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 1:
        raise ValueError("observed_at_ns must be a positive integer")

    body = {
        "schema_version": RUN_EVENT_SCHEMA_VERSION,
        "kind": kind,
        "training_run_id": training_run_id,
        "attempt_count": attempt_count,
        "attempt_id": attempt_id,
        "observed_at_ns": timestamp,
        "payload": {"run_snapshot": copy.deepcopy(dict(snapshot))},
    }
    event_id = hashlib.sha256(_canonical_json(body)).hexdigest()
    return {**body, "event_id": event_id}


def append_training_run_event(
    snapshot: Mapping[str, Any],
    *,
    kind: RunEventKind,
    observed_at_ns: int | None = None,
    is_async: bool = False,
) -> None | Awaitable[None]:
    """Publish one immutable event; identical publications are idempotent."""
    event = build_training_run_event(
        snapshot,
        kind=kind,
        observed_at_ns=observed_at_ns,
    )
    key = (
        f"{int(event['attempt_count']):08d}-{event['attempt_id']}-"
        f"{event['kind']}-{event['event_id']}"
    )
    return vol_put_immutable(
        _event_store(str(event["training_run_id"])),
        key,
        event,
        is_async=is_async,
    )


def _validated_event(raw: Any, training_run_id: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RunEventConflict("training-run journal contains a non-object event")
    event = copy.deepcopy(dict(raw))
    if event.get("schema_version") != RUN_EVENT_SCHEMA_VERSION:
        raise RunEventConflict("training-run journal has an unsupported schema")
    if event.get("training_run_id") != training_run_id:
        raise RunEventConflict("training-run journal contains a foreign run ID")
    kind = event.get("kind")
    if kind not in RUN_EVENT_KINDS:
        raise RunEventConflict("training-run journal contains an invalid event kind")
    attempt_count = event.get("attempt_count")
    attempt_id = event.get("attempt_id")
    observed_at_ns = event.get("observed_at_ns")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
        or not isinstance(attempt_id, str)
        or not attempt_id.strip()
        or isinstance(observed_at_ns, bool)
        or not isinstance(observed_at_ns, int)
        or observed_at_ns < 1
    ):
        raise RunEventConflict("training-run journal has an invalid attempt identity")
    event_id = event.pop("event_id", None)
    expected_id = hashlib.sha256(_canonical_json(event)).hexdigest()
    event["event_id"] = event_id
    if event_id != expected_id:
        raise RunEventConflict("training-run event content hash does not match")
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        raise RunEventConflict("training-run event payload is not an object")
    snapshot = payload.get("run_snapshot")
    if not isinstance(snapshot, Mapping):
        raise RunEventConflict("training-run snapshot event has no snapshot")
    if snapshot.get("training_run_id") != training_run_id:
        raise RunEventConflict("training-run snapshot has a foreign run ID")
    if _attempt_identity(snapshot) != (attempt_count, attempt_id):
        raise RunEventConflict("training-run snapshot attempt identity disagrees")
    status = snapshot.get("status")
    if kind == "failure" and status != "failed":
        raise RunEventConflict("failure training-run event is not failed")
    if kind == "terminal" and status not in _TERMINAL_STATUSES:
        raise RunEventConflict("terminal training-run event is not terminal")
    if kind in {"started", "snapshot"} and status in _TERMINAL_STATUSES:
        raise RunEventConflict("nonterminal training-run event is terminal")
    return event


def _event_order(event: Mapping[str, Any]) -> tuple[int, str]:
    return int(event["observed_at_ns"]), str(event["event_id"])


def _snapshot(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event["payload"]
    assert isinstance(payload, Mapping)
    snapshot = payload["run_snapshot"]
    assert isinstance(snapshot, Mapping)
    return copy.deepcopy(dict(snapshot))


def _terminal_identity(event: Mapping[str, Any]) -> tuple[str, str]:
    snapshot = _snapshot(event)
    return str(snapshot.get("status") or ""), str(snapshot.get("error_message") or "")


def materialize_training_run_payload(
    training_run_id: str,
    base_payload: Mapping[str, Any] | None,
    raw_events: Sequence[Any],
) -> dict[str, Any] | None:
    """Project an order-independent event set into the authoritative run view."""
    if not raw_events:
        return copy.deepcopy(dict(base_payload)) if base_payload is not None else None

    events_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_events:
        event = _validated_event(raw, training_run_id)
        event_id = str(event["event_id"])
        previous = events_by_id.get(event_id)
        if previous is not None and previous != event:
            raise RunEventConflict("one training-run event ID has conflicting bytes")
        events_by_id[event_id] = event
    events = list(events_by_id.values())

    ids_by_count: dict[int, set[str]] = defaultdict(set)
    counts_by_id: dict[str, set[int]] = defaultdict(set)
    for event in events:
        count = int(event["attempt_count"])
        attempt_id = str(event["attempt_id"])
        ids_by_count[count].add(attempt_id)
        counts_by_id[attempt_id].add(count)
    forks = {count: ids for count, ids in ids_by_count.items() if len(ids) != 1}
    reused = {
        attempt_id: counts
        for attempt_id, counts in counts_by_id.items()
        if len(counts) != 1
    }
    if forks or reused:
        raise RunEventConflict(
            f"training-run attempt journal forked: counts={forks}, ids={reused}"
        )
    observed_counts = sorted(ids_by_count)
    if observed_counts != list(range(1, observed_counts[-1] + 1)):
        raise RunEventConflict(
            f"training-run attempt counts are not contiguous: {observed_counts}"
        )
    for count, ids in ids_by_count.items():
        attempt_id = next(iter(ids))
        starts = [
            event
            for event in events
            if event["attempt_count"] == count
            and event["attempt_id"] == attempt_id
            and event["kind"] == "started"
        ]
        if len(starts) != 1:
            raise RunEventConflict(
                f"training attempt {count} has {len(starts)} start events"
            )

    active_count = max(ids_by_count)
    active_id = next(iter(ids_by_count[active_count]))
    active = [
        event
        for event in events
        if event["attempt_count"] == active_count and event["attempt_id"] == active_id
    ]
    terminals = [event for event in active if event["kind"] == "terminal"]
    terminal_identities = {_terminal_identity(event) for event in terminals}
    if len(terminal_identities) > 1:
        raise RunEventConflict("active training attempt has conflicting terminals")
    if terminals:
        selected = max(terminals, key=_event_order)
        result = _snapshot(selected)
    else:
        failures = [event for event in active if event["kind"] == "failure"]
        failure_identities = {_terminal_identity(event) for event in failures}
        if len(failure_identities) > 1:
            raise RunEventConflict("active training attempt has conflicting failures")
        if failures:
            selected = max(failures, key=_event_order)
            result = _snapshot(selected)
        else:
            snapshots = [
                event for event in active if event["kind"] in {"started", "snapshot"}
            ]
            if snapshots:
                selected = max(snapshots, key=_event_order)
                result = _snapshot(selected)
            elif base_payload is not None and _attempt_identity(base_payload) == (
                active_count,
                active_id,
            ):
                selected = None
                result = copy.deepcopy(dict(base_payload))
            else:
                raise RunEventConflict("active training attempt has no state snapshot")

            # Framework phase/progress is presentation state, not scientific
            # authority. Preserve the latest same-attempt cache view without
            # creating one immutable Volume object per optimizer-step report.
            # A terminal/failure event above always dominates this cache, and
            # an older attempt's cache cannot be overlaid onto a newer attempt.
            if (
                base_payload is not None
                and _attempt_identity(base_payload) == (active_count, active_id)
                and base_payload.get("status") == "running"
            ):
                if base_payload.get("framework_status") is not None:
                    result["framework_status"] = copy.deepcopy(
                        base_payload["framework_status"]
                    )
                base_metadata = base_payload.get("metadata")
                base_progress = (
                    base_metadata.get("framework_progress")
                    if isinstance(base_metadata, Mapping)
                    else None
                )
                if isinstance(base_progress, Mapping):
                    result_metadata = dict(result.get("metadata") or {})
                    result_metadata["framework_progress"] = copy.deepcopy(
                        dict(base_progress)
                    )
                    result["metadata"] = result_metadata
                result["updated_at"] = max(
                    int(result.get("updated_at") or 0),
                    int(base_payload.get("updated_at") or 0),
                )

    if _attempt_identity(result) != (active_count, active_id):
        raise RunEventConflict("materialized run does not own the active attempt")
    result["updated_at"] = max(
        int(result.get("updated_at") or 0),
        max(int(event["observed_at_ns"]) // 1_000_000_000 for event in active),
    )
    return result


def load_materialized_training_run(
    training_run_id: str,
    base_payload: Mapping[str, Any] | None,
    *,
    is_async: bool = False,
) -> dict[str, Any] | None | Awaitable[dict[str, Any] | None]:
    """Read and materialize the immutable journal for one logical run."""
    store = _event_store(training_run_id)
    if is_async:

        async def _run() -> dict[str, Any] | None:
            events = await vol_list(store, is_async=True)
            return materialize_training_run_payload(
                training_run_id,
                base_payload,
                events,
            )

        return _run()
    events = vol_list(store)
    return materialize_training_run_payload(training_run_id, base_payload, events)


def delete_training_run_events(training_run_id: str) -> int:
    """Remove one deleted run's immutable journal so it cannot be resurrected."""
    return vol_remove_store(_event_store(training_run_id))


__all__ = [
    "RUN_EVENT_SCHEMA_VERSION",
    "RunEventConflict",
    "append_training_run_event",
    "build_training_run_event",
    "delete_training_run_events",
    "load_materialized_training_run",
    "materialize_training_run_payload",
]
