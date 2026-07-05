"""Trainer-side rollout-control primitives (rank probe + session affinity).

Vendored from the stitch ``slime_disagg`` cookbook so the trainer-side bulletin
hooks resolve these by dotted path inside the slime process.
"""

from __future__ import annotations

from typing import Any


def distributed_rank() -> int | None:
    """Return this process's torch-distributed rank, or ``None`` if torch
    distributed isn't initialized (single-process / pre-init).

    Used to gate rank-0-only side effects (pointer writes, transport copies,
    pool wakes) so only one writer acts per publish.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        return None
    return None


def apply_session_affinity(
    request: dict[str, Any], session_id: Any, header: str
) -> None:
    """Stamp ``header: session_id`` onto a rollout request's headers (idempotent).

    Sticky-session routing: a sample carrying a ``session_id`` should land on the
    same replica across turns. ``setdefault`` so an explicit caller header wins.
    """
    if not session_id:
        return
    headers = dict(request.get("headers") or {})
    headers.setdefault(header, str(session_id))
    request["headers"] = headers
