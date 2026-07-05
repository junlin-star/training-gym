"""Stitch bulletin-board hooks for the training-gym stitch framework.

Thin wrappers around stitch's bulletin board protocol: the trainer publishes
weight deltas to a Modal Volume, and these hooks advance the ``latest``
pointer, commit the Volume, and wake the Flash rollout pool.

These are invoked inside the slime training process via dotted-string
references (``custom_delta_pre_push_path``, etc.).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def commit_and_wake(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """Trainer ``custom_delta_pre_push_path`` hook.

    Advance the ``latest`` pointer on the bulletin board, commit the Modal
    Volume so rollout sidecars see the new version, then best-effort wake
    the Flash pool.
    """
    del rollout_engines
    from stitch.bulletin import FilesystemBulletinBoard
    from stitch.protocol import PointerRewind, parse_weight_identity
    from stitch.providers.modal import commit_volume

    version = parse_weight_identity(Path(version_dir).name)
    rank = _distributed_rank()

    if version is not None and rank in (None, 0):
        board = FilesystemBulletinBoard(_transport_root(args), layout="slime")
        try:
            board.advance(_run_id(args), version)
        except PointerRewind:
            logger.warning(
                "publish of version %s would rewind latest; dropping (run %r)",
                version,
                _run_id(args),
                exc_info=True,
            )
            return
    commit_volume(_volume_name(args))

    if version is None or rank not in (None, 0):
        return
    _best_effort_wake(args, version)


def claim_pool(args: Any) -> None:
    """Trainer launch hook (rank 0): claim the rollout pool for this run.

    Write the empty base pointer, commit the Volume, and wake the pool so
    every replica resets to base before the first delta publishes.
    """
    from stitch.bulletin import FilesystemBulletinBoard
    from stitch.protocol import BASE_VERSION
    from stitch.providers.modal import commit_volume

    if _distributed_rank() not in (None, 0):
        return
    board = FilesystemBulletinBoard(_transport_root(args), layout="slime")
    board.claim(_run_id(args))
    commit_volume(_volume_name(args))
    _best_effort_wake(args, BASE_VERSION)


def _best_effort_wake(args: Any, version: int) -> None:
    """Nudge warm Flash containers to reconcile now."""
    try:
        from stitch.providers.modal import (
            discover_flash_targets,
            wake_targets,
        )

        app_name = (
            getattr(args, "rollout_modal_flash_app_name", None)
            or os.environ["DELTA_APP_NAME"]
        )
        cls_name = getattr(
            args, "rollout_modal_flash_server_cls_name", None
        ) or os.getenv("DELTA_SERVER_CLS_NAME", "Server")
        wake_targets(
            discover_flash_targets(app_name=app_name, cls_name=cls_name),
            version,
        )
    except Exception:
        logger.warning(
            "Best-effort rollout wake failed for version %s; sidecars will self-sync",
            version,
            exc_info=True,
        )


def _distributed_rank() -> int | None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        return None
    return None


def _volume_name(args: Any) -> str:
    return str(
        getattr(args, "update_weight_delta_volume_name", None)
        or os.environ["DELTA_VOLUME_NAME"]
    )


def _transport_root(args: Any) -> str:
    return str(
        Path(
            getattr(args, "update_weight_disk_dir", None)
            or os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin")
        ).parent
    )


def _run_id(args: Any) -> str:
    run_id = getattr(args, "run_id", None)
    if not run_id:
        raise ValueError("run_id is required (pass it via custom_config_path)")
    return str(run_id)
