"""Modal Volume bulletin-board hooks for delta publish + rollout gating.

Vendored from the stitch ``slime_disagg`` cookbook (``cookbook/bulletin_hooks.py``).
The slime trainer resolves these by dotted path off the recipe:

- :func:`commit_and_wake` — ``StitchRecipe.custom_delta_pre_push_path``: advance
  the ``latest`` pointer, commit the Volume, best-effort wake the Flash pool.
- :func:`claim_pool` — called by the launcher before training publishes, to reset
  the pool to base for this run.
- :func:`gated_rollout_request_hook` — optional
  ``custom_rollout_request_hook_path`` that pins each request to a bounded-
  staleness weight version.

Config is read off the trainer's ``args`` namespace (slime setattr's every
``--custom-config-path`` key onto ``args``), with ``DELTA_*`` env vars as
fallback.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import BASE_VERSION, PointerRewind, parse_weight_identity
from stitch.providers.modal import (
    commit_volume,
    discover_flash_targets,
    volume_reloader,
    wake_targets,
)

from modal_training_gym.frameworks.stitch.rollout_control import (
    apply_session_affinity,
    distributed_rank,
)

logger = logging.getLogger(__name__)


# ── Publish hook ──────────────────────────────────────────────────────────────


def commit_and_wake(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """Trainer ``custom_delta_pre_push_path`` hook (publish-only, bulletin board).

    The trainer has written ``weight_v{N}/`` to the Modal Volume. Advance the
    committed ``latest`` pointer, commit the Volume so the rollout pool's
    ``reload`` sees the new version, then best-effort wake the Flash pool. The
    sidecars self-sync (wake RPC, periodic poll, startup), so a missed wake only
    costs latency.
    """
    del rollout_engines
    version = parse_weight_identity(Path(version_dir).name)
    rank = distributed_rank()

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
    _barrier_until_synced(args, version)


def _barrier_until_synced(args: Any, version: int) -> None:
    """Block until the Flash pool serves ``version`` before returning control to
    slime, so the next rollout generation targets a fully-synced pool. Without
    this, slime dispatches the next rollout's requests while the pool is still
    applying the just-published delta; sglang drops those in-flight requests on
    reload and the rollout hangs waiting for completions that never arrive.

    Bounded by a timeout (then proceeds): the staleness-gated rollout requests
    are the backstop, and a hard block here would be worse than a brief skew.
    Toggle via the ``rollout_sync_barrier`` config key (or ``DELTA_ROLLOUT_SYNC_BARRIER``)."""
    if not _barrier_enabled(args):
        return
    try:
        from modal_training_gym.frameworks.stitch.trainer_helpers import (
            wait_pool_synced,
        )

        app_name = (
            getattr(args, "rollout_modal_flash_app_name", None)
            or os.environ["DELTA_APP_NAME"]
        )
        cls_name = getattr(
            args, "rollout_modal_flash_server_cls_name", None
        ) or os.getenv("DELTA_SERVER_CLS_NAME", "Server")
        wait_pool_synced(
            app_name=app_name,
            cls_name=cls_name,
            version=version,
            timeout_seconds=float(
                getattr(args, "rollout_sync_barrier_timeout_seconds", None) or 600.0
            ),
            poll_interval=float(
                getattr(args, "rollout_sync_barrier_poll_seconds", None) or 3.0
            ),
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Rollout sync barrier for version %s failed; proceeding "
            "(staleness-gated requests remain the backstop)",
            version,
            exc_info=True,
        )


def _barrier_enabled(args: Any) -> bool:
    val = getattr(args, "rollout_sync_barrier", None)
    if val is None:
        val = os.getenv("DELTA_ROLLOUT_SYNC_BARRIER", "1")
    return str(val).strip().lower() not in ("0", "false", "no", "")


def claim_pool(args: Any) -> None:
    """Launch hook (rank 0): claim the rollout pool for this run.

    Write the empty pointer ``<run_id>/weight_v000000``, commit the Volume, and
    wake the pool — so every replica resets to base before the first delta
    publishes. ``run_id`` must be fresh per launch (the run's fence token).
    """
    if distributed_rank() not in (None, 0):
        return
    board = FilesystemBulletinBoard(_transport_root(args), layout="slime")
    board.claim(_run_id(args))
    commit_volume(_volume_name(args))
    _best_effort_wake(args, BASE_VERSION)


def _best_effort_wake(args: Any, version: int) -> None:
    """Nudge warm Flash containers to reconcile now. Best-effort: a transient
    Modal control-plane error must not kill the training step — ``latest`` is
    already committed and sidecars self-sync on their next poll/startup."""
    try:
        app_name = (
            getattr(args, "rollout_modal_flash_app_name", None)
            or os.environ["DELTA_APP_NAME"]
        )
        cls_name = getattr(
            args, "rollout_modal_flash_server_cls_name", None
        ) or os.getenv("DELTA_SERVER_CLS_NAME", "Server")
        wake_targets(
            discover_flash_targets(app_name=app_name, cls_name=cls_name), version
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Best-effort rollout wake failed for version %s; sidecars will self-sync",
            version,
            exc_info=True,
        )


# ── Staleness-gated rollout requests ──────────────────────────────────────────


class CachedLatestPointer:
    """TTL-cached ``(run_id, version)`` from the bulletin board's ``latest`` pointer."""

    def __init__(self) -> None:
        self.version: int = 0
        self.run_id: str | None = None
        self._refreshed_at: float = -1e9
        self._board: FilesystemBulletinBoard | None = None

    async def get(self, args: Any, ttl: float = 2.0) -> int:
        now = time.monotonic()
        if self._board is None:
            self._board = _gate_board(args)
        if now - self._refreshed_at >= ttl:
            self._refreshed_at = now
            try:
                await self._board.refresh()
                run_id, version = self._board.read_latest()
                self.run_id = run_id
                self.version = int(version)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "gate: could not read latest published version; using cached %s",
                    self.version,
                    exc_info=True,
                )
        return self.version


_latest_cache = CachedLatestPointer()


async def gated_rollout_request_hook(
    args: Any, sample: Any, request: dict[str, Any]
) -> None:
    """Optional ``custom_rollout_request_hook_path``: gate each rollout on
    ``weight_version - k`` so unusable (too-stale) rollouts are never generated."""
    mode = str(getattr(args, "rollout_request_weight_version_mode", "min"))
    if mode != "none":
        latest = await _latest_cache.get(args)
        lag = int(getattr(args, "rollout_request_weight_version_lag", 0))
        target = max(0, latest - lag)
        key = "exact_version" if mode == "exact" else "min_required_version"
        request["payload"]["weight_version"] = {key: target}

    request["max_retries"] = int(
        getattr(args, "rollout_request_retry_attempts", request.get("max_retries", 60))
    )
    request["retry_sleep"] = float(
        getattr(args, "rollout_request_retry_sleep", request.get("retry_sleep", 1.0))
    )

    header = str(getattr(args, "rollout_session_affinity_header", "x-session-affinity"))
    apply_session_affinity(request, getattr(sample, "session_id", None), header)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _volume_name(args: Any) -> str:
    return str(
        getattr(args, "update_weight_delta_volume_name", None)
        or os.environ["DELTA_VOLUME_NAME"]
    )


def bulletin_root(args: Any) -> str:
    """Where the trainer writes version dirs: ``<transport_root>/<run_id>``."""
    return str(
        getattr(args, "update_weight_disk_dir", None)
        or os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin")
    )


def _transport_root(args: Any) -> str:
    """The Volume mount root holding the canonical ``latest`` pointer — the
    parent of the per-run write dir."""
    return str(Path(bulletin_root(args)).parent)


def _run_id(args: Any) -> str:
    """The run partition (chain identity), passed explicitly via custom_config."""
    run_id = getattr(args, "run_id", None)
    if not run_id:
        raise ValueError(
            "run_id is required (pass it via custom_config_path); the bulletin "
            "hooks no longer derive it from the write-dir basename"
        )
    return str(run_id)


def _gate_board(args: Any) -> FilesystemBulletinBoard:
    vol = getattr(args, "update_weight_delta_volume_name", None) or os.environ.get(
        "DELTA_VOLUME_NAME"
    )
    refresh = volume_reloader(vol) if vol else None
    return FilesystemBulletinBoard(
        _transport_root(args), refresh=refresh, layout="slime"
    )
