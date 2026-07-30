"""Rollout-replica sidecar: the versioned proxy in front of the local SGLang.

Vendored from the stitch cookbook (``cookbook/common/sidecar.py``). Each rollout
container launches this as a subprocess via
:func:`sidecar_process.start_sidecar`, which passes every setting explicitly —
the recipe is the single source of truth, and the defaults here only exist so the
sidecar can be run standalone in dev.
"""

from __future__ import annotations

import argparse
import logging
import sys

from stitch.engines.sglang import SGLangEngine
from stitch.service import serve
from stitch.stores.modal_volume import ModalVolumeStore


def _configure_logging() -> None:
    """Emit INFO logs to stdout (uvicorn configures only its own loggers)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream", default="http://127.0.0.1:8001")
    parser.add_argument("--bulletin-root", required=True)
    parser.add_argument("--base-checkpoint-dir", required=True)
    parser.add_argument("--local-checkpoint-dir")
    parser.add_argument("--delta-update-mode", choices=["disk", "cpu"], required=True)
    parser.add_argument("--disk-load-format", default="auto")
    parser.add_argument("--volume-name", default="")
    parser.add_argument(
        "--commit-mode", choices=["in_place", "quiesce"], default="in_place"
    )
    parser.add_argument("--flush-cache-on-commit", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--debug-requests", action="store_true")
    # 0 disables the periodic re-check (the pool wake still drives reconciles).
    parser.add_argument("--reconcile-interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.delta_update_mode == "disk" and not args.local_checkpoint_dir:
        parser.error("--local-checkpoint-dir is required in disk mode")

    store = ModalVolumeStore(args.bulletin_root, volume_name=args.volume_name or None)
    engine = SGLangEngine(
        args.upstream,
        args.base_checkpoint_dir,
        args.local_checkpoint_dir,
        delta_update_mode=args.delta_update_mode,
        disk_load_format=args.disk_load_format,
    )
    serve(
        store,
        engine,
        run_id=args.run_id,
        commit_mode=args.commit_mode,
        flush_cache_on_commit=args.flush_cache_on_commit,
        host=args.host,
        port=args.port,
        debug_requests=args.debug_requests,
        reconcile_interval=args.reconcile_interval,
    )


if __name__ == "__main__":
    main()
