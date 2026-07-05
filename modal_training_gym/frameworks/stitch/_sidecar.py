"""SGLang weight-sync sidecar entry point for the stitch framework.

Launched as ``python3 -m modal_training_gym.frameworks.stitch._sidecar``
by each rollout replica. Uses slime's host-side disk_delta decoder.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import logging
import os
import shutil
from typing import Callable

logger = logging.getLogger(__name__)

DISK_DELTA_MODULE = "slime.utils.disk_delta"


def _parallel_init_local_checkpoint(
    disk_delta_module: str, workers: int = 32
) -> Callable[[str, str], None]:
    """Return a concurrent ``init_local_checkpoint(local, base)``."""
    import importlib

    def _base_fingerprint(base_dir: str) -> str:
        h = hashlib.sha256()
        for e in sorted(os.scandir(base_dir), key=lambda e: e.name):
            if e.is_file():
                st = e.stat()
                h.update(f"{e.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        return h.hexdigest()

    def _init(local_ckpt_dir: str, base_dir: str) -> None:
        dd = importlib.import_module(disk_delta_module)
        try:
            apply_lock = dd._apply_lock
            read_version = dd._read_applied_version
            write_version = dd._write_applied_version
            drop_page_cache = dd.drop_page_cache
        except AttributeError:
            dd.init_local_checkpoint(local_ckpt_dir, base_dir)
            return

        fp_path = os.path.join(local_ckpt_dir, ".base_fingerprint")
        base_fp = _base_fingerprint(base_dir)
        with apply_lock(local_ckpt_dir):
            if read_version(local_ckpt_dir) is not None:
                try:
                    cur = open(fp_path).read().strip()
                except FileNotFoundError:
                    cur = None
                if cur == base_fp:
                    return
                logger.warning("stale /local-checkpoint (base changed) — wiping")
                shutil.rmtree(local_ckpt_dir, ignore_errors=True)
            logger.info(
                "Materializing base %s -> %s (%d workers)",
                base_dir,
                local_ckpt_dir,
                workers,
            )
            os.makedirs(local_ckpt_dir, exist_ok=True)
            files = [e for e in os.scandir(base_dir) if e.is_file()]

            def _copy(entry: os.DirEntry) -> None:
                shutil.copy2(
                    entry.path,
                    os.path.join(local_ckpt_dir, entry.name),
                )
                drop_page_cache(entry.path)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(workers, max(1, len(files)))
            ) as ex:
                for _ in ex.map(_copy, files):
                    pass
            write_version(local_ckpt_dir, "000000")
            with open(fp_path, "w") as f:
                f.write(base_fp)

    return _init


def main() -> None:
    """Parse args and run the sidecar uvicorn server."""
    from stitch.bulletin import FilesystemBulletinBoard
    from stitch.engines.sglang import SGLangDiskDeltaAdapter
    from stitch.servers.sglang import create_app
    from stitch.sync import WeightSyncManager

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument(
        "--bulletin-root",
        default=os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin"),
    )
    parser.add_argument(
        "--volume-name",
        default=os.environ.get("DELTA_VOLUME_NAME", ""),
    )
    parser.add_argument(
        "--local-checkpoint-dir",
        default=os.environ.get("STITCH_LOCAL_CHECKPOINT_DIR", "/local-checkpoint"),
    )
    parser.add_argument(
        "--base-checkpoint-dir",
        default=os.environ.get("STITCH_BASE_CHECKPOINT_DIR"),
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("DISAGG_RUN_ID"),
    )
    parser.add_argument(
        "--base-copy-workers",
        type=int,
        default=int(os.environ.get("SIDECAR_BASE_COPY_WORKERS", "32")),
    )
    parser.add_argument(
        "--commit-mode",
        choices=("quiesce", "in_place"),
        default=os.environ.get("SIDECAR_COMMIT_MODE", "in_place"),
    )
    parser.add_argument(
        "--debug-requests",
        action="store_true",
        default=os.environ.get("SIDECAR_DEBUG_REQUESTS", "").lower()
        in {"1", "true", "yes"},
    )
    parser.add_argument(
        "--upstream-timeout",
        type=float,
        default=float(os.environ.get("SIDECAR_UPSTREAM_TIMEOUT", "3600")),
    )
    args = parser.parse_args()
    if not args.base_checkpoint_dir:
        raise SystemExit(
            "--base-checkpoint-dir is required: deltas are applied "
            "host-side on top of a copy of this base HF checkpoint."
        )

    logging.basicConfig(level=logging.INFO)

    refresh = None
    if args.volume_name:
        from stitch.providers.modal import volume_reloader

        refresh = volume_reloader(args.volume_name)

    board = FilesystemBulletinBoard(args.bulletin_root, refresh=refresh, layout="slime")
    engine = SGLangDiskDeltaAdapter(
        upstream_url=args.upstream_url,
        local_checkpoint_dir=args.local_checkpoint_dir,
        base_checkpoint_dir=args.base_checkpoint_dir,
        apply_deltas=None,
        init_local_checkpoint=_parallel_init_local_checkpoint(
            DISK_DELTA_MODULE, args.base_copy_workers
        ),
    )
    manager = WeightSyncManager(
        board=board,
        engine=engine,
        run_id=args.run_id,
        commit_mode=args.commit_mode,
        debug_requests=args.debug_requests,
    )

    import uvicorn

    uvicorn.run(
        create_app(
            manager,
            upstream_url=args.upstream_url,
            upstream_timeout=args.upstream_timeout,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
