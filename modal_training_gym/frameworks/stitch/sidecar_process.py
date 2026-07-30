"""Subprocess + runtime helpers for the disaggregated flow.

Vendored from the stitch cookbook (``cookbook/common/process.py``): launch the
sidecar beside SGLang, wait on HTTP liveness, terminate cleanly, trace host RAM,
and probe torch-distributed rank/barrier for the rank-gated publish hooks.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

SIDECAR_MODULE = "modal_training_gym.frameworks.stitch.sidecar"


def start_sidecar(
    *,
    sidecar_port: int,
    sglang_port: int,
    bulletin_root: str,
    base_checkpoint_dir: str,
    local_checkpoint_dir: str | None,
    delta_update_mode: str,
    disk_load_format: str,
    volume_name: str,
    commit_mode: str,
    flush_cache_on_commit: bool = False,
    debug_requests: bool = False,
) -> subprocess.Popen:
    """Launch the versioned rollout proxy (the stitch sidecar) beside SGLang."""
    cmd = [
        "python3",
        "-m",
        SIDECAR_MODULE,
        "--host",
        "0.0.0.0",
        "--port",
        str(sidecar_port),
        "--upstream",
        f"http://127.0.0.1:{sglang_port}",
        "--bulletin-root",
        bulletin_root,
        "--base-checkpoint-dir",
        base_checkpoint_dir,
        "--delta-update-mode",
        delta_update_mode,
        "--disk-load-format",
        disk_load_format,
        "--volume-name",
        volume_name,
        "--commit-mode",
        commit_mode,
    ]
    if local_checkpoint_dir is not None:
        cmd.extend(["--local-checkpoint-dir", local_checkpoint_dir])
    if flush_cache_on_commit:
        cmd.append("--flush-cache-on-commit")
    if debug_requests:
        cmd.append("--debug-requests")
    print("Starting sidecar:", " ".join(cmd))
    # Note: do not tee this to the bulletin volume. An open file on a Modal Volume
    # makes ``Volume.reload()`` fail with "there are open files preventing the
    # operation", which is exactly the call the reconciler uses to see new
    # versions — the replica would then never sync.
    return subprocess.Popen(cmd, start_new_session=True)


def wait_http(url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"process exited while waiting for {url}: code={process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}; last error: {last_error}")


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def start_host_mem_monitor(interval_s: int = 20) -> None:
    """Trace this node's host RAM from a daemon thread. Modal exposes no host-RAM
    metric, so this log line is the only signal for the OOM peak (the publish
    weight-gather). Best-effort."""
    host = socket.gethostname()

    def _meminfo() -> tuple[float, float]:
        total = avail = 0.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / 1024 / 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) / 1024 / 1024
        except Exception:  # noqa: BLE001
            pass
        return total, avail

    def _loop() -> None:
        # One line per node per tick floods a multi-node log, so stay quiet unless host
        # RAM is climbing toward a host OOM, or on a sparse heartbeat.
        heartbeat = max(1, 600 // interval_s)
        i = 0
        while True:
            total, avail = _meminfo()
            if i == 0 or avail < 500 or i % heartbeat == 0:
                print(
                    f"[hostmem] {host} used={total - avail:.0f}GiB "
                    f"avail={avail:.0f}GiB total={total:.0f}GiB",
                    flush=True,
                )
            i += 1
            time.sleep(interval_s)

    threading.Thread(target=_loop, daemon=True, name="host-mem-monitor").start()


def dist_rank() -> int | None:
    """This process's torch-distributed rank, or ``None`` off the distributed path.

    Gates rank-0-only side effects (pointer writes, pool wakes) so only one writer
    acts per publish.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        return None
    return None


def dist_barrier() -> None:
    """Wait for all ranks; a no-op off the distributed path."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
