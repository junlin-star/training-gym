"""Sidecar process launch + HTTP liveness helpers.

Vendored from the stitch ``slime_disagg`` cookbook. The weight-sync sidecar is
launched as ``python3 -m modal_training_gym.frameworks.stitch.sidecar`` on each
rollout replica.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

SIDECAR_MODULE = "modal_training_gym.frameworks.stitch.sidecar"


def start_sglang_sidecar(
    *,
    sidecar_port: int,
    sglang_port: int,
    bulletin_root: str,
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    volume_name: str,
    commit_mode: str,
    debug_requests: bool = False,
) -> subprocess.Popen:
    """Launch the weight-sync sidecar as a subprocess."""
    cmd = [
        "python3",
        "-m",
        SIDECAR_MODULE,
        "--host",
        "0.0.0.0",
        "--port",
        str(sidecar_port),
        "--upstream-url",
        f"http://127.0.0.1:{sglang_port}",
        "--bulletin-root",
        bulletin_root,
        "--local-checkpoint-dir",
        local_checkpoint_dir,
        "--base-checkpoint-dir",
        base_checkpoint_dir,
        "--volume-name",
        volume_name,
        "--commit-mode",
        commit_mode,
    ]
    if debug_requests:
        cmd.append("--debug-requests")
    print("Starting sidecar:", " ".join(cmd))
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
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
