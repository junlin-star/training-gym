"""Launch-side helpers for the disaggregated slime flow.

Vendored from the stitch cookbook (``cookbook/common/launch.py`` +
``cookbook/common/smoke.py``): resolve HF repo ids and materialize inline YAML
configs, build the ``train.py`` command, and smoke the deployed Flash pool.
"""

from __future__ import annotations

import json
import os
import shlex
import time
import urllib.request
from collections.abc import Iterable
from typing import Any

from stitch.pools.modal_flash import ModalFlashPool
from stitch.types import VersionRef

# ── Config preparation ────────────────────────────────────────────────────────


def prepare_config(cfg: Any, tmpdir: str, yaml_config_fields: Iterable[str]) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    import yaml
    from huggingface_hub import snapshot_download

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(cfg, attr, None)) and not str(val).startswith("/"):
            setattr(cfg, attr, snapshot_download(val, local_files_only=True))

    for field in yaml_config_fields:
        if isinstance(val := getattr(cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(cfg, field, path)


def build_train_cmd(cfg: Any, trainer_root: str, *, model_script_attr: str) -> str:
    """Build the training command, sourcing model arch args if needed."""
    train_script = (
        f"{trainer_root}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    )
    model_script = getattr(cfg, model_script_attr, "")
    if model_script:
        inner = (
            f"source {trainer_root}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(cfg.cli_args())}"


# ── Flash pool smoke check ────────────────────────────────────────────────────


class VersionAheadError(RuntimeError):
    """A monotonic rollout pool has already advanced past the smoke's expected version."""


def smoke_flash_pool(
    *,
    app_name: str,
    cls_name: str,
    model_name: str,
    weight_version: int,
    expect_min_containers: int,
    timeout_seconds: int,
) -> None:
    """Poll until the pool serves completions at ``weight_version`` — through the
    gateway (Flash holds the request through a scaled-down pool's cold start) and then
    each live replica's ``/server_info``."""
    pool = ModalFlashPool(app_name, cls_name)
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        try:
            _smoke_once(pool, model_name, weight_version, expect_min_containers)
            return
        except VersionAheadError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if time.time() >= deadline:
            raise TimeoutError(
                f"Flash pool smoke did not pass before timeout: {last_error}"
            )
        print(f"Waiting for Flash pool readiness: {last_error}")
        time.sleep(10)


def _smoke_once(
    pool: ModalFlashPool,
    model_name: str,
    expected: int,
    expect_min_containers: int,
) -> None:
    gateway = pool.gateway_url()
    print(f"Gateway URL: {gateway}")
    # A fresh pool has no claimed run, so version 0 is unpinnable — an exact-version
    # request would 409. Gate on plain serving until a run has claimed the pool.
    if _get_json(f"{gateway}/server_info", timeout=60).get("run_id") is None:
        if expected != 0:
            raise RuntimeError(
                f"pool is unclaimed; cannot serve expected weight version {expected}"
            )
        data = _post_json(
            f"{gateway}/v1/chat/completions", _completion(model_name), timeout=900
        )
        _check_serves(data)
        print(f"Pool serves base (unclaimed): {data.get('choices')}")
        return
    data = _post_json(
        f"{gateway}/v1/chat/completions",
        _completion(model_name, expected),
        timeout=900,
    )
    print(f"Gateway completion: {data}")
    _check_completion(data, expected)
    replicas = pool.discover_replicas()
    if len(replicas) < expect_min_containers:
        raise RuntimeError(
            f"expected at least {expect_min_containers} containers, "
            f"found {len(replicas)}: {replicas}"
        )
    for target in replicas:
        info = _get_json(f"{target}/server_info", timeout=30)
        print(f"{target} server_info={info}")
        _check_version(_applied_version(info), expected, target)


def _applied_version(info: dict) -> int:
    applied = info.get("applied")
    return VersionRef.parse(applied).version if applied else -1


def _check_version(current: int, expected: int, target: str) -> None:
    if current > expected:
        raise VersionAheadError(
            f"{target} applied={current} already past expected {expected}"
        )
    if current != expected:
        raise RuntimeError(f"{target} applied={current}, expected {expected}")


def _check_completion(data: dict, expected: int) -> None:
    start = int(data.get("weight_version_start", -1))
    end = int(data.get("weight_version_end", -1))
    if start > expected or end > expected:
        raise VersionAheadError(
            f"gateway served {start}->{end}, already past expected {expected}"
        )
    if start != expected or end != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")


def _check_serves(data: dict) -> None:
    choices = data.get("choices") or []
    if not choices or not ((choices[0].get("message") or {}).get("content")):
        raise RuntimeError(f"pool did not return a completion: {data}")


def _completion(model_name: str, expected: int | None = None) -> dict:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if expected is not None:  # pin the version only against a claimed pool
        payload["weight_version"] = {"exact_version": expected}
    return payload


def _get_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.load(resp)
