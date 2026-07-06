"""Shared trainer-side launch helpers for the disaggregated slime flow.

Vendored from the stitch ``slime_disagg`` cookbook: resolve HF repo ids +
materialize inline YAML configs, build the ``train.py`` command (optionally
sourcing a model arch script), and smoke the deployed Flash rollout pool.
"""

from __future__ import annotations

import json
import os
import shlex
import time
import urllib.error
import urllib.request
from collections import namedtuple
from collections.abc import Iterable
from typing import Any

from stitch.providers.modal import discover_flash_targets, resolve_flash_gateway_url


# ── Config preparation ────────────────────────────────────────────────────────


def prepare_config(cfg: Any, tmpdir: str, yaml_config_fields: Iterable[str]) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    from huggingface_hub import snapshot_download
    import yaml

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
    """Raised when a monotonic rollout pool has already advanced past a smoke version."""


def smoke_flash_pool(
    *,
    app_name: str,
    cls_name: str,
    model_name: str,
    weight_version: int,
    expect_min_containers: int,
    timeout_seconds: int,
) -> None:
    """Poll the Flash gateway until the warm pool serves completions at the
    expected weight version, via the gateway and each container directly."""
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        try:
            _smoke_warm_floor(
                app_name,
                cls_name,
                model_name,
                weight_version,
                expect_min_containers,
            )
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


def _smoke_warm_floor(
    app_name: str,
    cls_name: str,
    model_name: str,
    expected: int,
    expect_min_containers: int,
) -> None:
    gateway = resolve_flash_gateway_url(app_name, cls_name)
    targets = discover_flash_targets(app_name, cls_name)
    if len(targets) < expect_min_containers:
        raise RuntimeError(
            f"expected at least {expect_min_containers} containers, found {len(targets)}: {targets}"
        )
    print(f"Gateway URL: {gateway}")
    print(f"Direct container URLs ({len(targets)}):")
    for target in targets:
        print(f"  {target}")
    _assert_containers_at_version([gateway, *targets], expected)
    _assert_gateway_completion_exact(gateway, model_name, expected)


def _assert_containers_at_version(targets: list[str], expected: int) -> None:
    for target in targets:
        info = _get_json(f"{target}/server_info", timeout=30)
        print(f"{target} server_info={info}")
        current = int(info["current_version"])
        if current > expected:
            raise VersionAheadError(
                f"{target} current_version={current} already passed expected {expected}"
            )
        if current != expected:
            raise RuntimeError(
                f"{target} current_version={current} expected {expected}"
            )


def _assert_gateway_completion_exact(
    gateway: str, model_name: str, expected: int
) -> None:
    data = _post_json(
        f"{gateway}/v1/chat/completions",
        _completion_payload(model_name, expected),
        timeout=180,
    )
    print(f"Gateway completion: {data}")
    if (
        int(data.get("weight_version_start", -1)) != expected
        or int(data.get("weight_version_end", -1)) != expected
    ):
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")


def _completion_payload(model_name: str, expected: int) -> dict:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "weight_version": {"exact_version": expected},
        "chat_template_kwargs": {"enable_thinking": False},
    }


# ── Post-publish sync barrier ─────────────────────────────────────────────────

_SyncDetail = namedtuple("_SyncDetail", "synced count summary")


def wait_pool_synced(
    *,
    app_name: str,
    cls_name: str,
    version: int,
    timeout_seconds: float,
    poll_interval: float,
    min_containers: int = 1,
) -> bool:
    """Block until every discovered Flash target reports ``current_version >=
    version`` (and at least ``min_containers`` are present), so the next rollout
    generates against a fully-synced pool instead of racing servers that are
    still applying the just-published delta (which drops in-flight requests and
    hangs generation).

    Returns ``True`` once the pool is synced, or ``False`` if it did not reach
    the version within ``timeout_seconds`` — the caller proceeds regardless,
    since the staleness-gated rollout requests (bounded retries) remain the
    backstop and a hard block here would be worse than a brief version skew.
    """
    deadline = time.time() + timeout_seconds
    while True:
        detail = _pool_sync_detail(app_name, cls_name, version)
        if detail.synced and detail.count >= min_containers:
            print(f"[sync barrier] pool at v>={version}: {detail.summary}")
            return True
        if time.time() >= deadline:
            print(
                f"[sync barrier] pool did not reach v{version} within "
                f"{timeout_seconds:.0f}s ({detail.summary}); proceeding anyway"
            )
            return False
        print(f"[sync barrier] waiting for pool to reach v{version}: {detail.summary}")
        time.sleep(poll_interval)


def _pool_sync_detail(app_name: str, cls_name: str, version: int) -> _SyncDetail:
    try:
        targets = discover_flash_targets(app_name, cls_name)
    except Exception as exc:  # noqa: BLE001
        return _SyncDetail(False, 0, f"discover failed: {type(exc).__name__}: {exc}")
    versions: list[int] = []
    for target in targets:
        try:
            info = _get_json(f"{target}/server_info", timeout=15)
            versions.append(int(info.get("current_version", -1)))
        except Exception:  # noqa: BLE001
            versions.append(-1)
    synced = bool(versions) and all(v >= version for v in versions)
    at = sum(1 for v in versions if v >= version)
    summary = f"{at}/{len(versions)} at >=v{version} (versions={versions})"
    return _SyncDetail(synced, len(targets), summary)


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
