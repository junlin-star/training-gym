from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit

import httpx
import modal

from modal_training_gym.common.checkpoint import to_volume_path
from modal_training_gym.common.config import load_proxy_auth, modal_proxy_auth_headers
from modal_training_gym.common.errors import TrainingGymConfigError

_MODAL_HOST_SUFFIXES = (
    ".modal.run",
    ".modal.host",
    ".modal.direct",
    ".modal.dev",
)

_DEAD_ENDPOINT_STATUSES = frozenset({"failed", "cancelled", "cancelling", "stopped"})


def is_modal_host(base_url: str) -> bool:
    parsed = urlsplit(base_url)
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return any(
        host == suffix.lstrip(".") or host.endswith(suffix)
        for suffix in _MODAL_HOST_SUFFIXES
    )


def proxy_auth_headers(base_url: str) -> dict[str, str]:
    return modal_proxy_auth_headers() if is_modal_host(base_url) else {}


def _list_endpoints(*, env: str | None = None) -> list[dict[str, Any]]:
    command = [sys.executable, "-m", "modal", "endpoint", "list", "--json"]
    if env:
        command.extend(["--env", env])
    return json.loads(subprocess.check_output(command, text=True, timeout=60))


def _find_endpoint(name: str, endpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    for endpoint in endpoints:
        if endpoint.get("name") == name:
            return endpoint
    return None


def ensure_endpoint(
    *,
    name: str,
    model: str,
    unauthenticated: bool = True,
    routing_region: str | None = None,
    custom_hf_repo: str | None = None,
    custom_hf_revision: str | None = None,
    custom_volume_name: str | None = None,
    custom_volume_path: str | None = None,
    custom_volume_mount_path: str = "/checkpoints",
    env: str | None = None,
    wait_timeout_sec: float = 15 * 60,
) -> str:
    """Create or reuse a managed Modal Endpoint and return its ready base URL.

    The endpoint name is derived from *name* and the serving configuration.
    Set ``unauthenticated=False`` for Modal proxy authentication. Custom
    weights may come from either a Hugging Face repository or a Modal Volume.
    """
    if custom_hf_repo and custom_volume_name:
        raise TrainingGymConfigError(
            "Pass either custom_hf_repo or custom_volume_name, not both."
        )
    if custom_hf_revision and not custom_hf_repo:
        raise TrainingGymConfigError("custom_hf_revision requires custom_hf_repo.")
    if custom_volume_path is not None and not custom_volume_name:
        raise TrainingGymConfigError("custom_volume_path requires custom_volume_name.")
    if custom_volume_name and not custom_volume_path:
        raise TrainingGymConfigError("custom_volume_name requires custom_volume_path.")
    if not unauthenticated and not load_proxy_auth():
        raise TrainingGymConfigError(
            "Authenticated Modal Endpoints require MODAL_KEY and MODAL_SECRET. "
            "Run `training-gym set-proxy-auth` or export both variables."
        )

    volume_path = (
        to_volume_path(custom_volume_path, custom_volume_mount_path)
        if custom_volume_path
        else None
    )
    if custom_volume_name and volume_path == "":
        volume_path = "/"

    spec = {
        "model": model,
        "routing_region": routing_region,
        "custom_hf_repo": custom_hf_repo,
        "custom_hf_revision": custom_hf_revision,
        "custom_volume_name": custom_volume_name,
        "custom_volume_path": volume_path,
        "unauthenticated": unauthenticated,
    }
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    endpoint_name = f"{name[:48].rstrip('-')}-{digest}"

    existing = _find_endpoint(endpoint_name, _list_endpoints(env=env))
    if existing:
        status = str(existing.get("status") or "").strip().lower()
        if status in _DEAD_ENDPOINT_STATUSES:
            raise RuntimeError(
                f"Modal Endpoint {endpoint_name!r} is {status}. "
                "Choose another name or remove the failed endpoint."
            )
    else:
        command = [
            sys.executable,
            "-m",
            "modal",
            "endpoint",
            "create",
            "--name",
            endpoint_name,
            "--model",
            model,
        ]
        if env:
            command.extend(["--env", env])
        if unauthenticated:
            command.append("--unauthenticated")
        if routing_region:
            command.extend(["--routing-region", routing_region])
        if custom_hf_repo:
            command.extend(["--custom-hf-repo", custom_hf_repo])
        if custom_hf_revision:
            command.extend(["--custom-hf-revision", custom_hf_revision])
        if custom_volume_name:
            command.extend(["--custom-volume-name", custom_volume_name])
        if volume_path is not None:
            command.extend(["--custom-volume-path", volume_path])
        subprocess.run(command, check=True, timeout=120)

    return _wait_for_endpoint_url(
        endpoint_name,
        env=env,
        proxy_auth=not unauthenticated,
        timeout_sec=wait_timeout_sec,
    )


def wait_for_server_url(
    server: Any,
    *,
    timeout_sec: float = 15 * 60,
    label: str = "Server",
    proxy_auth: bool = False,
) -> str:
    """Wait for a Modal server URL and its OpenAI-compatible model route."""
    deadline = time.monotonic() + timeout_sec
    url: str | None = None
    while time.monotonic() < deadline:
        try:
            raw = server.get_url()
        except modal.exception.NotFoundError:
            raw = None
        if raw:
            url = raw.rstrip("/")
            break
        time.sleep(1)
    else:
        raise TimeoutError(f"Timed out waiting for a URL from {label}")

    headers: dict[str, str] = {}
    if proxy_auth:
        headers = proxy_auth_headers(url)
        if not headers:
            raise TrainingGymConfigError(
                "Proxy authentication requires an HTTPS Modal URL and "
                "MODAL_KEY and MODAL_SECRET."
            )
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                f"{url}/v1/models",
                headers=headers,
                timeout=30,
                follow_redirects=False,
            )
            if response.status_code == 200:
                return url
            if 300 <= response.status_code < 400:
                raise RuntimeError(
                    f"{label} readiness redirected to "
                    f"{response.headers.get('location', '<unknown>')!r}"
                )
            if response.status_code in {401, 403}:
                if not proxy_auth:
                    raise RuntimeError(
                        f"{label} returned HTTP {response.status_code} without "
                        "proxy credentials. Retry with proxy_auth=True if it uses "
                        "Modal proxy authentication."
                    )
                raise RuntimeError(
                    f"{label} rejected proxy authentication. "
                    "Run `training-gym set-proxy-auth` and retry."
                )
            if response.status_code not in {404, 429, 500, 502, 503, 504}:
                response.raise_for_status()
            last_error = RuntimeError(
                f"{label} readiness returned HTTP {response.status_code}"
            )
        except httpx.RequestError as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(
        f"Timed out waiting for {label} at {url} to become ready"
    ) from last_error


def _wait_for_endpoint_url(
    name: str,
    *,
    env: str | None = None,
    timeout_sec: float = 15 * 60,
    proxy_auth: bool = False,
) -> str:
    return wait_for_server_url(
        modal.Server.from_name(f"ep-{name}", "Server", environment_name=env),
        timeout_sec=timeout_sec,
        label=f"Modal Endpoint {name!r}",
        proxy_auth=proxy_auth,
    )


def endpoint_chat(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int = 120,
    max_attempts: int = 4,
    extra_body: dict[str, Any] | None = None,
    proxy_auth: bool = False,
    **kwargs: Any,
) -> str:
    """Send a chat-completions request and return the assistant text."""
    message = endpoint_chat_message(
        base_url,
        model=model,
        messages=messages,
        timeout=timeout,
        max_attempts=max_attempts,
        extra_body=extra_body,
        proxy_auth=proxy_auth,
        **kwargs,
    )
    return str(message.get("content") or message.get("reasoning_content") or "")


def endpoint_chat_message(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int = 120,
    max_attempts: int = 4,
    extra_body: dict[str, Any] | None = None,
    proxy_auth: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Send a chat-completions request and return the assistant message.

    Set ``proxy_auth=True`` for endpoints protected by Modal proxy
    authentication. Transient server and rate-limit responses are retried up
    to *max_attempts*.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body: dict[str, Any] = {"model": model, "messages": messages, **kwargs}
    if extra_body:
        body.update(extra_body)
    headers = {"Content-Type": "application/json"}
    if proxy_auth:
        proxy_headers = proxy_auth_headers(base_url)
        if not proxy_headers:
            raise TrainingGymConfigError(
                "Proxy authentication requires an HTTPS Modal URL and "
                "MODAL_KEY and MODAL_SECRET."
            )
        headers.update(proxy_headers)
    transient = {429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        try:
            resp = httpx.post(
                url,
                json=body,
                timeout=timeout,
                headers=headers,
                follow_redirects=False,
            )
            if 300 <= resp.status_code < 400:
                raise RuntimeError(
                    f"Modal Endpoint redirected to "
                    f"{resp.headers.get('location', '<unknown>')!r}"
                )
            if resp.status_code in transient and attempt < max_attempts:
                time.sleep(min(2 * attempt, 5))
                continue
            if resp.status_code in {401, 403}:
                if not proxy_auth:
                    raise RuntimeError(
                        f"HTTP {resp.status_code} from {url}. Retry with "
                        "proxy_auth=True if the endpoint uses Modal proxy "
                        "authentication."
                    )
                raise RuntimeError(
                    f"HTTP {resp.status_code} from {url}. Proxy credentials were "
                    "rejected. Refresh them with `modal workspace proxy-tokens "
                    "create` and export MODAL_KEY and MODAL_SECRET."
                )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]
        except httpx.RequestError:
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 * attempt, 5))

    raise RuntimeError(f"Chat completions exhausted {max_attempts} attempts at {url}")
