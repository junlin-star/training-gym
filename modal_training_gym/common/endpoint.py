from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from typing import Any

import httpx
import modal

from modal_training_gym.common.config import modal_proxy_auth_headers
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.model import ModelConfig


def _resolve_model_name(model: ModelConfig | str) -> str:
    return model if isinstance(model, str) else model.model_name


def _create_endpoint_and_wait_for_url(
    *,
    endpoint_name: str,
    model_name: str,
    unauthenticated: bool,
    environment: str | None,
    routing_region: str | None,
    wait_timeout_sec: float,
) -> str:
    command = [
        sys.executable,
        "-m",
        "modal",
        "endpoint",
        "create",
        "--name",
        endpoint_name,
        "--model",
        model_name,
    ]

    if environment:
        command.extend(["--env", environment])
    if unauthenticated:
        command.append("--unauthenticated")
    if routing_region:
        command.extend(["--routing-region", routing_region])

    subprocess.run(command, check=True, timeout=120)

    server = modal.Server.from_name(
        f"ep-{endpoint_name}", "Server", environment_name=environment
    )
    deadline = time.monotonic() + wait_timeout_sec
    while time.monotonic() < deadline:
        try:
            raw = server.get_url()
        except modal.exception.NotFoundError:
            raw = None
        if raw:
            return raw.rstrip("/")
        time.sleep(1)
    else:
        raise TimeoutError(
            f"Timed out waiting for a URL for endpoint {endpoint_name!r}"
        )


class Endpoint:
    url: str
    endpoint_name: str
    model_name: str
    requires_proxy_auth: bool

    def __init__(
        self,
        url: str,
        *,
        endpoint_name: str,
        model_name: str,
        requires_proxy_auth: bool,
    ):
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.url = url
        self.requires_proxy_auth = requires_proxy_auth

    @classmethod
    def launch(
        cls,
        model: ModelConfig | str,
        *,
        endpoint_name: str | None = None,
        unauthenticated: bool,
        environment: str | None = None,
        routing_region: str | None = None,
        wait_timeout_sec: float = 60,
    ):
        model_name = _resolve_model_name(model)

        if not endpoint_name:
            spec = {
                "model": model_name,
                "routing_region": routing_region,
                "unauthenticated": unauthenticated,
            }

            digest = hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:12]
            endpoint_name = f"training-gym-{digest}"

        url = _create_endpoint_and_wait_for_url(
            endpoint_name=endpoint_name,
            model_name=model_name,
            unauthenticated=unauthenticated,
            environment=environment,
            routing_region=routing_region,
            wait_timeout_sec=wait_timeout_sec,
        )

        return cls(
            url,
            endpoint_name=endpoint_name,
            model_name=model_name,
            requires_proxy_auth=not unauthenticated,
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.requires_proxy_auth:
            headers = modal_proxy_auth_headers()
            if not headers:
                raise TrainingGymConfigError(
                    "Proxy authentication requires an HTTPS Modal URL and "
                    "MODAL_KEY and MODAL_SECRET."
                )
        return headers

    def wait_until_ready(self, timeout_sec: float = 15 * 60) -> None:
        last_error: Exception | None = None
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"{self.url}/v1/models",
                    headers=self._headers(),
                    timeout=30,
                    follow_redirects=False,
                )
                if response.status_code == 200:
                    return
                if response.status_code in {401, 403}:
                    raise RuntimeError(
                        f"Endpoint {self.endpoint_name} rejected proxy authentication. "
                        "Run `training-gym set-proxy-auth` and retry."
                    )
                if response.status_code not in {404, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                last_error = RuntimeError(
                    f"Endpoint {self.endpoint_name} readiness returned HTTP {response.status_code}"
                )
            except httpx.RequestError as exc:
                last_error = exc
            time.sleep(2)
        raise TimeoutError(
            f"Timed out waiting for endpoint {self.endpoint_name} at {self.url} to become ready"
        ) from last_error

    def chat(
        self,
        messages: list[dict[str, Any]],
        timeout: int = 120,
        max_attempts: int = 4,
        extra_parameters: dict[str, Any] | None = None,
    ):
        url = f"{self.url}/v1/chat/completions"
        body: dict[str, Any] = {"model": self.model_name, "messages": messages}
        if extra_parameters:
            body.update(extra_parameters)

        headers = self._headers()
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
                if resp.status_code in transient and attempt < max_attempts:
                    time.sleep(min(2 * attempt, 5))
                    continue
                if resp.status_code in {401, 403}:
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

        raise RuntimeError(
            f"Chat completions exhausted {max_attempts} attempts at {url}"
        )
