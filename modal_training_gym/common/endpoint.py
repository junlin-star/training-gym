from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from typing import Any

import httpx
import modal

from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.config import modal_proxy_auth_headers
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.model import ModelConfig


def _resolve_model_name(model: ModelConfig | str) -> str:
    return model if isinstance(model, str) else model.model_name


def _create_endpoint_and_wait_for_url(
    *,
    endpoint_name: str,
    model_name: str,
    checkpoint: Checkpoint | None,
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

    if checkpoint:
        command.extend(["--custom-volume-name", checkpoint.checkpoints_volume_name])
        command.extend(["--custom-volume-path", checkpoint.path_relative_to_volume])

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
    """A handle to a [Modal Endpoint](https://modal.com/docs/guide/endpoints).

    Modal Endpoints are managed LLM inference: a tuned open-source serving
    engine behind a low-latency request proxy, with scale-to-zero autoscaling
    and usage-based pricing, so an idle endpoint bills nothing. Modal picks a
    compatible serving recipe from the model, which can be a model from the
    [catalog](https://modal.com/endpoints) or your own fine-tune.

    Use ``Endpoint.launch()`` to provision one, or construct this class
    directly to talk to an endpoint that already exists.

    An endpoint serves the OpenAI Chat Completions API under ``/v1``, so
    ``chat()`` is a convenience rather than the only option — you can point the
    OpenAI SDK or any OpenAI-compatible client at ``f"{endpoint.url}/v1"``.

    Endpoints require a Modal
    [proxy token](https://modal.com/docs/guide/webhook-proxy-auth) pair unless
    launched with ``unauthenticated=True``. Create one with
    ``modal workspace proxy-tokens create``, which prints a token id
    (``wk-…``) and secret (``ws-…``); the secret is shown only once. Export
    them as ``MODAL_KEY`` / ``MODAL_SECRET`` or save them with
    ``training-gym set-proxy-auth``, and this class attaches them to every
    request. On RBAC-enabled workspaces the token also has to be granted
    access to the environment (``modal workspace proxy-tokens allow wk-… main``).

    Endpoints outlive the process that launched them. List them with
    ``modal endpoint list`` and tear one down — which stops its serving
    containers and their billing — with ``modal endpoint stop <name>``.

    ## Attributes

    url : str
        Base URL of the endpoint, with no trailing slash, so request paths
        such as ``/v1/chat/completions`` are appended to it.
    endpoint_name : str
        Modal endpoint name, as passed to ``modal endpoint create --name`` and
        shown by ``modal endpoint list``. The endpoint's Modal app is named
        ``ep-{endpoint_name}``.
    model_name : str
        Model id sent in request bodies. Custom weights are always served
        against a base model from the catalog, so for a checkpoint this stays
        the base model's repo id rather than the checkpoint path.
    requires_proxy_auth : bool
        Whether requests carry ``Modal-Key`` / ``Modal-Secret`` proxy-auth
        headers. False for endpoints launched with ``unauthenticated=True``.
    """

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
        checkpoint: Checkpoint | None = None,
        *,
        endpoint_name: str | None = None,
        unauthenticated: bool = False,
        environment: str | None = None,
        routing_region: str | None = None,
        wait_timeout_sec: float = 300,
    ):
        """Provision a Modal endpoint for ``model`` and return a handle to it.

        Shells out to ``modal endpoint create``; see the [Modal Endpoints
        guide](https://modal.com/docs/guide/endpoints) for the full set of
        options and the catalog of supported model families.

        ``model`` is a ``ModelConfig`` or a model repo id, and selects the
        serving recipe. Pass ``checkpoint`` to serve trained weights instead of
        the base model: Modal mounts the checkpoint's volume into the endpoint
        (``--custom-volume-name`` / ``--custom-volume-path``) while still
        serving them against ``model`` as the base. The checkpoint directory
        has to look like a HuggingFace model — it must contain ``config.json``,
        so convert Megatron checkpoints with ``convert_checkpoint_to_hf()``
        first.

        When ``endpoint_name`` is omitted, a stable name is derived by hashing
        the serving spec, so relaunching with the same model, checkpoint, and
        routing options reuses the existing endpoint instead of creating a
        second one. Endpoints require proxy auth unless
        ``unauthenticated=True``. ``environment`` defaults to the active Modal
        environment. ``routing_region`` anchors the request proxy nearest your
        callers — one of ``us-west`` (Modal's default), ``us-east``,
        ``ca-central``, ``eu-west``, or ``ap-south`` — and, like the model and
        checkpoint, is part of the derived name.

        Returns once the endpoint has a URL, which is well before it can serve
        traffic; call ``wait_until_ready()`` for that. Raises ``TimeoutError``
        if no URL is published within ``wait_timeout_sec``.
        """
        model_name = _resolve_model_name(model)

        if not endpoint_name:
            spec = {
                "model": model_name,
                "routing_region": routing_region,
                "unauthenticated": unauthenticated,
            }

            if checkpoint:
                spec.update(
                    {
                        "checkpoint_run": checkpoint.training_run_id,
                        "checkpoint_name": checkpoint.name,
                    }
                )

            digest = hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:12]
            endpoint_name = f"training-gym-{digest}"

        url = _create_endpoint_and_wait_for_url(
            endpoint_name=endpoint_name,
            model_name=model_name,
            checkpoint=checkpoint,
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
        """Block until the endpoint can serve traffic.

        Polls ``/v1/models`` and returns as soon as it answers. A fresh
        endpoint has to pull its image and load weights before it can respond,
        and because endpoints scale to zero, an idle one pays that cost again
        on its next request — loading a large checkpoint off a Volume can take
        tens of minutes, hence the generous default timeout.

        Raises ``TimeoutError`` if the endpoint is still not ready by then, and
        ``RuntimeError`` if the endpoint rejects the proxy credentials.
        """
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
        """POST one chat completion to ``/v1/chat/completions``.

        ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts, and
        ``extra_parameters`` carries any other body fields the OpenAI Chat
        Completions API accepts, such as ``temperature`` or ``max_tokens``.
        Returns the assistant message as a dict, preserving structured fields
        like ``tool_calls`` and ``reasoning_content``.

        Connection errors and the transient statuses an autoscaling endpoint
        returns (429 plus 5xx) are retried up to ``max_attempts`` times with a
        short backoff, while ``timeout`` bounds each individual request. Raises
        ``RuntimeError`` if the endpoint rejects the proxy credentials, and
        propagates the underlying ``httpx`` error when the last attempt fails.
        """
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
