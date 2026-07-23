from __future__ import annotations

import base64

import httpx
import pytest

from modal_training_gym.cli import client as client_module
from modal_training_gym.cli.client import (
    DEFAULT_TIMEOUT_SECONDS,
    DashboardClient,
)
from modal_training_gym.cli.errors import (
    AuthenticationError,
    DashboardConfigurationError,
    DashboardError,
    DashboardNetworkError,
    DashboardServerError,
    DashboardTimeoutError,
    MalformedResponseError,
    ResourceNotFoundError,
)


def test_uses_configured_url_and_encodes_query(monkeypatch):
    monkeypatch.setattr(
        client_module, "get_dashboard_url", lambda: "https://example.test/root/"
    )
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    with DashboardClient(transport=httpx.MockTransport(respond)) as client:
        result = client.get_json(
            "/api/items",
            params={"model": "a b", "limit": 2, "unused": None},
        )

    assert result == {"ok": True}
    assert str(seen[0].url) == ("https://example.test/root/api/items?model=a+b&limit=2")
    assert seen[0].extensions["timeout"]["read"] == DEFAULT_TIMEOUT_SECONDS


def test_sends_basic_auth_when_password_exists(monkeypatch):
    monkeypatch.setenv("TRAINING_GYM_DASHBOARD_PASSWORD", "secret")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    with DashboardClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(respond),
    ) as client:
        client.get_json("/api/items")

    expected = base64.b64encode(b"training-gym:secret").decode()
    assert requests[0].headers["authorization"] == f"Basic {expected}"


def test_omits_auth_when_password_is_absent(monkeypatch):
    monkeypatch.delenv("TRAINING_GYM_DASHBOARD_PASSWORD", raising=False)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    with DashboardClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(respond),
    ) as client:
        client.get_json("/api/items")

    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize(
    ("status_code", "error_type", "exit_code"),
    [
        (401, AuthenticationError, 4),
        (403, AuthenticationError, 4),
        (404, ResourceNotFoundError, 3),
        (500, DashboardServerError, 5),
        (503, DashboardServerError, 5),
        (400, DashboardError, 5),
    ],
)
def test_maps_http_errors(status_code, error_type, exit_code):
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status_code, text="ignored")
    )

    with DashboardClient(
        base_url="https://example.test", transport=transport
    ) as client:
        with pytest.raises(error_type) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.exit_code == exit_code


def test_maps_timeout_without_leaking_transport_details():
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret transport detail", request=request)

    with DashboardClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(timeout),
    ) as client:
        with pytest.raises(DashboardTimeoutError) as exc_info:
            client.get_json("/api/items")

    assert "secret transport detail" not in str(exc_info.value)


def test_maps_connection_failure():
    def disconnect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("host detail", request=request)

    with DashboardClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(disconnect),
    ) as client:
        with pytest.raises(DashboardNetworkError):
            client.get_json("/api/items")


def test_rejects_malformed_json():
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, text="<html>not json</html>")
    )

    with DashboardClient(
        base_url="https://example.test", transport=transport
    ) as client:
        with pytest.raises(MalformedResponseError):
            client.get_json("/api/items")


@pytest.mark.parametrize(
    "url",
    [None, "", "not-a-url", "ftp://example.test"],
)
def test_rejects_missing_or_invalid_configuration(monkeypatch, url):
    monkeypatch.setattr(client_module, "get_dashboard_url", lambda: url)

    with pytest.raises(DashboardConfigurationError):
        DashboardClient()


def test_rejects_absolute_request_path():
    with DashboardClient(base_url="https://example.test") as client:
        with pytest.raises(DashboardError):
            client.get_json("https://other.test/api/items")
