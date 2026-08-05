from __future__ import annotations

import stat
from types import SimpleNamespace
from typing import Any

import pytest

from modal_training_gym.common import config as local_config
from modal_training_gym.common.checkpoint import to_volume_path
from modal_training_gym.common.endpoint import (
    endpoint_chat_message,
    ensure_endpoint,
    is_modal_host,
    proxy_auth_headers,
    wait_for_server_url,
)
from modal_training_gym.common.errors import TrainingGymConfigError


@pytest.mark.parametrize(
    ("checkpoint_dir", "mount_path", "expected"),
    [
        ("/checkpoints/run-1/iter_10_hf", "/checkpoints", "run-1/iter_10_hf"),
        ("/checkpoints/run-1/iter_10_hf/", "/checkpoints", "run-1/iter_10_hf"),
        ("/checkpoints", "/checkpoints", ""),
        ("run-1/iter_10_hf", "/checkpoints", "run-1/iter_10_hf"),
        ("/data/ckpt/iter_5_hf", "/data/ckpt", "iter_5_hf"),
        ("/checkpoints/run-1/iter_2_hf", None, "run-1/iter_2_hf"),
    ],
)
def test_to_volume_path_strips_mount_prefix(
    checkpoint_dir: str, mount_path: str | None, expected: str
) -> None:
    if mount_path is None:
        assert to_volume_path(checkpoint_dir) == expected
    else:
        assert to_volume_path(checkpoint_dir, mount_path) == expected


@pytest.mark.parametrize(
    ("checkpoint_dir", "mount_path"),
    [
        ("/elsewhere/iter_5_hf", "/checkpoints"),
        ("/checkpointsX/iter_5_hf", "/checkpoints"),
        ("/data/ckpt/iter_5_hf", "/checkpoints"),
    ],
)
def test_to_volume_path_rejects_paths_outside_the_mount(
    checkpoint_dir: str, mount_path: str
) -> None:
    with pytest.raises(TrainingGymConfigError):
        to_volume_path(checkpoint_dir, mount_path)


@pytest.mark.parametrize(
    ("base_url", "modal_host"),
    [
        ("https://ws--ep-abc.modal.run", True),
        ("https://ws--ep-abc.modal.run/v1/chat/completions", True),
        ("https://something.modal.host", True),
        ("https://gym.modal.dev", True),
        ("https://evil.example.com", False),
        ("https://modal.run.evil.com", False),
        ("https://my-modal.dev.attacker.net", False),
        ("http://localhost:8000", False),
        ("http://gym.modal.dev", False),
    ],
)
def test_proxy_auth_headers_only_for_modal_https_hosts(
    base_url: str, modal_host: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODAL_KEY", "wk-test")
    monkeypatch.setenv("MODAL_SECRET", "ws-test")

    assert is_modal_host(base_url) is modal_host
    if modal_host:
        assert proxy_auth_headers(base_url) == {
            "Modal-Key": "wk-test",
            "Modal-Secret": "ws-test",
        }
    else:
        assert proxy_auth_headers(base_url) == {}


def test_proxy_auth_headers_empty_without_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.modal_proxy_auth_headers", lambda: {}
    )

    assert proxy_auth_headers("https://ws--ep-abc.modal.run") == {}


def test_wait_for_server_url_only_attaches_proxy_headers_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.httpx.get",
        lambda *_, **kwargs: requests.append(kwargs) or _FakeResponse(),
    )
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.proxy_auth_headers",
        lambda _: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    server = SimpleNamespace(get_url=lambda: "https://ws--ep.modal.run")

    wait_for_server_url(server, timeout_sec=1)
    wait_for_server_url(server, timeout_sec=1, proxy_auth=True)

    assert requests[0]["headers"] == {}
    assert requests[1]["headers"]["Modal-Key"] == "wk"


@pytest.mark.parametrize(
    ("proxy_auth", "expected"),
    [
        (False, "proxy_auth=True"),
        (True, "rejected proxy authentication"),
    ],
)
def test_wait_for_server_url_distinguishes_unauthenticated_requests(
    monkeypatch: pytest.MonkeyPatch,
    proxy_auth: bool,
    expected: str,
) -> None:
    response = _FakeResponse()
    response.status_code = 401
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.httpx.get",
        lambda *_, **__: response,
    )
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.proxy_auth_headers",
        lambda _: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    server = SimpleNamespace(get_url=lambda: "https://ws--ep.modal.run")

    with pytest.raises(RuntimeError, match=expected):
        wait_for_server_url(server, timeout_sec=1, proxy_auth=proxy_auth)


def test_save_proxy_auth_creates_private_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config_path = tmp_path / ".training-gym.toml"
    monkeypatch.setattr(local_config, "CONFIG_PATH", config_path)
    real_fchmod = local_config.os.fchmod
    mode_before_fchmod: int | None = None

    def capture_mode(fd: int, mode: int) -> None:
        nonlocal mode_before_fchmod
        mode_before_fchmod = stat.S_IMODE(local_config.os.fstat(fd).st_mode)
        real_fchmod(fd, mode)

    monkeypatch.setattr(local_config.os, "fchmod", capture_mode)
    previous_umask = local_config.os.umask(0o022)
    try:
        local_config.save_proxy_auth("wk-test", "ws-test")
    finally:
        local_config.os.umask(previous_umask)

    assert mode_before_fchmod == 0o600


def test_save_proxy_auth_restricts_existing_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config_path = tmp_path / ".training-gym.toml"
    config_path.write_text('[dashboard]\nurl = "https://dashboard"\n')
    config_path.chmod(0o644)
    monkeypatch.setattr(local_config, "CONFIG_PATH", config_path)

    local_config.save_proxy_auth("wk-test", "ws-test")

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert local_config.get_proxy_auth() == ("wk-test", "ws-test")


class _FakeEndpointCli:
    def __init__(self, listed: list[dict[str, Any]]) -> None:
        self.listed = listed
        self.commands: list[list[str]] = []
        self.waits: list[tuple[str, str | None, bool]] = []

    def run(self, command: list[str], **_: Any) -> None:
        self.commands.append(command)


class _FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}]}


@pytest.fixture
def fake_endpoint_cli(monkeypatch: pytest.MonkeyPatch):
    def _install(listed: list[dict[str, Any]]) -> _FakeEndpointCli:
        cli = _FakeEndpointCli(listed)
        monkeypatch.setattr(
            "modal_training_gym.common.endpoint._list_endpoints",
            lambda **_: cli.listed,
        )
        monkeypatch.setattr(
            "modal_training_gym.common.endpoint.subprocess.run", cli.run
        )
        monkeypatch.setattr(
            "modal_training_gym.common.endpoint.load_proxy_auth", lambda: True
        )
        monkeypatch.setattr(
            "modal_training_gym.common.endpoint.proxy_auth_headers", lambda _: {}
        )

        def _wait(name: str, *, env=None, proxy_auth=False, **_: Any) -> str:
            cli.waits.append((name, env, proxy_auth))
            return f"https://ws--ep-{name}.modal.run"

        monkeypatch.setattr(
            "modal_training_gym.common.endpoint._wait_for_endpoint_url",
            _wait,
        )
        monkeypatch.setattr(
            "modal_training_gym.common.endpoint.httpx.get",
            lambda *_, **__: _FakeResponse(),
        )
        return cli

    return _install


@pytest.mark.parametrize(
    ("custom_volume_path", "mount_path", "expected_rel"),
    [
        ("/checkpoints/run-1/iter_10_hf", None, "run-1/iter_10_hf"),
        ("/data/ckpt/run-1/iter_10_hf", "/data/ckpt", "run-1/iter_10_hf"),
        ("/checkpoints", None, "/"),
    ],
)
def test_ensure_endpoint_converts_volume_path_and_defaults_unauthenticated(
    fake_endpoint_cli,
    custom_volume_path: str,
    mount_path: str | None,
    expected_rel: str,
) -> None:
    cli = fake_endpoint_cli([])
    kwargs: dict[str, Any] = {
        "name": "my-ft",
        "model": "Qwen/Qwen3-4B",
        "custom_volume_name": "gym-checkpoints",
        "custom_volume_path": custom_volume_path,
    }
    if mount_path is not None:
        kwargs["custom_volume_mount_path"] = mount_path

    url = ensure_endpoint(**kwargs)

    assert url.startswith("https://ws--ep-my-ft-")
    (command,) = cli.commands
    assert command[command.index("--custom-volume-path") + 1] == expected_rel
    assert "--unauthenticated" in command
    assert cli.waits[-1][2] is False


def test_ensure_endpoint_rejects_empty_volume_path(fake_endpoint_cli) -> None:
    fake_endpoint_cli([])

    with pytest.raises(TrainingGymConfigError, match="requires custom_volume_path"):
        ensure_endpoint(
            name="my-ft",
            model="Qwen/Qwen3-4B",
            custom_volume_name="gym-checkpoints",
            custom_volume_path="",
        )


def test_ensure_endpoint_reuses_live_endpoint_without_creating(
    fake_endpoint_cli,
) -> None:
    cli = fake_endpoint_cli([])
    ensure_endpoint(name="my-ft", model="Qwen/Qwen3-4B")
    endpoint_name = cli.waits[-1][0]
    cli.commands.clear()
    cli.listed = [{"name": endpoint_name, "status": "running"}]

    url = ensure_endpoint(name="my-ft", model="Qwen/Qwen3-4B")

    assert url == f"https://ws--ep-{endpoint_name}.modal.run"
    assert cli.commands == []


def test_ensure_endpoint_names_include_auth_mode(fake_endpoint_cli) -> None:
    cli = fake_endpoint_cli([])
    ensure_endpoint(name="my-ft", model="Qwen/Qwen3-4B")
    public_name = cli.waits[-1][0]
    assert cli.waits[-1][2] is False
    assert "--unauthenticated" in cli.commands[-1]

    ensure_endpoint(name="my-ft", model="Qwen/Qwen3-4B", unauthenticated=False)
    authenticated_name = cli.waits[-1][0]
    assert cli.waits[-1][2] is True

    assert public_name != authenticated_name
    assert "--unauthenticated" not in cli.commands[-1]


def test_ensure_endpoint_requires_proxy_credentials_when_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.load_proxy_auth", lambda: False
    )

    with pytest.raises(TrainingGymConfigError, match="MODAL_KEY"):
        ensure_endpoint(name="my-ft", model="Qwen/Qwen3-4B", unauthenticated=False)


def test_endpoint_chat_attaches_proxy_headers_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def _post(*_: Any, **kwargs: Any) -> _FakeResponse:
        requests.append(kwargs)
        return _FakeResponse()

    monkeypatch.setattr("modal_training_gym.common.endpoint.httpx.post", _post)
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.proxy_auth_headers",
        lambda _: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )

    endpoint_chat_message(
        "https://ws--ep.modal.run",
        model="model",
        messages=[],
    )
    assert requests[0]["headers"] == {"Content-Type": "application/json"}

    endpoint_chat_message(
        "https://ws--ep.modal.run",
        model="model",
        messages=[],
        proxy_auth=True,
    )
    assert requests[1]["headers"]["Modal-Key"] == "wk"

    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.proxy_auth_headers", lambda _: {}
    )
    with pytest.raises(TrainingGymConfigError, match="Proxy authentication requires"):
        endpoint_chat_message(
            "https://ws--ep.modal.run",
            model="model",
            messages=[],
            proxy_auth=True,
        )


@pytest.mark.parametrize(
    ("proxy_auth", "expected"),
    [
        (False, "proxy_auth=True"),
        (True, "Proxy credentials were rejected"),
    ],
)
@pytest.mark.parametrize("status_code", [401, 403])
def test_endpoint_chat_auth_error_distinguishes_unauthenticated_requests(
    monkeypatch: pytest.MonkeyPatch,
    proxy_auth: bool,
    expected: str,
    status_code: int,
) -> None:
    response = _FakeResponse()
    response.status_code = status_code
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.httpx.post",
        lambda *_, **__: response,
    )
    monkeypatch.setattr(
        "modal_training_gym.common.endpoint.proxy_auth_headers",
        lambda _: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )

    with pytest.raises(RuntimeError, match=expected):
        endpoint_chat_message(
            "https://ws--ep.modal.run",
            model="model",
            messages=[],
            proxy_auth=proxy_auth,
        )
