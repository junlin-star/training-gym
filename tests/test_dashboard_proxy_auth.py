from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from modal_training_gym.cli import setup as cli_setup_module
from modal_training_gym.common import config
from modal_training_gym.common import status_reporter
from modal_training_gym.frameworks.slime import reporting


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b""


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / ".training-gym.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    return path


def test_dashboard_proxy_auth_mode_is_persisted(config_path):
    config.save_dashboard_url("https://dashboard.test", proxy_auth=True)

    assert config.get_dashboard_url() == "https://dashboard.test"
    assert config.get_dashboard_proxy_auth() is True
    assert "proxy_auth = true" in config_path.read_text()


def test_unspecified_mode_defaults_to_no_proxy_auth(config_path):
    assert cli_setup_module._resolve_dashboard_proxy_auth(None) is False


def test_unspecified_mode_requires_choice_after_proxy_auth(config_path):
    config.save_dashboard_url("https://dashboard.test", proxy_auth=True)

    with pytest.raises(cli_setup_module.ProxyAuthChoiceRequired):
        cli_setup_module._resolve_dashboard_proxy_auth(None)


def test_dashboard_import_sets_proxy_auth_mode(monkeypatch):
    dashboard = SimpleNamespace()
    observed = []

    def reload_module(module):
        observed.append(config.dashboard_requires_proxy_auth())
        return module

    monkeypatch.setattr(config, "_dashboard_requires_proxy_auth", False)
    monkeypatch.setitem(sys.modules, "modal_training_gym._dashboard", dashboard)
    monkeypatch.setattr(importlib, "reload", reload_module)

    loaded = cli_setup_module._load_dashboard_for_deploy(True)

    assert loaded is dashboard
    assert observed == [True]
    assert config.dashboard_requires_proxy_auth() is True


def _capture_report(reporter, monkeypatch):
    requests = []

    def urlopen(request, **_kwargs):
        requests.append(request)
        return _Response()

    monkeypatch.setattr(reporter, "urlopen", urlopen)
    return requests


def test_status_reporting_posts_include_proxy_auth_headers(config_path, monkeypatch):
    config.save_proxy_auth("wk-test", "ws-test")
    monkeypatch.delenv("MODAL_KEY", raising=False)
    monkeypatch.delenv("MODAL_SECRET", raising=False)
    requests = _capture_report(status_reporter, monkeypatch)
    status_reporter._post(
        {
            "_url": "https://dashboard.test/api/framework-status",
            "_timeout": 1,
            "_token": "run-token",
            "training_run_id": "run-1",
            "phase": "training",
        }
    )

    headers = dict(requests[0].header_items())
    assert headers["Modal-key"] == "wk-test"
    assert headers["Modal-secret"] == "ws-test"
    assert headers["Authorization"] == "Bearer run-token"


def test_slime_reporting_posts_include_proxy_auth_headers(config_path, monkeypatch):
    config.save_proxy_auth("wk-test", "ws-test")
    monkeypatch.delenv("MODAL_KEY", raising=False)
    monkeypatch.delenv("MODAL_SECRET", raising=False)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_TOKEN", "run-token")
    requests = _capture_report(reporting, monkeypatch)
    reporting._post(
        {
            "_url": "https://dashboard.test/api/framework-status",
            "_timeout": 1,
            "training_run_id": "run-1",
            "phase": "training",
        }
    )

    headers = dict(requests[0].header_items())
    assert headers["Modal-key"] == "wk-test"
    assert headers["Modal-secret"] == "ws-test"
    assert headers["Authorization"] == "Bearer run-token"
