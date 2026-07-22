from __future__ import annotations

from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from modal_training_gym import cli as cli_module
from modal_training_gym.cli.errors import (
    AuthenticationError,
    DashboardError,
    ResourceNotFoundError,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_root_help_lists_existing_commands_by_panel(runner):
    result = runner.invoke(cli_module.cli, ["--help"])

    assert result.exit_code == 0
    assert "Configuration:" in result.stdout
    assert "Utilities:" in result.stdout
    for command in (
        "setup",
        "open",
        "set-password",
        "set-proxy-auth",
        "cleanup",
    ):
        assert command in result.stdout


def test_root_supports_short_help(runner):
    result = runner.invoke(cli_module.cli, ["-h"])

    assert result.exit_code == 0
    assert result.stdout.startswith("Usage:")


def test_root_without_command_shows_help(runner):
    result = runner.invoke(cli_module.cli, [])

    assert result.exit_code == 2
    assert "Usage:" in result.stderr


def test_main_returns_help_exit_code(capsys):
    assert cli_module.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out
    assert captured.err == ""


def test_main_returns_no_command_exit_code(capsys):
    assert cli_module.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Usage:" in captured.err


def test_setup_dispatches_to_existing_function(runner, monkeypatch):
    setup = Mock()
    monkeypatch.setattr("modal_training_gym.cli.setup.setup", setup)

    result = runner.invoke(cli_module.cli, ["setup"])

    assert result.exit_code == 0
    assert result.stderr == ""
    setup.assert_called_once_with()


def test_open_dispatches_to_existing_function(runner, monkeypatch):
    open_dashboard = Mock()
    monkeypatch.setattr("modal_training_gym.cli.setup.open_dashboard", open_dashboard)

    result = runner.invoke(cli_module.cli, ["open"])

    assert result.exit_code == 0
    open_dashboard.assert_called_once_with()


def test_set_proxy_auth_dispatches_to_existing_function(runner, monkeypatch):
    set_proxy_auth = Mock()
    monkeypatch.setattr("modal_training_gym.cli.setup.set_proxy_auth", set_proxy_auth)

    result = runner.invoke(cli_module.cli, ["set-proxy-auth"])

    assert result.exit_code == 0
    set_proxy_auth.assert_called_once_with()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["set-password"], None),
        (["set-password", "--password", "secret"], "secret"),
        (["set-password", "--password", ""], ""),
    ],
)
def test_set_password_preserves_arguments(runner, monkeypatch, args, expected):
    set_password = Mock()
    monkeypatch.setattr("modal_training_gym.cli.setup.set_password", set_password)

    result = runner.invoke(cli_module.cli, args)

    assert result.exit_code == 0
    set_password.assert_called_once_with(password=expected)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["cleanup"], {"older_than_days": 7, "dry_run": False}),
        (
            ["cleanup", "--older-than-days", "3", "--dry-run"],
            {"older_than_days": 3, "dry_run": True},
        ),
    ],
)
def test_cleanup_preserves_arguments(runner, monkeypatch, args, expected):
    cleanup = Mock()
    monkeypatch.setattr("modal_training_gym.cli.cleanup.cleanup", cleanup)

    result = runner.invoke(cli_module.cli, args)

    assert result.exit_code == 0
    cleanup.assert_called_once_with(**expected)


def test_click_usage_errors_use_exit_two_and_stderr(runner):
    result = runner.invoke(cli_module.cli, ["cleanup", "--unknown"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "No such option '--unknown'" in result.stderr


def test_click_usage_errors_use_modal_style_when_rich_is_enabled(runner):
    result = runner.invoke(
        cli_module.cli,
        ["cleanup", "--unknown"],
        env={"TRAINING_GYM_RICH_CLI": "1"},
        color=True,
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error" in result.stderr
    assert "No such option" in result.stderr
    assert "╭" in result.stderr


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (ResourceNotFoundError("missing"), 3),
        (AuthenticationError("denied"), 4),
        (DashboardError("offline"), 5),
    ],
)
def test_typed_errors_use_stable_exit_codes(runner, monkeypatch, error, exit_code):
    def fail():
        raise error

    monkeypatch.setattr("modal_training_gym.cli.setup.setup", fail)

    result = runner.invoke(cli_module.cli, ["setup"])

    assert result.exit_code == exit_code
    assert result.stdout == ""
    assert str(error) in result.stderr


def test_main_maps_unexpected_errors(monkeypatch, capsys):
    def fail(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module.cli, "main", fail)

    assert cli_module.main(["setup"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: boom\n"


def test_main_maps_interrupts(monkeypatch, capsys):
    def interrupt(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module.cli, "main", interrupt)

    assert cli_module.main(["setup"]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Interrupted.\n"
