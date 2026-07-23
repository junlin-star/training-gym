from __future__ import annotations

import os
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner
from rich.table import Column

from modal_training_gym.cli import options, output


def test_console_adapts_to_terminal_width(monkeypatch):
    monkeypatch.setattr(
        output.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((180, 24)),
    )

    assert output._console().width == 180


def test_print_error_uses_stderr(monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_GYM_RICH_CLI", "0")

    output.print_error("problem")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "problem\n"


def test_use_rich_output_honors_override(monkeypatch):
    monkeypatch.setenv("TRAINING_GYM_RICH_CLI", "1")
    assert output.use_rich_output() is True
    monkeypatch.setenv("TRAINING_GYM_RICH_CLI", "0")
    assert output.use_rich_output() is False


def test_json_option_uses_non_shadowing_parameter_name():
    @click.command()
    @options.json_option
    def command(json_output):
        click.echo(str(json_output))

    runner = CliRunner()
    assert runner.invoke(command, ["-j"]).stdout == "True\n"
    assert runner.invoke(command, ["--json"]).stdout == "True\n"


def test_yes_option_supports_short_and_long_flags():
    @click.command()
    @options.yes_option
    def command(yes):
        click.echo(str(yes))

    runner = CliRunner()
    assert runner.invoke(command, ["-y"]).stdout == "True\n"
    assert runner.invoke(command, ["--yes"]).stdout == "True\n"


def test_confirmation_requires_yes_without_tty(monkeypatch):
    stdin = Mock()
    stdin.isatty.return_value = False
    monkeypatch.setattr(options.sys, "stdin", stdin)

    with pytest.raises(click.UsageError, match="rerun with --yes"):
        options.confirm_or_require_yes("Proceed?")


def test_confirmation_uses_click_with_tty(monkeypatch):
    stdin = Mock()
    stdin.isatty.return_value = True
    confirm = Mock()
    monkeypatch.setattr(options.sys, "stdin", stdin)
    monkeypatch.setattr(options.click, "confirm", confirm)

    options.confirm_or_require_yes("Proceed?")

    confirm.assert_called_once_with("Proceed?", default=False, abort=True)
