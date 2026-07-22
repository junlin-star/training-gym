from __future__ import annotations

from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner
from rich.table import Column

from modal_training_gym.cli import options, output


def test_print_json_is_deterministic_plain_stdout(capsys):
    output.print_json({"z": "café", "a": [2, 1]})

    captured = capsys.readouterr()
    assert captured.out == ('{\n  "a": [\n    2,\n    1\n  ],\n  "z": "café"\n}\n')
    assert captured.err == ""


def test_print_table_is_plain_when_rich_is_disabled(monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_GYM_RICH_CLI", "0")

    output.print_table(
        [Column("Name"), "Count"],
        [["first", 2], ["second", None]],
        title="Items",
    )

    captured = capsys.readouterr()
    assert "Items" in captured.out
    assert "Name" in captured.out
    assert "first" in captured.out
    assert "2" in captured.out
    assert "\x1b[" not in captured.out
    assert captured.err == ""


def test_rich_output_can_be_forced(monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_GYM_RICH_CLI", "1")

    output.print_table(["Name"], [["first"]])

    assert "\x1b[" in capsys.readouterr().out


def test_print_error_uses_stderr(monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_GYM_RICH_CLI", "0")

    output.print_error("problem")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "problem\n"


def test_print_error_uses_modal_style_when_rich_is_enabled(monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_GYM_RICH_CLI", "1")

    output.print_error("problem")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error" in captured.err
    assert "problem" in captured.err
    assert "╭" in captured.err


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
