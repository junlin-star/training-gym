"""CLI entry point: ``training-gym <command>``."""

from __future__ import annotations

import sys

import click

from .builtin import (
    cleanup_command,
    open_command,
    set_password_command,
    set_proxy_auth_command,
    setup_command,
)
from .commands import _TrainingGymGroup
from .errors import ExitCode
from .output import print_error
from .run import run_group


@click.group(
    cls=_TrainingGymGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def entrypoint_cli() -> None:
    """Launch, inspect, and manage training runs."""


def _register_commands() -> None:
    entrypoint_cli.add_command(run_group, panel="Training runs")
    entrypoint_cli.add_command(setup_command, panel="Configuration")
    entrypoint_cli.add_command(set_password_command, panel="Configuration")
    entrypoint_cli.add_command(set_proxy_auth_command, panel="Configuration")
    entrypoint_cli.add_command(open_command, panel="Utilities")
    entrypoint_cli.add_command(cleanup_command, panel="Utilities")


_register_commands()


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    try:
        result = entrypoint_cli.main(
            args=argv,
            prog_name="training-gym",
            standalone_mode=False,
        )
        return int(result) if isinstance(result, int) else int(ExitCode.SUCCESS)
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.Abort:
        print_error("Interrupted.")
        return int(ExitCode.INTERRUPTED)
    except Exception as exc:
        print_error(f"Error: {str(exc) or type(exc).__name__}")
        return int(ExitCode.UNEXPECTED)


if __name__ == "__main__":
    raise SystemExit(main())
