"""Small, TTY-aware output helpers for CLI commands."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Column, Table
from rich.text import Text


RICH_OUTPUT_ENV = "TRAINING_GYM_RICH_CLI"
FALLBACK_OUTPUT_WIDTH = 120


def use_rich_output() -> bool:
    """Return whether interactive color and styling should be enabled."""
    override = os.environ.get(RICH_OUTPUT_ENV)
    if override in {"0", "1"}:
        return override == "1"
    return sys.stdout.isatty()


def _console(*, stderr: bool = False) -> Console:
    width = shutil.get_terminal_size(fallback=(FALLBACK_OUTPUT_WIDTH, 24)).columns
    rich_enabled = use_rich_output()
    return Console(
        stderr=stderr,
        highlight=False,
        force_jupyter=False,
        force_terminal=rich_enabled,
        color_system="standard" if rich_enabled else None,
        width=width,
    )


def print_json(value: Any) -> None:
    """Write JSON to stdout (without Rich formatting)."""
    print(
        json.dumps(value, ensure_ascii=False, indent=2),
        file=sys.stdout,
    )


def print_table(
    columns: Sequence[Column | str],
    rows: Sequence[Sequence[object]],
    *,
    title: str = "",
) -> None:
    """Render a compact human-readable table to stdout."""
    table = Table(
        *columns,
        title=title or None,
    )
    for row in rows:
        cells = [
            cell if cell is None or isinstance(cell, (str, Text)) else str(cell)
            for cell in row
        ]
        table.add_row(*cells)
    _console().print(table)


def print_error(message: str) -> None:
    """Write an error to stderr when Rich output is enabled."""
    console = _console(stderr=True)
    if use_rich_output():
        console.print(
            Panel(
                Text(message),
                title="Error",
                title_align="left",
                border_style="red",
                expand=True,
            )
        )
    else:
        console.print(message, markup=False)
