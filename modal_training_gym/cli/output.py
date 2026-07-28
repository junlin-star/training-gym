"""Small, TTY-aware output helpers for CLI commands."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Column, Table
from rich.text import Text


FALLBACK_OUTPUT_WIDTH = 120


def _console(*, stderr: bool = False) -> Console:
    width = shutil.get_terminal_size(fallback=(FALLBACK_OUTPUT_WIDTH, 24)).columns
    rich_enabled = sys.stdout.isatty()
    return Console(
        stderr=stderr,
        highlight=False,
        force_jupyter=False,
        force_terminal=rich_enabled,
        color_system="standard" if rich_enabled else None,
        width=width,
    )


def print_json(value: Any, *, stderr: bool = False) -> None:
    """Write JSON without Rich formatting."""
    print(
        json.dumps(value, ensure_ascii=False, indent=2),
        file=sys.stderr if stderr else sys.stdout,
    )


def print_table(
    columns: Sequence[Column | str],
    rows: Sequence[Sequence[object]],
    *,
    title: str = "",
    show_header: bool = True,
) -> None:
    table = Table(
        *columns,
        title=title or None,
        show_header=show_header,
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
    if sys.stdout.isatty():
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
