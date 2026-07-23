"""Commands for inspecting training runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import click
from pydantic import ValidationError

from modal_training_gym.common.run_list import (
    RunListItem,
    build_run_list_item,
    run_list_field_metadata,
)
from modal_training_gym.common.run_summary import RunSummary
from modal_training_gym.common.time_utils import parse_time

from .client import DashboardClient
from .commands import _TrainingGymGroup
from .errors import MalformedResponseError
from .options import json_option
from .output import print_json, print_table


DEFAULT_RUN_LIMIT = 50


def parse_since(value: str, *, now: float | None = None) -> int:
    """Parse epoch, ISO 8601, or relative time into epoch seconds."""
    parsed = parse_time(value, time.time() if now is None else now)
    if parsed is None:
        raise click.BadParameter(
            "must be epoch seconds, ISO 8601, or a relative time such as 24h"
        )
    return int(parsed)


def _run_filter_options(function: Callable[..., Any]) -> Callable[..., Any]:
    """Generate Click filters from fields marked filterable in RunListItem."""
    for name, metadata in reversed(run_list_field_metadata().items()):
        if not metadata.get("filterable"):
            continue
        function = click.option(
            f"--{name.replace('_', '-')}",
            name,
            default=None,
            metavar=name.upper(),
            help=f"Only runs with this {str(metadata['label']).lower()}.",
        )(function)
    return function


def _format_timestamp(value: object) -> str:
    if not isinstance(value, (int, float)) or not value:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(value))


def _table_rows(items: list[RunListItem]) -> list[list[object]]:
    fields = run_list_field_metadata()
    rows: list[list[object]] = []
    for item in items:
        rows.append(
            [
                _format_timestamp(getattr(item, name))
                if metadata.get("timestamp")
                else getattr(item, name) or "—"
                for name, metadata in fields.items()
            ]
        )
    return rows


def list_runs(
    *,
    since: str | None,
    limit: int,
    json_output: bool,
    filters: dict[str, str | None],
) -> None:
    """Fetch, validate, and render the run list."""
    params: dict[str, str | int | None] = {
        **filters,
        "since": parse_since(since) if since else None,
        "limit": limit,
    }
    with DashboardClient() as client:
        payload = client.get_json("/api/runs", params=params)

    if not isinstance(payload, list):
        raise MalformedResponseError("Dashboard returned an invalid run list.")
    try:
        summaries = [RunSummary.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise MalformedResponseError(
            "Dashboard returned an invalid run summary."
        ) from exc
    items = [build_run_list_item(summary) for summary in summaries]

    if json_output:
        print_json([item.model_dump(mode="json") for item in items])
        return

    fields = run_list_field_metadata()
    print_table(
        [str(metadata["label"]) for metadata in fields.values()],
        _table_rows(items),
    )


@click.group("run", cls=_TrainingGymGroup)
def run_group() -> None:
    """Inspect and manage training runs."""


@run_group.command(
    "list",
    help=(
        "List training runs with their top-level metadata.\n\n"
        "Supports filtering on status, model, dataset, recipe, group, or by "
        "recency, all with a limit. Sorted by most recently updated."
    ),
    epilog=(
        "Examples:\n"
        "  training-gym run list --status failed --since 24h\n"
        "  training-gym run list --status completed "
        "--group nightly-tau-bench -j"
    ),
)
@_run_filter_options
@click.option(
    "--since",
    default=None,
    metavar="TIME",
    help="Only runs created or updated since this timestamp or relative time.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=DEFAULT_RUN_LIMIT,
    show_default=True,
    metavar="N",
    help="Maximum number of runs to return.",
)
@json_option
def list_command(
    *,
    since: str | None,
    limit: int,
    json_output: bool,
    **filters: str | None,
) -> None:
    """List training runs with their top-level metadata."""
    list_runs(
        since=since,
        limit=limit,
        json_output=json_output,
        filters=filters,
    )
