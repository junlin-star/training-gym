"""Commands for inspecting training runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import click
from pydantic import ValidationError

from modal_training_gym.common.run_list import run_list_field_metadata
from modal_training_gym.common.run_summary import RunSummary
from modal_training_gym.common.time import parse_time

from .client import DashboardClient
from .commands import _TrainingGymGroup
from .errors import CLIError, ExitCode
from .options import json_option
from .output import print_json, print_table


DEFAULT_RUN_LIMIT = 50
CLI_FIELD_NAMES = {
    "display_status": "status",
    "display_stage": "stage",
    "group_id": "group",
    "updated_at": "last_updated_at",
}


def _run_filter_options(function: Callable[..., Any]) -> Callable[..., Any]:
    """Generate Click filters from list fields marked filterable on RunSummary."""
    for name, metadata in reversed(run_list_field_metadata().items()):
        if not metadata.get("filterable"):
            continue
        option_name = CLI_FIELD_NAMES.get(name, name)
        function = click.option(
            f"--{option_name.replace('_', '-')}",
            name,
            default=None,
            metavar=option_name.upper(),
            help=f"Only runs with this {str(metadata['label']).lower()}.",
        )(function)
    return function


def _format_timestamp(value: object) -> str:
    if not isinstance(value, (int, float)) or not value:
        return "—"
    return (
        datetime.fromtimestamp(value, tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _format_table_timestamp(value: object, *, now: float | None = None) -> str:
    if not isinstance(value, (int, float)) or not value:
        return "—"
    age = max(0, (time.time() if now is None else now) - value)
    if age < 60:
        return "now"
    if age < 3_600:
        return f"{int(age // 60)}m ago"
    if age < 86_400:
        return f"{int(age // 3_600)}h ago"
    if age < 2_592_000:
        return f"{int(age // 86_400)}d ago"
    return datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%d")


def _table_rows(
    summaries: list[RunSummary],
    fields: dict[str, dict[str, object]],
) -> list[list[object]]:
    rows: list[list[object]] = []
    for summary in summaries:
        rows.append(
            [
                _format_table_timestamp(getattr(summary, name))
                if metadata.get("timestamp")
                else getattr(summary, name) or "—"
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
    parsed_since = parse_time(since, time.time()) if since else None
    if since and parsed_since is None:
        raise click.BadParameter(
            "Must be epoch seconds, ISO 8601, or a relative time such as 24h"
        )
    params: dict[str, str | int | None] = {
        **filters,
        "since": int(parsed_since) if parsed_since is not None else None,
        "limit": limit,
    }
    with DashboardClient() as client:
        payload = client.get_json("/api/runs", params=params)

    if not isinstance(payload, list):
        raise CLIError(
            "Dashboard returned an invalid run list.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )
    try:
        summaries = [RunSummary.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise CLIError(
            "Dashboard returned an invalid run summary.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        ) from exc
    fields = run_list_field_metadata()
    if json_output:
        print_json(
            [
                {
                    CLI_FIELD_NAMES.get(name, name): (
                        _format_timestamp(getattr(summary, name))
                        if metadata.get("timestamp")
                        else getattr(summary, name)
                    )
                    for name, metadata in fields.items()
                }
                for summary in summaries
            ]
        )
    else:
        print_table(
            [str(metadata["label"]) for metadata in fields.values()],
            _table_rows(summaries, fields),
        )


@click.group("run", cls=_TrainingGymGroup)
def run_group() -> None:
    """Inspect and manage training runs."""


@run_group.command(
    "list",
    help=(
        "List training runs with their top-level metadata.\n\n"
        "Supports filtering on status, model, dataset, recipe, group, "
        "or by recency, all with a limit. Sorted by most recently updated."
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
