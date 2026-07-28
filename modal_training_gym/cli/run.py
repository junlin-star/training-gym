"""Commands for inspecting training runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import click
from pydantic import ValidationError

from modal_training_gym.common.run_list import run_list_field_metadata
from modal_training_gym.common.run_summary import RunSummary
from modal_training_gym.common.time import parse_time
from modal_training_gym.common.training_rollout import TrainingRolloutSummary

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
    age = (time.time() if now is None else now) - value
    if 0 <= age < 60:
        return "now"
    if 0 <= age < 3_600:
        return f"{int(age // 60)}m ago"
    if 0 <= age < 86_400:
        return f"{int(age // 3_600)}h ago"
    if 0 <= age < 2_592_000:
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


def _format_reward(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _current_step(summary: RunSummary) -> tuple[int | None, int | None, str]:
    progress = summary.framework_progress
    if progress is None:
        return None, None, "step"
    return progress.current, progress.total, progress.unit


def _run_payload(summary: RunSummary) -> dict[str, object]:
    current_step, total_steps, step_unit = _current_step(summary)
    return {
        "run_id": summary.run_id,
        "status": summary.display_status,
        "stage": summary.display_stage or None,
        "current_step": current_step,
        "total_steps": total_steps,
        "step_unit": step_unit,
        "current_reward": (
            summary.latest_rollout.mean if summary.latest_rollout is not None else None
        ),
        "model": summary.model or None,
        "dataset": summary.dataset or None,
        "recipe": summary.recipe or None,
        "group": summary.group_id or None,
        "created_at": _format_timestamp(summary.created_at),
        "last_updated_at": _format_timestamp(summary.updated_at),
    }


def _validate_run_summary(payload: object) -> RunSummary:
    try:
        return RunSummary.model_validate(payload)
    except ValidationError as exc:
        raise CLIError(
            "Dashboard returned an invalid run summary.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        ) from exc


def _validate_rollouts(payload: object) -> list[TrainingRolloutSummary]:
    if not isinstance(payload, list):
        raise CLIError(
            "Dashboard returned invalid rollout data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )
    try:
        return [TrainingRolloutSummary.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise CLIError(
            "Dashboard returned invalid rollout data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        ) from exc


def _format_step(summary: RunSummary) -> str:
    current, total, unit = _current_step(summary)
    if current is None:
        return "—"
    value = f"{current} / {total}" if total is not None else str(current)
    return f"{value} {unit}".strip()


def get_run(*, run_id: str, verbose: bool, json_output: bool) -> None:
    """Fetch and render one run, optionally including rollout history."""
    encoded_run_id = quote(run_id, safe="")
    not_found_error = CLIError(
        f"Training run {run_id!r} was not found.",
        error="run_not_found",
        exit_code=ExitCode.NOT_FOUND,
        run_id=run_id,
        hint="training-gym run list",
    )
    with DashboardClient() as client:
        summary = _validate_run_summary(
            client.get_json(
                f"/api/runs/{encoded_run_id}",
                params=None,
                not_found_error=not_found_error,
            )
        )
        rollouts = (
            _validate_rollouts(
                client.get_json(
                    f"/api/runs/{encoded_run_id}/rollouts",
                    params=None,
                )
            )
            if verbose
            else []
        )

    if json_output:
        payload = _run_payload(summary)
        if verbose:
            payload["reward_over_time"] = [
                {
                    "rollout_id": rollout.rollout_id,
                    "reward": rollout.mean,
                    "created_at": _format_timestamp(rollout.created_at),
                }
                for rollout in rollouts
            ]
            payload["rollouts"] = [
                rollout.model_dump(mode="json", exclude_none=True)
                for rollout in rollouts
            ]
        print_json(payload)
        return

    print_table(
        ["Field", "Value"],
        [
            ["Run", summary.run_id],
            ["Status", summary.display_status or "—"],
            ["Stage", summary.display_stage or "—"],
            ["Current step", _format_step(summary)],
            [
                "Current reward",
                _format_reward(
                    summary.latest_rollout.mean
                    if summary.latest_rollout is not None
                    else None
                ),
            ],
            ["Model", summary.model or "—"],
            ["Dataset", summary.dataset or "—"],
            ["Recipe", summary.recipe or "—"],
            ["Group", summary.group_id or "—"],
            ["Created", _format_timestamp(summary.created_at)],
            ["Last updated", _format_timestamp(summary.updated_at)],
        ],
        title=f"Run {summary.run_id}",
        show_header=False,
    )
    if not verbose:
        return

    print_table(
        ["Rollout", "Reward", "Recorded"],
        [
            [
                rollout.rollout_id,
                _format_reward(rollout.mean),
                _format_timestamp(rollout.created_at),
            ]
            for rollout in rollouts
        ],
        title="Reward over time",
    )
    print_table(
        ["Rollout", "Samples", "Duration", "Errors"],
        [
            [
                rollout.rollout_id,
                rollout.total,
                (
                    f"{rollout.rollout_time:.2f}s"
                    if rollout.rollout_time is not None
                    else "—"
                ),
                (
                    rollout.error_summary.get("verdict", "—")
                    if rollout.error_summary is not None
                    else "—"
                ),
            ]
            for rollout in rollouts
        ],
        title="Rollouts",
    )


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
    summaries = [_validate_run_summary(item) for item in payload]
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
    "get",
    help="Show status and top-level metadata for a single run.",
    epilog=(
        "Examples:\n"
        "  training-gym run get run_8f2a\n"
        "  training-gym run get run_8f2a --verbose"
    ),
)
@click.argument("run_id")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Include reward-over-time and rollout data.",
)
@json_option
def get_command(*, run_id: str, verbose: bool, json_output: bool) -> None:
    """Show status and top-level metadata for a single run."""
    get_run(run_id=run_id, verbose=verbose, json_output=json_output)


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
