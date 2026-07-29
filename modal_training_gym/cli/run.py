"""Commands for inspecting training runs."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import click
from pydantic import ValidationError
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from modal_training_gym.common.run_list import run_list_field_metadata
from modal_training_gym.common.run_summary import RunSummary
from modal_training_gym.common.time import parse_time
from modal_training_gym.common.training_rollout import TrainingRolloutSummary

from .client import DashboardClient
from .commands import _TrainingGymGroup
from .errors import CLIError, ExitCode
from .options import json_option
from .output import print_json, print_renderable, print_table


DEFAULT_RUN_LIMIT = 50
DEFAULT_LOG_TAIL = 100
MAX_LOG_TAIL = 20_000
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


def _chip(value: str, *, style: str) -> Text:
    return Text(f" {value} ", style=style)


def _run_summary_panel(summary: RunSummary) -> Panel:
    status = summary.display_status or "pending"
    status_style = {
        "completed": "bold bright_green",
        "failed": "bold red",
        "cancelled": "bold yellow",
        "stopped": "bold yellow",
        "pending": "bold cyan",
    }.get(status, "bold")
    heading = Text()
    heading.append("● ", style=status_style)
    heading.append(status.upper(), style=status_style)
    if summary.display_stage:
        heading.append("  ")
        heading.append(summary.display_stage, style="bold")

    reward = summary.latest_rollout.mean if summary.latest_rollout is not None else None
    metrics = Table.grid(padding=(0, 4))
    metrics.add_row(
        Text.assemble(("Step  ", "dim"), (_format_step(summary), "bold")),
        Text.assemble(("Reward  ", "dim"), (_format_reward(reward), "bold")),
    )

    chips = [
        _chip(summary.model, style="black on bright_green") if summary.model else None,
        _chip(summary.dataset, style="black on cyan") if summary.dataset else None,
        _chip(summary.recipe, style="black on white") if summary.recipe else None,
        _chip(summary.group_id, style="white on grey23") if summary.group_id else None,
    ]
    footer = Text.assemble(
        ("Updated ", "dim"),
        (_format_table_timestamp(summary.updated_at), "dim bold"),
        ("  ·  Created ", "dim"),
        (_format_table_timestamp(summary.created_at), "dim bold"),
    )
    body = Group(
        heading,
        Text(""),
        metrics,
        Text(""),
        Columns([chip for chip in chips if chip is not None], padding=(0, 1)),
        Text(""),
        footer,
    )
    return Panel(
        body,
        title=summary.run_id,
        title_align="left",
        border_style="bright_green",
        padding=(1, 2),
    )


def _reward_sparkline(rollouts: list[TrainingRolloutSummary]) -> str:
    if not rollouts:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    rewards = [rollout.mean for rollout in rollouts]
    low, high = min(rewards), max(rewards)
    if low == high:
        return blocks[len(blocks) // 2] * len(rewards)
    return "".join(
        blocks[round((reward - low) / (high - low) * (len(blocks) - 1))]
        for reward in rewards
    )


def _reward_panel(rollouts: list[TrainingRolloutSummary]) -> Panel:
    if not rollouts:
        content: Text | Group = Text("No rollout rewards recorded.", style="dim")
    else:
        first, latest = rollouts[0].mean, rollouts[-1].mean
        table = Table(
            "Rollout",
            "Reward",
            "Samples",
            "Duration",
            "Errors",
            box=None,
            header_style="bold bright_green",
            pad_edge=False,
            expand=True,
        )
        for rollout in rollouts:
            table.add_row(
                str(rollout.rollout_id),
                _format_reward(rollout.mean),
                str(rollout.total),
                (
                    f"{rollout.rollout_time:.2f}s"
                    if rollout.rollout_time is not None
                    else "—"
                ),
                (
                    str(rollout.error_summary.get("verdict", "—"))
                    if rollout.error_summary is not None
                    else "—"
                ),
            )
        content = Group(
            Text(_reward_sparkline(rollouts), style="bold bright_green"),
            Text.assemble(
                (_format_reward(first), "dim"),
                ("  →  ", "dim"),
                (_format_reward(latest), "bold"),
                (f"   {len(rollouts)} rollouts", "dim"),
            ),
            Text(""),
            table,
        )
    return Panel(
        content,
        title="Reward over time",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
    )


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

    print_renderable(_run_summary_panel(summary))
    if not verbose:
        return

    print_renderable(_reward_panel(rollouts))


def show_run_params(*, run_id: str, json_output: bool) -> None:
    """Fetch and render the framework recipe parameters for one run."""
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

    config = summary.config
    params = config.get("recipe") or config.get("preset") or {}
    if not isinstance(params, dict):
        raise CLIError(
            "Dashboard returned invalid framework parameters.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
            run_id=run_id,
        )

    if json_output:
        print_json(params)
        return

    configured_params = {
        name: value
        for name, value in params.items()
        if value is not None and value != ""
    }
    print_table(
        ["Parameter", "Value"],
        [
            [
                name,
                (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else str(value)
                ),
            ]
            for name, value in configured_params.items()
        ],
        title=f"Training recipe for {run_id}",
        show_header=False,
    )


def _validate_log_payload(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("logs"), list):
        raise CLIError(
            "Dashboard returned invalid log data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )

    logs: list[dict[str, object]] = []
    for entry in payload["logs"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("line"), str):
            raise CLIError(
                "Dashboard returned invalid log data.",
                error="invalid_dashboard_response",
                exit_code=ExitCode.BACKEND,
            )
        logs.append(entry)
    return logs


def _print_log_line(line: str) -> None:
    click.echo(line, nl=not line.endswith("\n"))


def _decode_stream_event(event: str, data: str) -> dict[str, object]:
    try:
        payload = json.loads(data)
    except (TypeError, ValueError) as exc:
        raise CLIError(
            "Dashboard returned malformed log stream data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        ) from exc
    if not isinstance(payload, dict):
        raise CLIError(
            "Dashboard returned invalid log stream data.",
            error="invalid_dashboard_response",
            exit_code=ExitCode.BACKEND,
        )
    if event != "message":
        payload = {"event": event, **payload}
    return payload


def show_run_logs(
    *,
    run_id: str,
    follow: bool,
    since: str | None,
    until: str | None,
    tail: int | None,
    search: str | None,
    json_output: bool,
) -> None:
    """Fetch historical logs or follow the live dashboard log stream."""
    if follow and (since or until or tail is not None):
        raise click.UsageError(
            "--since, --until, and --tail apply only when fetching logs "
            "without --follow."
        )

    encoded_run_id = quote(run_id, safe="")
    not_found_error = CLIError(
        f"Training run {run_id!r} was not found.",
        error="run_not_found",
        exit_code=ExitCode.NOT_FOUND,
        run_id=run_id,
        hint="training-gym run list",
    )

    with DashboardClient() as client:
        if not follow:
            payload = client.get_json(
                f"/api/runs/{encoded_run_id}/logs",
                params={
                    "since": since,
                    "until": until,
                    "max_lines": tail or DEFAULT_LOG_TAIL,
                    "search": search,
                },
                not_found_error=not_found_error,
            )
            logs = _validate_log_payload(payload)
            if json_output:
                print_json(logs)
                return
            for entry in logs:
                _print_log_line(str(entry["line"]))
            return

        for event, data in client.iter_sse(
            f"/api/runs/{encoded_run_id}/logs/stream",
            params={"search": search},
            not_found_error=not_found_error,
        ):
            decoded = _decode_stream_event(event, data)
            if event == "done":
                return
            if event == "error":
                raise CLIError(
                    str(decoded.get("error") or "Dashboard log stream failed."),
                    error="log_stream_failed",
                    exit_code=ExitCode.BACKEND,
                    run_id=run_id,
                )
            if json_output:
                click.echo(json.dumps(decoded, ensure_ascii=False))
            elif event == "message":
                line = decoded.get("line")
                if not isinstance(line, str):
                    raise CLIError(
                        "Dashboard returned invalid log stream data.",
                        error="invalid_dashboard_response",
                        exit_code=ExitCode.BACKEND,
                    )
                _print_log_line(line)
            elif event == "dropped":
                click.echo(
                    f"[dropped {decoded.get('dropped', 0)} log lines]",
                    err=True,
                )
            elif event == "reconnect":
                click.echo("[reconnecting log stream]", err=True)


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
    "params",
    help=("Show the framework training recipe for a single run."),
    epilog=(
        "Examples:\n"
        "  training-gym run params run_8f2a\n"
        "  training-gym run params run_8f2a --json"
    ),
)
@click.argument("run_id")
@json_option
def params_command(*, run_id: str, json_output: bool) -> None:
    """Show the framework training recipe for a single run."""
    show_run_params(run_id=run_id, json_output=json_output)


@run_group.command(
    "logs",
    help=(
        "Fetch or stream Modal app logs for a run.\n\n"
        "By default, fetches the most recent entries and exits; pass --follow "
        "to live-stream instead."
    ),
    epilog=(
        "Examples:\n"
        "  training-gym run logs brave-falcon-3fa8 --follow\n"
        "  training-gym run logs brave-falcon-3fa8 --since 30m -j"
    ),
)
@click.argument("run_id")
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    default=False,
    help="Stream new log output until interrupted or the app stops.",
)
@click.option(
    "--since",
    default=None,
    metavar="START",
    help="Only entries at/after START (ISO 8601 or relative: 30m, 2h, 1d).",
)
@click.option(
    "--until",
    default=None,
    metavar="END",
    help="Only entries at/before END (ISO 8601 or relative: 30m, 2h, 1d).",
)
@click.option(
    "-n",
    "--tail",
    type=click.IntRange(min=1, max=MAX_LOG_TAIL),
    default=None,
    metavar="N",
    help=f"Show only the last N entries (default: {DEFAULT_LOG_TAIL}).",
)
@click.option(
    "--search",
    default=None,
    metavar="TEXT",
    help="Filter by search text.",
)
@json_option
def logs_command(
    *,
    run_id: str,
    follow: bool,
    since: str | None,
    until: str | None,
    tail: int | None,
    search: str | None,
    json_output: bool,
) -> None:
    """Fetch or stream Modal app logs for a run."""
    show_run_logs(
        run_id=run_id,
        follow=follow,
        since=since,
        until=until,
        tail=tail,
        search=search,
        json_output=json_output,
    )


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
