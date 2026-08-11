"""
Input: string [ model name ]
Output: string [ Formatted test result ]
Optional args:
    -j: json formatted output
    -o: output file path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modal_training_gym.model_validation import (
    TutorialResult,
    available_model_names,
    build_validation_plan,
    fmt_secs,
    phase_timing_rows,
    run_validation_plan,
    status_label,
    total_step_time_s,
)


def _format_secs_delta(
    current: float | int | None,
    baseline: float | int | None,
) -> str | None:
    if current is None or baseline is None:
        return None
    current_f = float(current)
    baseline_f = float(baseline)
    delta_s = current_f - baseline_f
    if baseline_f <= 0:
        return f"{delta_s:+.3f}s"
    percent = delta_s / baseline_f * 100
    return f"{delta_s:+.3f}s ({percent:+.0f}%)"


def _training_run_link(training_run_id: str, dashboard_url: str | None) -> str:
    if not dashboard_url:
        return f"`{training_run_id}`"
    base = dashboard_url.rstrip("/")
    return f"[`{training_run_id}`]({base}/training/{training_run_id})"


def _format_result_details(
    result: TutorialResult,
    baseline: TutorialResult | None = None,
    baseline_commit_sha: str | None = None,
    baseline_commit_url: str | None = None,
    dashboard_url: str | None = None,
) -> list[str]:
    lines = [
        "<details>",
        f"<summary>{result.base_model_name}</summary>",
        "",
        f"{_training_run_link(result.training_run_id, dashboard_url)} — "
        f"{status_label(result)}",
    ]
    if baseline is not None:
        baseline_bits = [
            _training_run_link(baseline.training_run_id, dashboard_url),
        ]
        if baseline_commit_sha and baseline_commit_url:
            baseline_bits.append(
                f"on [`{baseline_commit_sha[:7]}`]({baseline_commit_url})"
            )
        lines.append(f"Baseline: {' '.join(baseline_bits)}")
    lines.append("")

    current_rows = [
        row for row in phase_timing_rows(result) if row[0] != "Total duration"
    ]
    if not current_rows:
        lines.extend(["_No step timing data._", "", "</details>", ""])
        return lines

    if baseline is None:
        lines.extend(["| Phase | Duration |", "| --- | --- |"])
    else:
        lines.extend(["| Phase | Duration | Delta |", "| --- | --- | --- |"])

    baseline_timings: dict[str, float] = {}
    if baseline is not None:
        for label, duration in phase_timing_rows(baseline):
            if duration is None:
                continue
            baseline_timings[label] = float(duration)
    for label, duration in current_rows:
        if baseline is None:
            lines.append(f"| {label} | {fmt_secs(duration)} |")
            continue
        delta = _format_secs_delta(duration, baseline_timings.get(label)) or "—"
        lines.append(f"| {label} | {fmt_secs(duration)} | {delta} |")

    step_labels = [label for label, _ in current_rows if label.startswith("Step ")]
    if len(step_labels) > 1:
        total = total_step_time_s(result)
        if baseline is None:
            lines.append(f"| Total step time | {fmt_secs(total)} |")
        else:
            delta = _format_secs_delta(total, total_step_time_s(baseline)) or "—"
            lines.append(f"| Total step time | {fmt_secs(total)} | {delta} |")
    lines.extend(["", "</details>", ""])
    return lines


def summarize_results(
    results_dir: str,
    baseline_dir: str | None = None,
    dashboard_url: str | None = None,
) -> str:
    rows = []
    details: list[str] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        result = TutorialResult.from_dict(json.loads(path.read_text()))
        status = status_label(result)
        row = (
            f"| {result.base_model_name} | {status} "
            f"| {total_step_time_s(result):.1f}s | {result.step_count} "
            f"| {_training_run_link(result.training_run_id, dashboard_url)} |"
        )
        baseline: TutorialResult | None = None
        baseline_commit_sha: str | None = None
        baseline_commit_url: str | None = None
        if baseline_dir is not None:
            baseline_path = Path(baseline_dir) / path.name
            if baseline_path.is_file():
                baseline = TutorialResult.from_dict(
                    json.loads(baseline_path.read_text())
                )
                delta = (
                    _format_secs_delta(
                        total_step_time_s(result),
                        total_step_time_s(baseline),
                    )
                    or "—"
                )
                baseline_run = _training_run_link(
                    baseline.training_run_id, dashboard_url
                )
                row += f" {delta} from {baseline_run} |"
                meta_path = baseline_path.with_name(baseline_path.stem + ".meta.json")
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text())
                    sha = meta.get("commit_sha")
                    url = meta.get("commit_url")
                    if sha and url:
                        baseline_commit_sha = str(sha)
                        baseline_commit_url = str(url)
            else:
                row += " — |"
        rows.append(row)
        details.extend(
            _format_result_details(
                result,
                baseline,
                baseline_commit_sha,
                baseline_commit_url,
                dashboard_url,
            )
        )

    header = "| Model | Status | Step time | Steps | Run |"
    divider = "| --- | --- | --- | --- | --- |"
    empty = "| _no results_ | | | | |"
    if baseline_dir is not None:
        header += " Delta |"
        divider += " --- |"
        empty += " |"
    lines = [
        "<!-- validate-models-comment -->",
        "## Model Validation Results",
        "",
        header,
        divider,
    ]
    lines.extend(rows or [empty])
    if details:
        lines.extend(["", "### Step timings", ""])
        lines.extend(details)
    return "\n".join(lines).rstrip() + "\n"


def __main__():
    parser = argparse.ArgumentParser(
        description="Validate a model config by running base training on slime."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run base training for a single model."
    )
    check_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Base model name to run training on (e.g. qwen3-4b).",
    )
    check_parser.add_argument(
        "-n",
        "--num_steps",
        type=int,
        default=1,
        help="Number of training steps (rollouts) to run. Defaults to 1.",
    )
    check_parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Override the recipe eval_interval (eval every N rollouts).",
    )
    check_parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override the recipe save_interval (checkpoint every N rollouts).",
    )
    check_parser.add_argument(
        "--non-colocated",
        action="store_true",
        help="Allocate rollout GPUs separately from trainer GPUs.",
    )
    check_parser.add_argument(
        "-o",
        "--output",
        help="Write the result as JSON to this file path.",
    )
    check_parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print the result as JSON to stdout.",
    )
    check_parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project for validator runs. If omitted, W&B logging is disabled.",
    )
    check_parser.add_argument(
        "--wandb-group",
        default="",
        help="W&B group for validator runs. Defaults to model-validator-{model}-{dataset}.",
    )
    check_parser.add_argument(
        "--wandb-secret-name",
        default="wandb-secret",
        help="Modal Secret name containing WANDB_API_KEY.",
    )
    check_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging for this validator run.",
    )

    subparsers.add_parser(
        "list", help="Print available model names as a JSON array and exit."
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Render a markdown table from a directory of result JSON files.",
    )
    summarize_parser.add_argument(
        "-d",
        "--results-dir",
        required=True,
        help="Directory containing result JSON files written by `check --output`.",
    )
    summarize_parser.add_argument(
        "-b",
        "--baseline-dir",
        help="Directory containing baseline result JSON files to compare against.",
    )
    summarize_parser.add_argument(
        "--dashboard-url",
        help="Base URL of the training dashboard. If omitted, run ids are not linked.",
    )

    args = parser.parse_args()

    if args.command == "list":
        print(json.dumps(available_model_names()))
        return

    if args.command == "summarize":
        print(
            summarize_results(
                args.results_dir,
                args.baseline_dir,
                args.dashboard_url,
            )
        )
        return

    plan = build_validation_plan(
        args.model,
        step_count=args.num_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        colocate=False if args.non_colocated else None,
    )
    tutorial_result = run_validation_plan(
        plan,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_secret_name=args.wandb_secret_name,
    )
    tutorial_result.print_summary()

    if args.output:
        Path(args.output).write_text(json.dumps(tutorial_result.to_dict()))
    if args.json:
        print(json.dumps(tutorial_result.to_dict()))

    if not tutorial_result.succeeded:
        print("Training run failed")
        exit(1)
    print("Training run completed successfully")
    exit(0)


if __name__ == "__main__":
    __main__()
