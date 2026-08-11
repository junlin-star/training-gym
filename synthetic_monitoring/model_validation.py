"""Weekly synthetic monitoring for training-gym model validation.

Test it with:

```shell
    uv run modal run -m synthetic_monitoring.model_validation --model qwen3-4b --dryrun
```
"""

from __future__ import annotations

import os
import secrets
import time
import traceback
from pathlib import Path

import modal
import modal.exception
from pydantic import ValidationError

from modal_training_gym.common.modal_lifecycle import stop_app
from modal_training_gym.common.models.validation import VALIDATABLE_MODELS
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.model_validation import (
    TutorialResult,
    build_validation_plan,
    phase_timing_rows,
    run_validation_plan,
    status_label,
    validation_config_digest,
)
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items_healed,
)
from synthetic_monitoring.chart import (
    RunPoint,
    append_history,
    load_history,
    render_timing_history_chart,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_TIMEOUT_S = 60 * 60
CLEANUP_GRACE_S = 5 * 60
LAUNCH_TIMEOUT_S = (
    len(VALIDATABLE_MODELS) * (PROBE_TIMEOUT_S + CLEANUP_GRACE_S) + 30 * 60
)
MODAL_ENV = "training-gym"
SYNMON_GROUP_ID = "synmon-model-validation"
SYNMON_SCHEMA = 1
SLACK_CHANNEL_ID = "C0B9ZEA3ASD"

_TERMINAL_STATUSES = frozenset(
    {
        TrainingRunStatus.COMPLETED,
        TrainingRunStatus.FAILED,
        TrainingRunStatus.STOPPED,
        TrainingRunStatus.CANCELLED,
    }
)

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(REPO_ROOT), extra_options="--no-dev")
    .uv_pip_install("slack-sdk==3.27.1", "matplotlib==3.10.1")
    .env({"MODAL_ENVIRONMENT": MODAL_ENV})
    .add_local_python_source("modal_training_gym", "synthetic_monitoring")
)

slack_secret = modal.Secret.from_name("gym-bot-slack", environment_name=MODAL_ENV)
modal_creds_secret = modal.Secret.from_name(
    "_training-gym-modal-creds", environment_name=MODAL_ENV
)

app = modal.App("gym-synmon-launcher")


def pick_model(requested: str) -> str:
    catalog = [name for name, _cls in VALIDATABLE_MODELS]
    if not requested:
        raise ValueError(f"model name is required; available: {', '.join(catalog)}")
    lookup = {name.lower(): name for name in catalog}
    resolved = lookup.get(requested.lower())
    if resolved is None:
        raise ValueError(
            f"unknown model {requested!r}; available: {', '.join(catalog)}"
        )
    return resolved


def cleanup_synmon_probe_runs(
    *,
    probe_id: str,
    signature: str,
    reason: str,
) -> list[str]:
    items = vol_get_summary_items_healed(MetadataStore.TRAINING_RUNS_SUMMARY) or []
    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            summary_run = TrainingRun.model_validate(item)
        except ValidationError:
            continue
        if summary_run.group_id != SYNMON_GROUP_ID:
            continue
        try:
            run = TrainingRun.from_id(summary_run.training_run_id)
        except (
            KeyError,
            OSError,
            RuntimeError,
            ValidationError,
            modal.exception.Error,
        ) as exc:
            raise RuntimeError(
                f"load {summary_run.training_run_id} for cleanup: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        meta = run.metadata or {}
        tags = meta.get("group_tags")
        overrides = (
            tags.get("overrides")
            if isinstance(tags, dict) and isinstance(tags.get("overrides"), dict)
            else {}
        )
        if (
            run.group_id != SYNMON_GROUP_ID
            or overrides.get("synmon.schema") != SYNMON_SCHEMA
            or overrides.get("synmon.signature") != signature
            or overrides.get("synmon.probe_id") != probe_id
        ):
            continue
        if run.status in _TERMINAL_STATUSES:
            continue

        if run.function_call_id:
            try:
                modal.FunctionCall.from_id(run.function_call_id).cancel(
                    terminate_containers=True
                )
            except Exception as exc:
                print(
                    f"WARNING: cancel {run.function_call_id} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        if run.modal_app_id:
            stop_app(run.modal_app_id)
            print(f"watchdog stop requested for {run.training_run_id}: {reason}")
        cleaned.append(run.training_run_id)
    return cleaned


def notify(
    result: TutorialResult, *, error: str = "", status_override: str = ""
) -> None:
    from slack_sdk import WebClient

    history = load_history(result.base_model_name, environment_name=MODAL_ENV)
    if result.succeeded:
        timings: dict[str, float] = {}
        for label, duration in phase_timing_rows(result):
            if duration is None:
                continue
            timings[label] = float(duration)
        if timings:
            history = append_history(
                result.base_model_name,
                RunPoint(
                    ts=time.time(),
                    timings=timings,
                    training_run_id=str(result.training_run_id or ""),
                    succeeded=bool(result.succeeded),
                    total_duration_s=float(result.total_duration_s or 0.0),
                ),
                environment_name=MODAL_ENV,
            )

    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN missing")
    client = WebClient(token=token)
    failed = bool(error) or not result.succeeded
    status = status_override or status_label(result)
    fail_text = f"`{result.base_model_name}` {status}"
    if error:
        fail_text = f"{fail_text}\n```{error[:1500]}```"

    if not history:
        if not failed:
            return
        response = client.chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=fail_text,
            mrkdwn=True,
            unfurl_links=False,
            unfurl_media=False,
        )
        if not response.get("ok"):
            raise RuntimeError(
                f"Slack chat_postMessage failed: {response.get('error')}"
            )
        return

    try:
        png = render_timing_history_chart(history, model_name=result.base_model_name)
    except Exception as exc:
        print(
            f"WARNING: timing chart failed for {result.base_model_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        if failed:
            response = client.chat_postMessage(
                channel=SLACK_CHANNEL_ID,
                text=fail_text,
                mrkdwn=True,
                unfurl_links=False,
                unfurl_media=False,
            )
            if not response.get("ok"):
                raise RuntimeError(
                    f"Slack chat_postMessage failed: {response.get('error')}"
                )
        return

    filename = f"timing_history_{result.base_model_name.replace('/', '_')}.png"
    title = f"{result.base_model_name} (n={len(history)} runs)"
    upload_kwargs = {
        "channel": SLACK_CHANNEL_ID,
        "content": png,
        "filename": filename,
        "title": title,
    }
    if failed:
        upload_kwargs["initial_comment"] = fail_text
    try:
        upload = client.files_upload_v2(**upload_kwargs)
    except Exception as exc:
        print(
            f"WARNING: Slack chart upload failed for {filename}: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    if not upload.get("ok"):
        print(
            f"WARNING: Slack chart upload failed for {filename}: {upload.get('error')}"
        )


@app.function(
    image=probe_image,
    timeout=PROBE_TIMEOUT_S + CLEANUP_GRACE_S + 10 * 60,
    secrets=[modal_creds_secret],
)
def probe_watchdog(probe_id: str, signature: str, deadline: float) -> list[str]:
    remaining = deadline - time.time()
    if remaining > 0:
        time.sleep(remaining)
    return cleanup_synmon_probe_runs(
        probe_id=probe_id,
        signature=signature,
        reason="synmon watchdog deadline",
    )


@app.function(
    image=probe_image,
    timeout=PROBE_TIMEOUT_S + CLEANUP_GRACE_S,
    secrets=[slack_secret, modal_creds_secret],
)
def monitor(
    model: str = "",
    dryrun: bool = False,
    num_steps: int = 1,
) -> dict:
    selected = pick_model(model)
    print(f"synmon selected model={selected!r} dryrun={dryrun} num_steps={num_steps}")
    if dryrun:
        print(f"dryrun: would probe {selected!r} num_steps={num_steps}")
        return {
            "model": selected,
            "dryrun": True,
            "succeeded": True,
            "slack": False,
        }

    plan = build_validation_plan(selected, step_count=num_steps)
    signature = f"{SYNMON_SCHEMA}:{validation_config_digest(plan)}"
    probe_id = secrets.token_hex(8)
    group_overrides = {
        "synmon.schema": SYNMON_SCHEMA,
        "synmon.signature": signature,
        "synmon.model": selected,
        "synmon.probe_id": probe_id,
    }
    t0 = time.time()
    result_deadline = t0 + PROBE_TIMEOUT_S
    watchdog = probe_watchdog.spawn(
        probe_id=probe_id,
        signature=signature,
        deadline=result_deadline + CLEANUP_GRACE_S,
    )
    try:
        try:
            result = run_validation_plan(
                plan,
                wandb_project=None,
                group_id=SYNMON_GROUP_ID,
                group_overrides=group_overrides,
                group_axes=list(group_overrides),
                result_deadline=result_deadline,
            )
        except Exception as exc:
            try:
                cleanup_synmon_probe_runs(
                    probe_id=probe_id,
                    signature=signature,
                    reason=f"probe error: {type(exc).__name__}: {exc}",
                )
            except Exception as cleanup_exc:
                print(
                    f"WARNING: immediate probe cleanup failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            failed = TutorialResult(
                base_model_name=selected,
                step_count=num_steps,
                training_run_id="",
                training_run_status=TrainingRunStatus.FAILED,
                total_duration_s=float(time.time() - t0),
            )
            try:
                notify(
                    failed,
                    error=f"{type(exc).__name__}: {exc}",
                    status_override="❌ error",
                )
            except Exception as report_exc:
                exc.add_note(
                    "Slack reporting also failed: "
                    f"{type(report_exc).__name__}: {report_exc}"
                )
                raise exc from report_exc
            raise

        try:
            notify(result)
        except Exception as report_exc:
            raise RuntimeError(
                f"synmon reporting failed for {result.base_model_name}"
            ) from report_exc

        if not result.succeeded:
            raise RuntimeError(
                f"model validation failed for {result.base_model_name}: "
                f"{result.training_run_status.value} run={result.training_run_id}"
            )
        return {
            "model": result.base_model_name,
            "succeeded": result.succeeded,
            "training_run_id": result.training_run_id,
            "modal_app_url": result.modal_app_url,
            "total_duration_s": result.total_duration_s,
            "probe_id": probe_id,
            "signature": signature,
        }
    finally:
        try:
            watchdog.cancel(terminate_containers=True)
        except Exception as cancel_exc:
            print(
                f"WARNING: watchdog cancel failed: "
                f"{type(cancel_exc).__name__}: {cancel_exc}"
            )


@app.function(
    image=probe_image,
    timeout=LAUNCH_TIMEOUT_S,
    schedule=modal.Cron("17 0 * * 0"),
)
def launch_weekly(
    model: str = "",
    dryrun: bool = False,
    num_steps: int = 1,
) -> list[dict]:
    names = (
        [pick_model(model)] if model else [name for name, _cls in VALIDATABLE_MODELS]
    )
    if not names:
        raise RuntimeError("no validatable models registered")

    results: list[dict] = []
    for name, row in zip(
        names,
        monitor.map(
            names,
            kwargs={"dryrun": dryrun, "num_steps": num_steps},
            return_exceptions=True,
        ),
    ):
        if isinstance(row, Exception):
            err = f"{type(row).__name__}: {row}"
            tb = "".join(traceback.format_exception(type(row), row, row.__traceback__))
            print(f"ERROR: probe failed for {name}: {err}\n{tb}")
            results.append({"model": name, "succeeded": False, "error": err})
            continue
        results.append(row)

    failures = [
        f"{row['model']}: {row.get('error', 'failed')}"
        for row in results
        if not row.get("succeeded", False)
    ]
    if failures:
        raise RuntimeError(
            "synmon fan-out finished with failures:\n" + "\n".join(failures)
        )
    return results


@app.local_entrypoint()
def main(
    model: str = "",
    dryrun: bool = True,
    num_steps: int = 1,
) -> None:
    result = monitor.remote(model=model, dryrun=dryrun, num_steps=num_steps)
    print(result)
