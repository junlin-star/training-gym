"""Self-contained training-gym dashboard app.

When deployed from a pip install (no local repo checkout), the image build
clones the frontend source from GitHub. When running from a repo checkout,
it uses the local ``dashboards/frontend`` directory instead.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import secrets as _secrets
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import modal

# Imported at module scope so FastAPI can resolve the ``request: Request``
# annotation in stream_run_logs(). Under ``from __future__ import
# annotations`` all type hints are strings, and FastAPI evaluates them
# against the *defining function's* ``__globals__`` (i.e. this module).
# Importing ``Request`` only inside ``fastapi_app()`` makes the name
# invisible to FastAPI's introspection, which then mistakes the parameter
# for a query string and 422s with ``{"loc": ["query", "request"]}``.
from starlette.requests import Request

REPO_URL = "https://github.com/modal-projects/training-gym.git"
REPO_BRANCH = "main"

_repo_frontend = Path(__file__).resolve().parents[1] / "dashboards" / "frontend"
_has_local_frontend = _repo_frontend.is_dir()


def _build_image() -> modal.Image:
    base = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl")
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "apt-get install -y nodejs",
        )
        .pip_install("fastapi[standard]==0.118.0", "modal")
    )

    if _has_local_frontend:
        base = base.add_local_dir(
            str(_repo_frontend),
            remote_path="/app/frontend",
            copy=True,
            ignore=["node_modules", "dist"],
        )
    else:
        base = base.apt_install("git").run_commands(
            f"git clone --depth 1 -b {REPO_BRANCH} {REPO_URL} /tmp/training-gym",
            "mkdir -p /app && cp -r /tmp/training-gym/dashboards/frontend /app/frontend",
            "rm -rf /tmp/training-gym",
        )

    return base.add_local_python_source("modal_training_gym", copy=True).run_commands(
        "cd /app/frontend && npm install && npm run build",
    )


image = _build_image()

app = modal.App("training-gym-dashboard", image=image)

STATIC_DIR = "/app/frontend/dist"


# Underscore-prefixed so it shows up as an auto-managed secret in the Modal
# Secrets UI and is auto-created on first deploy from ~/.modal.toml.
MODAL_CREDS_SECRET_NAME = "_training-gym-modal-creds"


def _is_local() -> bool:
    """True when we're not running inside a Modal container."""
    return not os.environ.get("MODAL_IS_REMOTE")


def ensure_creds_secret(interactive: bool = False) -> bool:
    """Make sure the ``_training-gym-modal-creds`` Modal Secret exists.

    Idempotent: returns True if the secret was already present or if we
    successfully created it from ``~/.modal.toml``. Returns False if we
    can't find creds and ``interactive`` is False (or the user skipped).

    Called both from ``training-gym setup`` and at module-load of this file
    so that ``modal deploy dashboards/app.py`` works without any prior
    setup step — as long as the user has a valid ``~/.modal.toml``.
    """
    if not _is_local():
        # Inside a Modal container we have no credentials and no need to
        # create anything; the secret was already provisioned at deploy.
        return True

    from modal_training_gym.common.config import resolve_modal_creds

    token_id, token_secret, source = resolve_modal_creds()

    if not token_id or not token_secret:
        if not interactive:
            return False
        from getpass import getpass

        print(
            "\nThe dashboard needs Modal workspace credentials to stream "
            "training-run logs into the UI.\n"
            "Couldn't find creds in MODAL_TOKEN_* env vars or "
            "~/.modal.toml — provide them now (or Ctrl-C to skip).\n"
            "Find your tokens at https://modal.com/settings/tokens.\n"
        )
        try:
            token_id = input("MODAL_TOKEN_ID: ").strip()
            token_secret = getpass("MODAL_TOKEN_SECRET (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping Modal Secret setup.")
            return False
        source = "user input"

    if not token_id or not token_secret:
        return False

    try:
        env_values = {"MODAL_TOKEN_ID": token_id, "MODAL_TOKEN_SECRET": token_secret}
        modal.Secret.objects.create(
            MODAL_CREDS_SECRET_NAME, env_values, allow_existing=True
        )
        modal.Secret.from_name(MODAL_CREDS_SECRET_NAME).update(env_values)
        print(f"Provisioned Modal Secret {MODAL_CREDS_SECRET_NAME!r} (from {source}).")
        return True
    except Exception as exc:
        print(
            f"WARNING: failed to create Modal Secret {MODAL_CREDS_SECRET_NAME!r}: {exc}"
        )
        return False


# Auto-create the secret at module-load so `modal deploy dashboards/app.py`
# works out of the box. Side effect is gated to local context so it never
# fires inside the deployed container.
if _is_local():
    ensure_creds_secret(interactive=False)


def _run_compact_sync() -> None:
    """Rebuild all summary stores from canonical per-item metadata."""
    from modal_training_gym.utils.metadata import (
        vol_compact_summary_items,
        MetadataStore,
    )

    for summary_store, item_store, id_key, sk, rev in [
        (
            MetadataStore.TRAINING_RUNS_SUMMARY,
            MetadataStore.TRAINING_RUNS,
            "training_run_id",
            lambda item: (
                int(item.get("created_at", 0) or 0),
                str(item.get("training_run_id", "")),
            ),
            True,
        ),
        (
            MetadataStore.TRAIN_RESULTS_SUMMARY,
            MetadataStore.TRAIN_RESULTS,
            "training_run_id",
            lambda item: str(item.get("training_run_id", "")),
            True,
        ),
        (
            MetadataStore.DEPLOYMENTS_SUMMARY,
            MetadataStore.DEPLOYMENTS,
            "deployment_id",
            lambda item: (
                str(item.get("deployment_config", {}).get("app_name", "")),
                str(item.get("deployment_id", "")),
            ),
            True,
        ),
    ]:
        vol_compact_summary_items(
            summary_store,
            item_store,
            item_id_key=id_key,
            sort_key=sk,
            reverse=rev,
        )


@app.function(schedule=modal.Cron("*/30 * * * *"))
def compact_summaries() -> None:
    """Scheduled compaction of summary stores (every 30 min)."""
    _run_compact_sync()
    print("Compaction complete.")


@app.function(
    min_containers=1,
    secrets=[modal.Secret.from_name(MODAL_CREDS_SECRET_NAME)],
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import (
        FastAPI,
        Header,
        HTTPException,
    )  # Request imported at module scope
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    from modal_training_gym.common.modal_urls import modal_app_dashboard_url
    from modal_training_gym.common.run import TrainingRun
    from modal_training_gym.common.status import MilesStatus, SlimeStatus
    from modal_training_gym.common.training_rollout import TrainingRolloutResult
    from modal_training_gym.utils.metadata import (
        MetadataStore,
        vol_get,
        vol_get_summary_items,
        vol_put_summary_items,
    )

    web = FastAPI()
    cache_ttl_seconds = 5.0
    eval_summary_store = MetadataStore.EVALS
    eval_summary_key = "summary"
    eval_summary_payload_key = "summaries"
    cache_entries: dict[str, tuple[float, list[dict[str, Any]]]] = {
        "runs": (0.0, []),
        "train_results": (0.0, []),
        "evals": (0.0, []),
        "deployments": (0.0, []),
    }
    cache_locks = {
        "runs": asyncio.Lock(),
        "train_results": asyncio.Lock(),
        "evals": asyncio.Lock(),
        "deployments": asyncio.Lock(),
    }
    web.mount("/assets", StaticFiles(directory=f"{STATIC_DIR}/assets"), name="assets")

    async def get_cached_list(
        key: str, loader: Callable[[], Awaitable[list[dict[str, Any]]]]
    ) -> list[dict[str, Any]]:
        now = time.monotonic()
        expires_at, values = cache_entries[key]
        if now < expires_at:
            return values

        async with cache_locks[key]:
            now = time.monotonic()
            expires_at, values = cache_entries[key]
            if now < expires_at:
                return values
            values = await loader()
            cache_entries[key] = (now + cache_ttl_seconds, values)
            return values

    def list_from_payload(
        payload: Any,
        *,
        payload_key: str,
    ) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            items = payload.get(payload_key, [])
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []

    def add_modal_app_urls(
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        updated = []
        changed = False
        for item in items:
            new_item = dict(item)
            if not new_item.get("modal_app_url"):
                app_id = str(new_item.get("modal_app_id", "") or "")
                if app_id:
                    new_item["modal_app_url"] = modal_app_dashboard_url(app_id)
                    changed = True
            updated.append(new_item)
        return updated, changed

    async def load_eval_summaries() -> list[dict[str, Any]]:
        try:
            payload = await run_in_threadpool(
                vol_get, eval_summary_store, eval_summary_key
            )
        except KeyError:
            return []

        summaries = list_from_payload(payload, payload_key=eval_summary_payload_key)
        if not summaries:
            return []

        eval_ids = [
            str(s.get("eval_id", ""))
            for s in summaries
            if isinstance(s, dict) and s.get("eval_id")
        ]

        async def fetch_result(eval_id: str) -> tuple[str, dict | None]:
            try:
                r = await run_in_threadpool(
                    vol_get, MetadataStore.EVAL_RESULTS, eval_id
                )
                return eval_id, r
            except KeyError:
                return eval_id, None

        fetched = await asyncio.gather(*(fetch_result(eid) for eid in eval_ids))
        results_by_id = {eid: r for eid, r in fetched if r is not None}

        eval_config_ids = [
            str(s.get("eval_config_id", ""))
            for s in summaries
            if isinstance(s, dict) and s.get("eval_config_id")
        ]

        async def fetch_eval_config(
            eval_config_id: str,
        ) -> tuple[str, dict | None]:
            try:
                cfg = await run_in_threadpool(
                    vol_get, MetadataStore.EVAL_CONFIGS, eval_config_id
                )
                return eval_config_id, cfg
            except KeyError:
                return eval_config_id, None

        fetched_configs = await asyncio.gather(
            *(fetch_eval_config(cfg_id) for cfg_id in eval_config_ids)
        )
        configs_by_id = {
            cfg_id: cfg for cfg_id, cfg in fetched_configs if cfg is not None
        }

        enriched: list[dict[str, Any]] = []
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            eval_id = str(summary.get("eval_id", ""))
            eval_config_id = str(summary.get("eval_config_id", ""))
            result = results_by_id.get(eval_id)
            eval_config = configs_by_id.get(eval_config_id)
            if not result:
                merged = dict(summary)
                if eval_config:
                    merged["eval_config"] = eval_config
                    for field in (
                        "dataset_name",
                        "eval_fn_name",
                        "prompt_column",
                        "generate_kwargs",
                    ):
                        value = eval_config.get(field)
                        if field not in merged and value is not None:
                            merged[field] = value
                enriched.append(merged)
                continue
            merged = dict(summary)
            for field in ("deployment_id", "config", "status"):
                value = result.get(field)
                if field not in merged and value is not None:
                    merged[field] = value
            if eval_config:
                merged["eval_config"] = eval_config
                for field in (
                    "dataset_name",
                    "eval_fn_name",
                    "prompt_column",
                    "generate_kwargs",
                ):
                    value = eval_config.get(field)
                    if field not in merged and value is not None:
                        merged[field] = value
            enriched.append(merged)
        return enriched

    async def load_list_summary(
        summary_store: MetadataStore,
    ) -> list[dict[str, Any]]:
        items = await run_in_threadpool(vol_get_summary_items, summary_store)
        if items is None:
            return []
        items, changed = add_modal_app_urls(items)
        if changed:
            await run_in_threadpool(vol_put_summary_items, summary_store, items)
        return items

    async def load_runs() -> list[dict[str, Any]]:
        return await load_list_summary(MetadataStore.TRAINING_RUNS_SUMMARY)

    async def load_train_results() -> list[dict[str, Any]]:
        return await load_list_summary(MetadataStore.TRAIN_RESULTS_SUMMARY)

    async def load_deployments() -> list[dict[str, Any]]:
        return await load_list_summary(MetadataStore.DEPLOYMENTS_SUMMARY)

    def _resolve_framework_status(phase: str, framework: str) -> Any | None:
        phase = phase.strip()
        framework = framework.strip().lower()
        if framework == "miles":
            try:
                return MilesStatus(phase)
            except ValueError:
                return None
        try:
            return SlimeStatus(phase)
        except ValueError:
            try:
                return MilesStatus(phase)
            except ValueError:
                return None

    def _optional_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    # ── Response display ──────────────────────────────────────────────────
    # Responses are parsed at write time (slime recorder for rollouts, the eval
    # harness for evals) and stored as ``parsed_response``. Here we just surface
    # that cleaned content for display, keeping the raw under ``raw_response``.
    def _clean_prompt(text: str) -> str:
        """Make a chat-templated prompt readable for display.

        Dataset prompts often arrive as a chat template wrapping a Python repr
        of the messages list (e.g. ``<|im_start|>user\\n[{'content': '...',
        'role': 'user'}]<|im_end|>...``) plus a leaked reference/assistant turn.
        Pull the message content out and drop the template scaffolding; fall
        back to stripping special tokens when there's no messages repr.
        """
        start, end = text.find("[{"), text.rfind("}]")
        if start != -1 and end > start:
            try:
                data = ast.literal_eval(text[start : end + 2])
                if isinstance(data, list):
                    parts = [
                        str(m["content"])
                        for m in data
                        if isinstance(m, dict) and m.get("content")
                    ]
                    if parts:
                        return "\n\n".join(parts).strip()
            except (ValueError, SyntaxError):
                pass
        cleaned = re.sub(r"<\|[^|]*\|>", "", text)
        cleaned = re.sub(r"</?think>", "", cleaned)
        # Drop standalone role-header lines left behind by the template.
        cleaned = re.sub(r"(?m)^(system|user|assistant)\s*$\n?", "", cleaned)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    def _apply_parsed(rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            parsed = row.get("parsed_response")
            if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
                raw = row.get("response")
                if isinstance(raw, str):
                    row["raw_response"] = raw
                row["response"] = parsed.get("content") or ""
                if parsed.get("thinking"):
                    row["thinking"] = parsed["thinking"]
                if parsed.get("tool_calls"):
                    row["tool_calls"] = parsed["tool_calls"]
            # Clean the (chat-templated) prompt for display, keeping the raw.
            raw_prompt = row.get("prompt")
            if isinstance(raw_prompt, str) and raw_prompt:
                cleaned_prompt = _clean_prompt(raw_prompt)
                if cleaned_prompt and cleaned_prompt != raw_prompt:
                    row["raw_prompt"] = raw_prompt
                    row["prompt"] = cleaned_prompt

    def _bearer_token(authorization: str | None) -> str:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer":
            return ""
        return token.strip()

    async def _require_framework_status_token(
        training_run_id: str, authorization: str | None
    ) -> None:
        try:
            expected_token = str(
                (
                    await run_in_threadpool(
                        vol_get,
                        MetadataStore.FRAMEWORK_STATUS_TOKENS,
                        training_run_id,
                    )
                ).get("token", "")
            )
        except KeyError:
            expected_token = ""
        if not expected_token or not _secrets.compare_digest(
            _bearer_token(authorization), expected_token
        ):
            raise HTTPException(status_code=403, detail="Invalid status token")

    # ── Training runs ────────────────────────────────────────────────────

    @web.get("/api/runs")
    async def runs():
        try:
            data = await get_cached_list("runs", load_runs)
        except Exception:
            data = []
        return JSONResponse(data)

    @web.post("/api/framework-status")
    async def framework_status(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        training_run_id = str(payload.get("training_run_id", "") or "").strip()
        phase = str(payload.get("phase", "") or "").strip()
        if not training_run_id or not phase:
            raise HTTPException(
                status_code=400,
                detail="training_run_id and phase are required",
            )

        try:
            run = await run_in_threadpool(TrainingRun.from_id, training_run_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"TrainingRun {training_run_id!r} not found",
            )
        await _require_framework_status_token(training_run_id, authorization)

        status = _resolve_framework_status(phase, str(run.framework.value))
        if status is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported framework status {phase!r} for {run.framework.value}",
            )

        run.framework_status = status
        metadata = dict(run.metadata or {})
        progress = {
            "phase": status.value,
            "updated_at": int(time.time()),
        }
        # is_active: True = stage is actually running on hardware; False =
        # we've marked the stage but it's queuing for a GPU. Sent by the
        # orchestration code in common/train.py (queue=False) and by the
        # Modal function itself when its body starts executing (active=True).
        if "is_active" in payload:
            progress["is_active"] = bool(payload.get("is_active"))
        for src, dst in (
            ("progress_current", "current"),
            ("progress_total", "total"),
            ("progress_unit", "unit"),
            ("rollout_id", "rollout_id"),
            ("step_id", "step_id"),
        ):
            if src not in payload:
                continue
            value = payload.get(src)
            if dst in ("current", "total", "rollout_id", "step_id"):
                value = _optional_int(value)
                if value is None:
                    continue
            progress[dst] = value
        existing_progress = metadata.get("framework_progress")
        if isinstance(existing_progress, dict):
            # Drop the existing is_active when we get a fresh transition into
            # a different phase — it shouldn't bleed across stage changes.
            if existing_progress.get("phase") != progress.get("phase"):
                existing_progress = {
                    k: v for k, v in existing_progress.items() if k != "is_active"
                }
            progress = {**existing_progress, **progress}
        metadata["framework_progress"] = progress
        run.metadata = metadata
        await run.save_async()
        cache_entries["runs"] = (0.0, [])
        return JSONResponse({"status": "ok", "framework_status": status.value})

    # ── Training rollouts ────────────────────────────────────────────────

    @web.post("/api/training-rollouts")
    async def training_rollout(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        training_run_id = str(payload.get("training_run_id", "") or "").strip()
        rollout_id_raw = payload.get("rollout_id")
        if not training_run_id or rollout_id_raw is None:
            raise HTTPException(
                status_code=400,
                detail="training_run_id and rollout_id are required",
            )
        try:
            rollout_id = int(rollout_id_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="rollout_id must be an integer")

        samples_raw = payload.get("samples") or []
        if not isinstance(samples_raw, list):
            raise HTTPException(status_code=400, detail="samples must be a list")

        try:
            run = await run_in_threadpool(TrainingRun.from_id, training_run_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"TrainingRun {training_run_id!r} not found",
            )
        await _require_framework_status_token(training_run_id, authorization)

        result = TrainingRolloutResult(
            training_run_id=training_run_id,
            rollout_id=rollout_id,
            created_at=_optional_int(payload.get("created_at")) or int(time.time()),
            samples=samples_raw,
            metrics=payload.get("metrics") or {},
            rollout_time=(
                float(payload["rollout_time"])
                if isinstance(payload.get("rollout_time"), (int, float))
                else None
            ),
        )
        await result.save_async()

        metadata = dict(run.metadata or {})
        metadata["latest_rollout"] = {
            "rollout_id": result.rollout_id,
            "mean": result.mean,
            "total": result.total,
            "created_at": result.created_at,
        }
        run.metadata = metadata
        await run.save_async()
        cache_entries["runs"] = (0.0, [])

        return JSONResponse(
            {"status": "ok", "rollout_id": result.rollout_id, "mean": result.mean}
        )

    @web.get("/api/runs/{training_run_id}/rollouts")
    async def list_run_rollouts(training_run_id: str):
        summaries = await run_in_threadpool(
            TrainingRolloutResult.list_summaries_for_run, training_run_id
        )
        return JSONResponse(summaries)

    @web.get("/api/runs/{training_run_id}/rollouts/{rollout_id}")
    async def get_run_rollout(training_run_id: str, rollout_id: int):
        key = f"{training_run_id}__{int(rollout_id):08d}"
        try:
            data = await run_in_threadpool(
                vol_get, MetadataStore.TRAINING_ROLLOUTS, key
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Rollout {rollout_id} for run {training_run_id!r} not found",
            )
        if isinstance(data, dict):
            _apply_parsed(data.get("samples"))
        return JSONResponse(data)

    # ── Live Modal log stream (SSE, pure pass-through) ───────────────────

    @web.get("/api/runs/{training_run_id}/logs/stream")
    async def stream_run_logs(
        training_run_id: str,
        request: Request,
        search: str = "",
        max_lines_per_sec: int = 0,
    ):
        """Server-Sent Events stream of the underlying Modal app's logs.

        Pure pass-through: we open a long-poll ``AppGetLogs`` stream against
        the run's ``modal_app_id`` and forward each batch as an SSE ``data``
        event. Nothing is persisted on the dashboard side.

        Query params:
          - ``search``: case-insensitive substring filter; lines that don't
            match are silently dropped.
          - ``max_lines_per_sec``: integer rate cap. Lines exceeding the cap
            in any 1-second window are dropped; a single ``dropped`` event
            is emitted per second summarizing the count.
        """
        import json

        from modal.client import _Client
        from modal_proto import api_pb2

        try:
            run = await TrainingRun.from_id_async(training_run_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"TrainingRun {training_run_id!r} not found",
            )

        app_id = (run.modal_app_id or "").strip()
        if not app_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"TrainingRun {training_run_id!r} has no modal_app_id "
                    "yet — logs not available."
                ),
            )

        search_lower = search.strip().lower() if search else ""
        rate_cap = max(0, int(max_lines_per_sec or 0))

        async def event_stream():
            try:
                token_id = os.environ.get("MODAL_TOKEN_ID", "")
                token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
                if not token_id or not token_secret:
                    yield (
                        "event: error\n"
                        f"data: {json.dumps({'error': 'No Modal credentials configured. Run training-gym setup.'})}\n\n"
                    )
                    return
                client = await _Client.from_credentials(token_id, token_secret)
            except Exception as exc:
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': f'auth failed: {exc!s}'})}\n\n"
                )
                return

            last_entry_id = ""
            window_start = time.monotonic()
            window_emitted = 0
            window_dropped = 0
            consecutive_errors = 0
            max_consecutive_errors = 10

            def _drain_drop_event() -> str | None:
                nonlocal window_dropped
                if not window_dropped:
                    return None
                payload = {"dropped": window_dropped}
                window_dropped = 0
                return f"event: dropped\ndata: {json.dumps(payload)}\n\n"

            try:
                while True:
                    if await request.is_disconnected():
                        return
                    req = api_pb2.AppGetLogsRequest(
                        app_id=app_id,
                        timeout=55,
                        last_entry_id=last_entry_id,
                    )
                    try:
                        async for log_batch in client.stub.AppGetLogs.unary_stream(req):
                            if await request.is_disconnected():
                                return
                            consecutive_errors = 0
                            if log_batch.entry_id:
                                last_entry_id = log_batch.entry_id
                            for log in log_batch.items:
                                if not log.data:
                                    continue
                                if (
                                    search_lower
                                    and search_lower not in log.data.lower()
                                ):
                                    continue

                                now = time.monotonic()
                                if now - window_start >= 1.0:
                                    drop_event = _drain_drop_event()
                                    if drop_event:
                                        yield drop_event
                                    window_start = now
                                    window_emitted = 0

                                if rate_cap and window_emitted >= rate_cap:
                                    window_dropped += 1
                                    continue

                                window_emitted += 1
                                payload = {
                                    "task_id": log_batch.task_id,
                                    "line": log.data,
                                }
                                ts = getattr(log, "timestamp", 0) or 0
                                if ts:
                                    payload["ts"] = ts
                                yield f"data: {json.dumps(payload)}\n\n"
                            if log_batch.app_done:
                                drop_event = _drain_drop_event()
                                if drop_event:
                                    yield drop_event
                                yield "event: done\ndata: {}\n\n"
                                return
                    except asyncio.CancelledError:
                        return
                    except Exception as exc:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            yield (
                                "event: error\n"
                                f"data: {json.dumps({'error': f'log stream failed after {consecutive_errors} retries: {exc!s}'})}\n\n"
                            )
                            return
                        backoff = min(2 ** (consecutive_errors - 1), 10)
                        yield (
                            "event: reconnect\n"
                            f"data: {json.dumps({'reason': str(exc)})}\n\n"
                        )
                        try:
                            await asyncio.sleep(backoff)
                        except asyncio.CancelledError:
                            return
            finally:
                pass

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Train results ────────────────────────────────────────────────────

    @web.get("/api/train-results")
    async def train_results():
        try:
            data = await get_cached_list("train_results", load_train_results)
        except Exception:
            data = []
        return JSONResponse(data)

    @web.get("/api/train-results/{training_run_id}")
    async def train_result(training_run_id: str):
        try:
            data = await run_in_threadpool(
                vol_get, MetadataStore.TRAIN_RESULTS, training_run_id
            )
            return JSONResponse(data)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"TrainResult {training_run_id!r} not found",
            )

    # ── Eval results ─────────────────────────────────────────────────────

    @web.get("/api/evals")
    async def evals():
        try:
            data = await get_cached_list("evals", load_eval_summaries)
        except Exception:
            data = []
        return JSONResponse(data)

    @web.get("/api/evals/{eval_id}")
    async def eval_detail(eval_id: str):
        try:
            data = await run_in_threadpool(vol_get, MetadataStore.EVAL_RESULTS, eval_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"EvalResult {eval_id!r} not found",
            )
        if isinstance(data, dict):
            _apply_parsed(data.get("rows"))
        return JSONResponse(data)

    # ── Deployments ──────────────────────────────────────────────────────

    @web.get("/api/deployments")
    async def deployments():
        try:
            data = await get_cached_list("deployments", load_deployments)
        except Exception:
            data = []
        return JSONResponse(data)

    # ── Compaction (on-demand repair) ─────────────────────────────────────

    @web.post("/api/compact")
    async def compact():
        """Trigger an on-demand compaction (the scheduled cron is the primary path)."""
        await run_in_threadpool(_run_compact_sync)
        return JSONResponse({"status": "compacted"})

    # ── SPA fallback ─────────────────────────────────────────────────────

    @web.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(f"{STATIC_DIR}/index.html")

    return web
