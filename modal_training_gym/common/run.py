"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, PrivateAttr, computed_field, field_validator

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.status import FrameworkStatus, resolve_framework_status
from modal_training_gym.utils.metadata import (
    MetadataStore,
    _step_times_dict,
    vol_get,
    vol_put_with_summary,
)

if TYPE_CHECKING:
    from modal_training_gym.common.train_result import TrainResult
    from modal_training_gym.common.training_rollout import TrainingRolloutResult

TRAINING_RUNS_STORE_NAME = MetadataStore.TRAINING_RUNS.value


class FrameworkStatusUpdate(BaseModel):
    """Body of ``POST /api/framework-status``.

    Reporters (``common/status_reporter.py``, slime's ``phase_reporting``) post
    more keys than the dashboard tracks (``app_name``, ``metrics``, …); those
    extras are ignored. Progress values come from loosely-typed framework args,
    so anything non-numeric or negative reads as "not provided".
    """

    training_run_id: str
    phase: str
    attempt_id: str | None = None
    is_active: bool | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    progress_unit: str | None = None
    rollout_id: int | None = None
    step_id: int | None = None
    step_event: str = ""

    @field_validator(
        "progress_current", "progress_total", "rollout_id", "step_id", mode="before"
    )
    @classmethod
    def _non_negative_int_or_none(cls, value: object) -> int | None:
        if not isinstance(value, (int, float, str)):
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None


class TrainingRunStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingRun(BaseModel):
    training_run_id: str
    modal_app_id: str = ""
    modal_app_url: str = ""
    framework: Framework
    config: Any
    dataset_id: str = ""
    deployment_id: str = ""
    status: TrainingRunStatus = TrainingRunStatus.RUNNING
    framework_status: FrameworkStatus | None = None
    created_at: int = 0
    started_at: int = 0
    ended_at: int | None = None
    completed_at: int | None = None
    updated_at: int = 0
    duration_seconds: int | None = None
    step_times: dict[str, dict[str, int | None]] | None = None
    # Terminal failure message (Ray driver error / exception) for a failed run,
    # so the cause is queryable from the record and shown on the dashboard even
    # after logs roll off. None while running / on success.
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    # Handle to the spawned ``app.train`` Modal FunctionCall so a launched run
    # can be waited on (see ``result()`` / ``__await__``). Empty until the run
    # is actually spawned by ``TrainConfig.launch()``.
    function_call_id: str = ""

    # Runtime-only handles attached by ``TrainConfig.launch()``; never persisted.
    _function_call: Any = PrivateAttr(default=None)
    _status_display: Any = PrivateAttr(default=None)

    @computed_field
    @property
    def group_id(self) -> str | None:
        """Group id, derived from ``metadata`` (its single source of truth).

        Exposed as a top-level attribute/serialized field so the dashboard and
        other callers can read ``run.group_id`` directly, but not stored
        separately — ``TrainConfig`` writes it into ``metadata`` (and
        ``metadata['group_tags']``), and this reads it back so the two can never
        drift out of sync.
        """
        meta = self.metadata or {}
        gid = meta.get("group_id")
        if gid:
            return str(gid)
        tags = meta.get("group_tags")
        if isinstance(tags, dict) and tags.get("group_id"):
            return str(tags["group_id"])
        return None

    # ── Launch-handle behavior (waiting on the spawned run) ──────────────────

    @property
    def function_call(self) -> Any:
        if self._function_call is not None:
            return self._function_call
        import modal

        return modal.FunctionCall.from_id(self.function_call_id)

    def result(
        self,
        *,
        timeout: float | None = None,
        stop_app_on_success: bool = True,
    ) -> TrainResult:
        """Block until the spawned training call finishes and return its TrainResult."""
        from modal_training_gym.common.modal_lifecycle import stop_app
        from modal_training_gym.common.status_reporter import (
            flush as flush_status_reporter,
        )
        from modal_training_gym.common.train_result import TrainResult

        if self._status_display is not None:
            self._status_display.start_polling(self.training_run_id)
        try:
            result_dict = self.function_call.get(timeout=timeout)
        finally:
            if self._status_display is not None:
                self._status_display.stop_polling()
            flush_status_reporter(timeout_seconds=2.0)

        if stop_app_on_success and self.modal_app_id:
            stop_app(self.modal_app_id)
        result = TrainResult(**TrainResult._parse_model_config(result_dict))
        print(f"Training complete: {result.training_run_id}")
        return result

    def __await__(self):
        import asyncio

        async def _wait() -> TrainResult:
            return await asyncio.to_thread(self.result)

        return _wait().__await__()

    def _summary_sort_key(self, item: dict[str, Any]) -> tuple[int, str]:
        return (
            int(item.get("created_at", 0) or 0),
            str(item.get("training_run_id", "")),
        )

    def apply_framework_status(
        self, update: FrameworkStatusUpdate
    ) -> FrameworkStatus | None:
        """Apply one framework-status report to this run (without saving).

        Sets ``framework_status``, merges the report into the
        ``framework_progress`` metadata blob, and records step start/finish
        times. Returns the resolved status, or ``None`` (run untouched) when
        the update is rejected. Use ``framework_status_rejection`` when a
        caller needs to distinguish an invalid phase from a stale attempt or
        a terminal run.
        """
        if self.framework_status_rejection(update) is not None:
            return None

        metadata = dict(self.metadata or {})
        active_attempt_id = str(metadata.get("active_attempt_id") or "").strip()
        # Once a remote training attempt exists, only that immutable attempt
        # may mutate logical-run progress. This rejects delayed HTTP requests
        # from a dead container and also prevents reports arriving after the
        # logical run became terminal from regressing its presentation state.
        status = resolve_framework_status(update.phase, str(self.framework.value))
        assert status is not None

        self.framework_status = status
        progress: dict[str, Any] = {
            "phase": status.value,
            "updated_at": int(time.time()),
        }
        # is_active: True = stage is actually running on hardware; False =
        # we've marked the stage but it's queuing for a GPU. Sent by the
        # orchestration code in common/train.py (queue=False) and by the
        # Modal function itself when its body starts executing (active=True).
        if update.is_active is not None:
            progress["is_active"] = update.is_active
        for key, value in (
            ("current", update.progress_current),
            ("total", update.progress_total),
            ("unit", update.progress_unit),
            ("rollout_id", update.rollout_id),
            ("step_id", update.step_id),
        ):
            if value is not None:
                progress[key] = value

        existing_progress = metadata.get("framework_progress")
        if isinstance(existing_progress, dict):
            # Drop the existing is_active when we get a fresh transition into
            # a different phase — it shouldn't bleed across stage changes.
            if existing_progress.get("phase") != progress["phase"]:
                existing_progress = {
                    k: v for k, v in existing_progress.items() if k != "is_active"
                }
            progress = {**existing_progress, **progress}
        metadata["framework_progress"] = progress
        self.metadata = metadata

        current_step = progress.get("current")
        step_event = update.step_event.strip()
        if (
            step_event in ("start", "finish")
            and isinstance(current_step, int)
            and current_step > 0
        ):
            step_times = _step_times_dict()
            attempt_namespace = active_attempt_id or "legacy"
            step_times[
                f"{self.training_run_id}:{attempt_namespace}:"
                f"{current_step}:{step_event}"
            ] = time.time()
        return status

    def framework_status_rejection(
        self,
        update: FrameworkStatusUpdate,
    ) -> Literal["attempt_mismatch", "terminal_run", "unsupported_phase"] | None:
        """Explain why a framework-status report would be rejected."""
        metadata = dict(self.metadata or {})
        active_attempt_id = str(metadata.get("active_attempt_id") or "").strip()
        reported_attempt_id = str(update.attempt_id or "").strip()
        if active_attempt_id and reported_attempt_id != active_attempt_id:
            return "attempt_mismatch"
        if self.status != TrainingRunStatus.RUNNING:
            return "terminal_run"
        if resolve_framework_status(update.phase, str(self.framework.value)) is None:
            return "unsupported_phase"
        return None

    def record_latest_rollout(self, rollout: TrainingRolloutResult) -> None:
        """Stamp a just-saved rollout's summary onto this run's metadata."""
        metadata = dict(self.metadata or {})
        metadata["latest_rollout"] = {
            "rollout_id": rollout.rollout_id,
            "mean": rollout.mean,
            "total": rollout.total,
            "created_at": rollout.created_at,
        }
        self.metadata = metadata

    def _touch(self) -> None:
        self.updated_at = int(time.time())

    def save(
        self,
        *,
        is_async: bool = False,
        event_kind: Literal["started", "snapshot", "failure", "terminal"] | None = None,
    ) -> None | Awaitable[None]:
        """Durably publish attempt state, then refresh the mutable cache.

        Once an attempt exists, the append-only event is authoritative.  The
        historical per-run JSON and summary remain useful caches, but a racing
        cache write cannot regress the materialized attempt or terminal state.
        """
        self._touch()
        payload = self.model_dump(mode="json")
        identity_exists = bool(_attempt_identity_from_payload(payload))
        journal_enabled = identity_exists and training_run_event_journal_enabled(
            payload
        )
        if event_kind is None and journal_enabled:
            event_kind = (
                "terminal" if self.status != TrainingRunStatus.RUNNING else "snapshot"
            )
        if event_kind is not None and not journal_enabled:
            # Legacy recipes keep their historical mutable reporting path.
            # The explicit journal contract is enabled only for committed
            # attempt mode, whose retry semantics require it.
            event_kind = None
        if event_kind is not None and not identity_exists:
            raise RuntimeError("cannot journal a training run without an attempt")

        async def _save_async() -> None:
            if event_kind is not None:
                from modal_training_gym.common.run_events import (
                    append_training_run_event,
                )

                await append_training_run_event(
                    payload,
                    kind=event_kind,
                    is_async=True,
                )
            await self._save_cache_payload(payload, is_async=True)

        if is_async:
            return _save_async()
        if event_kind is not None:
            from modal_training_gym.common.run_events import append_training_run_event

            append_training_run_event(
                payload,
                kind=event_kind,
            )
        self._save_cache_payload(payload)
        return None

    def save_cache(self, *, is_async: bool = False) -> None | Awaitable[None]:
        """Refresh only the disposable per-run and summary cache."""
        self._touch()
        return self._save_cache_payload(
            self.model_dump(mode="json"),
            is_async=is_async,
        )

    def _save_cache_payload(
        self,
        payload: dict[str, Any],
        *,
        is_async: bool = False,
    ) -> None | Awaitable[None]:
        return vol_put_with_summary(
            MetadataStore.TRAINING_RUNS,
            self.training_run_id,
            payload,
            summary_store=MetadataStore.TRAINING_RUNS_SUMMARY,
            item_id_key="training_run_id",
            sort_key=self._summary_sort_key,
            reverse=True,
            is_async=is_async,
        )

    @classmethod
    def from_id(
        cls, run_id: str, *, is_async: bool = False
    ) -> TrainingRun | Awaitable[TrainingRun]:
        if is_async:

            async def _run() -> TrainingRun:
                try:
                    base = await vol_get(
                        MetadataStore.TRAINING_RUNS,
                        run_id,
                        is_async=True,
                    )
                except KeyError:
                    base = None
                if base is not None and not training_run_event_journal_enabled(base):
                    return cls.model_validate(base)
                from modal_training_gym.common.run_events import (
                    load_materialized_training_run,
                )

                data = await load_materialized_training_run(
                    run_id,
                    base,
                    is_async=True,
                )
                if data is None:
                    raise KeyError(run_id)
                return cls.model_validate(data)

            return _run()
        try:
            base = vol_get(MetadataStore.TRAINING_RUNS, run_id)
        except KeyError:
            base = None
        if base is not None and not training_run_event_journal_enabled(base):
            return cls.model_validate(base)
        from modal_training_gym.common.run_events import (
            load_materialized_training_run,
        )

        data = load_materialized_training_run(run_id, base)
        if data is None:
            raise KeyError(run_id)
        return cls.model_validate(data)


def _attempt_identity_from_payload(
    payload: dict[str, Any],
) -> tuple[int, str] | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    attempt_count = metadata.get("attempt_count")
    attempt_id = metadata.get("active_attempt_id")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
        or not isinstance(attempt_id, str)
        or not attempt_id.strip()
    ):
        return None
    return attempt_count, attempt_id


def training_run_event_journal_enabled(payload: dict[str, Any]) -> bool:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("attempt_mode") == "committed":
        return True
    config = payload.get("config")
    if not isinstance(config, dict):
        return False
    recipe = config.get("recipe")
    return isinstance(recipe, dict) and recipe.get("attempt_mode") == "committed"


async def materialize_training_run_summaries(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace mutable summary rows with append-only journal projections.

    The list cache can be the last loser of a status/terminal write race.  The
    dashboard must therefore treat it only as an index of run IDs and project
    every attempt-backed row through ``TrainingRun.from_id`` before display.
    """
    import asyncio

    async def _materialize(item: dict[str, Any]) -> dict[str, Any]:
        run_id = str(item.get("training_run_id") or "").strip()
        if not run_id or not training_run_event_journal_enabled(item):
            return item
        try:
            run = await TrainingRun.from_id(run_id, is_async=True)
        except KeyError:
            return item
        except Exception as exc:
            # Do not silently present a last-write-wins cache value when the
            # authority is corrupt. Surface an explicit fail-closed diagnostic
            # for this row while leaving other runs visible.
            failed = copy.deepcopy(item)
            metadata = dict(failed.get("metadata") or {})
            message = f"{type(exc).__name__}: {exc}"
            metadata["failure_reporting_integrity_error"] = message
            failed["metadata"] = metadata
            failed["status"] = TrainingRunStatus.FAILED.value
            failed["error_message"] = message
            return failed
        return run.model_dump(mode="json")

    return list(await asyncio.gather(*(_materialize(item) for item in items)))


def _resume_checkpoint(path: str, name: str, iteration: int | None) -> dict[str, Any]:
    return {
        "resume_checkpoint_path": path,
        "resume_checkpoint_name": name,
        "resume_from_iteration": iteration,
    }


def has_torch_dist_checkpoint(
    save_path: str,
    *,
    is_complete: Callable[[str], bool] | None = None,
) -> bool:
    return torch_dist_resume_checkpoint(save_path, is_complete=is_complete) is not None


def torch_dist_resume_checkpoint(
    save_path: str,
    *,
    is_complete: Callable[[str], bool] | None = None,
) -> dict[str, Any] | None:
    if not os.path.isdir(save_path):
        return None

    is_complete = is_complete or os.path.isdir
    tracker_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker_path):
        try:
            with open(tracker_path) as f:
                marker = f.read().strip()
        except OSError:
            marker = ""
        if marker == "release":
            path = os.path.join(save_path, "release")
            return (
                _resume_checkpoint(path, "release", None) if is_complete(path) else None
            )
        if marker.isdigit():
            name = f"iter_{int(marker):07d}"
            path = os.path.join(save_path, name)
            return (
                _resume_checkpoint(path, name, int(marker))
                if is_complete(path)
                else None
            )

    try:
        candidates: list[tuple[int, str, str]] = []
        release_path = ""
        for entry in os.scandir(save_path):
            if not entry.is_dir() or not is_complete(entry.path):
                continue
            if entry.name == "release":
                release_path = entry.path
            elif entry.name.startswith("iter_"):
                try:
                    iteration = int(entry.name.removeprefix("iter_"))
                except ValueError:
                    continue
                candidates.append((iteration, entry.name, entry.path))
    except OSError:
        return None

    if candidates:
        iteration, name, path = max(candidates)
        return _resume_checkpoint(path, name, iteration)
    if release_path:
        return _resume_checkpoint(release_path, "release", None)
    return None


def run_scoped_save_root(save_root: str, training_run_id: str) -> str:
    save_root = str(save_root).rstrip("/") or "/"
    if os.path.basename(save_root) == training_run_id:
        return save_root
    return os.path.join(save_root, training_run_id)


def mark_training_attempt_started(
    run: TrainingRun,
    *,
    started_at: int,
) -> int:
    from modal_training_gym.common.attempts import new_attempt_id

    metadata = dict(run.metadata or {})
    # Migrate records written before per-attempt provenance existed before the
    # retry clears the terminal top-level field.  Otherwise a secondary retry
    # failure would be incorrectly promoted to the logical run's root cause.
    legacy_error = str(run.error_message or "").strip()
    raw_primary = metadata.get("primary_failure")
    has_primary = isinstance(raw_primary, dict) and bool(
        str(raw_primary.get("message") or "").strip()
    )
    if legacy_error and not has_primary:
        legacy_recorded_at = (
            metadata.get("last_attempt_ended_at")
            or run.ended_at
            or run.completed_at
            or started_at
        )
        metadata["primary_failure"] = {
            "message": legacy_error,
            "recorded_at": int(legacy_recorded_at),
        }

    try:
        attempt_count = int(metadata.get("attempt_count") or 0) + 1
    except (TypeError, ValueError):
        attempt_count = 1

    attempt_id = new_attempt_id()
    raw_attempts = metadata.get("attempts")
    attempts = (
        [dict(item) for item in raw_attempts if isinstance(item, dict)]
        if isinstance(raw_attempts, list)
        else []
    )
    for prior in attempts:
        if prior.get("status") == "running":
            prior["status"] = "interrupted_without_terminal_report"
            prior["ended_at"] = started_at
    attempts.append(
        {
            "attempt": attempt_count,
            "attempt_id": attempt_id,
            "started_at": started_at,
            "status": "running",
        }
    )
    metadata["attempts"] = attempts
    metadata["attempt_count"] = attempt_count
    metadata["active_attempt_id"] = attempt_id
    metadata["last_attempt_started_at"] = started_at
    metadata["last_attempt_status"] = "running"
    metadata.pop("terminal_reason", None)

    run.status = TrainingRunStatus.RUNNING
    # A retry is still the same logical run, but it is no longer terminal.
    # Keep the first failure in metadata["primary_failure"] for provenance while
    # ensuring the top-level field obeys its contract (None while running or
    # after a successful retry).
    run.error_message = None
    run.ended_at = None
    run.completed_at = None
    run.duration_seconds = None
    run.metadata = metadata
    return attempt_count


def mark_training_attempt_finished(
    run: TrainingRun,
    *,
    status: str,
    ended_at: int,
) -> None:
    metadata = dict(run.metadata or {})
    attempt_id = str(metadata.get("active_attempt_id") or "")
    raw_attempts = metadata.get("attempts")
    if isinstance(raw_attempts, list):
        attempts: list[Any] = []
        for raw in raw_attempts:
            if isinstance(raw, dict) and raw.get("attempt_id") == attempt_id:
                item = dict(raw)
                item["status"] = status
                item["ended_at"] = ended_at
                attempts.append(item)
            else:
                attempts.append(raw)
        metadata["attempts"] = attempts
    metadata["last_attempt_status"] = status
    metadata["last_attempt_ended_at"] = ended_at
    run.metadata = metadata


def record_training_attempt_cluster_identity(
    run: TrainingRun,
    cluster_identity: dict[str, Any],
) -> None:
    """Attach a write-once Modal cluster identity to the active attempt.

    Retries append a new attempt before calling this helper, so each physical
    Modal cluster remains attributable without rewriting prior-attempt
    provenance. Repeating the exact write is idempotent; a conflicting write
    fails closed.
    """
    metadata = dict(run.metadata or {})
    attempt_id = str(metadata.get("active_attempt_id") or "")
    if not attempt_id:
        raise RuntimeError("Cannot record Modal cluster identity without an attempt ID")

    raw_attempts = metadata.get("attempts")
    if not isinstance(raw_attempts, list):
        raise RuntimeError(
            "Cannot record Modal cluster identity without attempt records"
        )

    identity = copy.deepcopy(cluster_identity)
    attempts: list[Any] = []
    matches = 0
    for raw_attempt in raw_attempts:
        if (
            not isinstance(raw_attempt, dict)
            or raw_attempt.get("attempt_id") != attempt_id
        ):
            attempts.append(raw_attempt)
            continue

        matches += 1
        attempt = dict(raw_attempt)
        existing = attempt.get("modal_cluster")
        if existing is not None and existing != identity:
            raise RuntimeError(
                f"Modal cluster identity for attempt {attempt_id} is immutable"
            )
        attempt["modal_cluster"] = copy.deepcopy(identity)
        attempts.append(attempt)

    if matches != 1:
        raise RuntimeError(
            f"Expected exactly one active attempt record for {attempt_id}, got {matches}"
        )

    metadata["attempts"] = attempts
    run.metadata = metadata


def record_resume_checkpoint(
    run: TrainingRun,
    resume_checkpoint: dict[str, Any] | None,
) -> None:
    metadata = dict(run.metadata or {})
    if resume_checkpoint is None:
        metadata["resumed_from_checkpoint"] = False
        for key in (
            "resume_checkpoint_path",
            "resume_checkpoint_name",
            "resume_from_iteration",
        ):
            metadata.pop(key, None)
    else:
        metadata["resumed_from_checkpoint"] = True
        metadata.update(resume_checkpoint)
    run.metadata = metadata


def wandb_run_id_for_attempt(training_run_id: str, attempt_count: int) -> str:
    base_run_id = training_run_id[:8]
    return base_run_id if attempt_count <= 1 else f"{base_run_id}-a{attempt_count}"


def record_wandb_attempt(
    run: TrainingRun,
    *,
    entity: str,
    project: str,
    group: str,
    run_id: str,
    attempt_count: int,
) -> None:
    if not project or not run_id:
        return

    metadata = dict(run.metadata or {})
    raw_attempts = metadata.get("wandb_attempts")
    attempts = raw_attempts if isinstance(raw_attempts, list) else []
    attempt = {
        "attempt": attempt_count,
        "entity": entity,
        "project": project,
        "group": group,
        "run_id": run_id,
    }
    attempts = [
        existing
        for existing in attempts
        if not (
            isinstance(existing, dict)
            and (
                existing.get("attempt") == attempt_count
                or existing.get("run_id") == run_id
            )
        )
    ]
    attempts.append(attempt)
    metadata["wandb_attempts"] = sorted(
        attempts,
        key=lambda item: int(item.get("attempt") or 0) if isinstance(item, dict) else 0,
    )
    metadata["wandb_latest_run_id"] = run_id
    run.metadata = metadata


def select_accepted_wandb_attempt(
    run: TrainingRun,
    *,
    accepted_attempt_id: str,
    skipped_attempt_count: int,
) -> str:
    """Point durable reporting at the attempt that produced accepted state.

    A retry may discover that its parent already committed the terminal
    boundary.  That retry starts no Slime/W&B process, so its prospective W&B
    ID must not be returned as if it owned the accepted metrics.
    """
    metadata = dict(run.metadata or {})
    raw_attempts = metadata.get("attempts")
    accepted_attempt_count: int | None = None
    if isinstance(raw_attempts, list):
        for raw_attempt in raw_attempts:
            if (
                isinstance(raw_attempt, dict)
                and raw_attempt.get("attempt_id") == accepted_attempt_id
            ):
                try:
                    accepted_attempt_count = int(raw_attempt.get("attempt"))
                except (TypeError, ValueError):
                    accepted_attempt_count = None
                break
    if accepted_attempt_count is None:
        return ""

    raw_wandb_attempts = metadata.get("wandb_attempts")
    if not isinstance(raw_wandb_attempts, list):
        return ""

    selected_run_id = ""
    updated: list[Any] = []
    for raw_wandb_attempt in raw_wandb_attempts:
        if not isinstance(raw_wandb_attempt, dict):
            updated.append(raw_wandb_attempt)
            continue
        item = dict(raw_wandb_attempt)
        try:
            item_attempt_count = int(item.get("attempt"))
        except (TypeError, ValueError):
            item_attempt_count = None
        if item_attempt_count == accepted_attempt_count:
            selected_run_id = str(item.get("run_id") or "")
            item["status"] = "accepted"
        elif item_attempt_count == skipped_attempt_count:
            item["status"] = "not_started_terminal_parent"
            item["accepted_attempt_id"] = accepted_attempt_id
        updated.append(item)

    if not selected_run_id:
        return ""
    metadata["wandb_attempts"] = updated
    metadata["wandb_latest_run_id"] = selected_run_id
    metadata["wandb_accepted_run_id"] = selected_run_id
    run.metadata = metadata
    return selected_run_id
