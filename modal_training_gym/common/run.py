"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.status import FrameworkStatus
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_get_async,
    vol_put,
    vol_put_async,
    vol_upsert_summary_item,
    vol_upsert_summary_item_async,
)

TRAINING_RUNS_STORE_NAME = MetadataStore.TRAINING_RUNS.value


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
    metadata: dict[str, Any] | None = None

    def _summary_sort_key(self, item: dict[str, Any]) -> tuple[int, str]:
        return (
            int(item.get("created_at", 0) or 0),
            str(item.get("training_run_id", "")),
        )

    def _touch(self) -> None:
        self.updated_at = int(time.time())

    def save(self) -> None:
        self._touch()
        payload = self.model_dump(mode="json")
        vol_put(MetadataStore.TRAINING_RUNS, self.training_run_id, payload)
        vol_upsert_summary_item(
            MetadataStore.TRAINING_RUNS_SUMMARY,
            payload,
            item_id_key="training_run_id",
            sort_key=self._summary_sort_key,
            reverse=True,
        )

    async def save_async(self) -> None:
        self._touch()
        payload = self.model_dump(mode="json")
        await vol_put_async(MetadataStore.TRAINING_RUNS, self.training_run_id, payload)
        await vol_upsert_summary_item_async(
            MetadataStore.TRAINING_RUNS_SUMMARY,
            payload,
            item_id_key="training_run_id",
            sort_key=self._summary_sort_key,
            reverse=True,
        )

    @classmethod
    def from_id(cls, run_id: str) -> "TrainingRun":
        return cls.model_validate(vol_get(MetadataStore.TRAINING_RUNS, run_id))

    @classmethod
    async def from_id_async(cls, run_id: str) -> "TrainingRun":
        data = await vol_get_async(MetadataStore.TRAINING_RUNS, run_id)
        return cls.model_validate(data)


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
    metadata = dict(run.metadata or {})
    try:
        attempt_count = int(metadata.get("attempt_count") or 0) + 1
    except (TypeError, ValueError):
        attempt_count = 1

    metadata["attempt_count"] = attempt_count
    metadata["last_attempt_started_at"] = started_at
    metadata["last_attempt_status"] = "running"
    metadata.pop("terminal_reason", None)

    run.status = TrainingRunStatus.RUNNING
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
    metadata["last_attempt_status"] = status
    metadata["last_attempt_ended_at"] = ended_at
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
