"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from __future__ import annotations

import time
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
    metadata: dict[str, Any] | None = None

    def _summary_sort_key(self, item: dict[str, Any]) -> tuple[int, str]:
        return (
            int(item.get("created_at", 0) or 0),
            str(item.get("training_run_id", "")),
        )

    def _touch(self) -> None:
        self.updated_at = int(time.time())

    def start_step(self, current_step: int, started_at: int) -> int | None:
        if current_step <= 0:
            return None

        metadata = dict(self.metadata or {})
        progress = dict(metadata.get("framework_progress") or {})
        step_key = str(current_step)
        step_times = dict(metadata.get("step_times") or {})
        step_time = step_times.get(step_key)
        if not isinstance(step_time, dict):
            step_time = {}
        if not isinstance(step_time.get("started_at"), int):
            step_time["started_at"] = started_at
        step_times[step_key] = step_time
        progress["current"] = current_step
        progress["step_started_at"] = step_time["started_at"]
        self.metadata = {
            **metadata,
            "framework_progress": progress,
            "step_times": step_times,
        }
        return step_time["started_at"]

    def finish_current_step(
        self, ended_at: int, current_step: int | None = None
    ) -> None:
        metadata = dict(self.metadata or {})
        progress = dict(metadata.get("framework_progress") or {})

        if current_step is None:
            current_step = progress.get("current")
        if not isinstance(current_step, int):
            return

        step_key = str(current_step)
        step_times = dict(metadata.get("step_times") or {})
        step_time = step_times.get(step_key)
        if not isinstance(step_time, dict):
            return

        started_at = step_time.get("started_at")
        if not isinstance(started_at, int):
            return

        if isinstance(step_time.get("ended_at"), int):
            return

        if progress.get("current") == current_step:
            progress.pop("step_started_at", None)

        step_time.update(
            {
                "ended_at": ended_at,
                "duration_s": ended_at - started_at,
            }
        )
        step_times[step_key] = step_time

        self.metadata = {
            **metadata,
            "framework_progress": progress,
            "step_times": step_times,
        }

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
