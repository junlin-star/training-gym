"""Per-rollout training data — prompts, responses, rewards.

Mirrors the EvalResult shape but lives under its own MetadataStore so the
dashboard can list rollouts per training run without scanning the eval store.
Records are written by slime's `log_rollout_data` hook through the async
phase-reporter; reads happen via the dashboard's
``/api/runs/{id}/rollouts`` endpoint.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_get_summary_items,
    vol_list,
    vol_put,
    vol_put_async,
    vol_upsert_summary_item,
    vol_upsert_summary_item_async,
)


class TrainingRolloutSample(BaseModel):
    """One generated sample inside a rollout."""

    score: float = 0.0
    prompt: str = ""
    response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingRolloutResult(BaseModel):
    """All samples from one rollout of one training run."""

    training_run_id: str
    rollout_id: int
    created_at: int = 0
    samples: list[TrainingRolloutSample] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    rollout_time: float | None = None

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.score for s in self.samples) / len(self.samples)

    @property
    def storage_key(self) -> str:
        # One canonical file per (run, rollout). Zero-padded rollout id so
        # `vol_list` returns them in step order without needing a sort.
        return f"{self.training_run_id}__{self.rollout_id:08d}"

    def to_summary(self) -> dict[str, Any]:
        return {
            "training_run_id": self.training_run_id,
            "rollout_id": self.rollout_id,
            "created_at": self.created_at,
            "total": self.total,
            "mean": self.mean,
        }

    def _touch_created_at(self) -> None:
        if not self.created_at:
            self.created_at = int(time.time())

    @staticmethod
    def _summary_sort_key(item: dict[str, Any]) -> tuple[str, int]:
        return (
            str(item.get("training_run_id", "")),
            int(item.get("rollout_id", 0) or 0),
        )

    def _summary_item(self) -> dict[str, Any]:
        # summary_key keeps (run_id, rollout_id) uniqueness across runs.
        return {**self.to_summary(), "summary_key": self.storage_key}

    def save(self) -> None:
        self._touch_created_at()
        payload = self.model_dump(mode="json")
        vol_put(MetadataStore.TRAINING_ROLLOUTS, self.storage_key, payload)
        vol_upsert_summary_item(
            MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
            self._summary_item(),
            item_id_key="summary_key",
            sort_key=self._summary_sort_key,
            reverse=False,
        )

    async def save_async(self) -> None:
        self._touch_created_at()
        payload = self.model_dump(mode="json")
        await vol_put_async(
            MetadataStore.TRAINING_ROLLOUTS, self.storage_key, payload
        )
        await vol_upsert_summary_item_async(
            MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
            self._summary_item(),
            item_id_key="summary_key",
            sort_key=self._summary_sort_key,
            reverse=False,
        )

    @classmethod
    def from_storage_key(cls, key: str) -> "TrainingRolloutResult":
        return cls.model_validate(vol_get(MetadataStore.TRAINING_ROLLOUTS, key))

    @classmethod
    def list_for_run(cls, training_run_id: str) -> list["TrainingRolloutResult"]:
        """All rollouts for one training run, sorted by rollout_id."""
        items = vol_list(MetadataStore.TRAINING_ROLLOUTS)
        out: list[TrainingRolloutResult] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            if raw.get("training_run_id") != training_run_id:
                continue
            try:
                out.append(cls.model_validate(raw))
            except Exception:
                continue
        out.sort(key=lambda r: r.rollout_id)
        return out

    @classmethod
    def list_summaries_for_run(
        cls, training_run_id: str
    ) -> list[dict[str, Any]]:
        """Lightweight per-rollout summaries for one run, sorted by rollout_id."""
        items = (
            vol_get_summary_items(MetadataStore.TRAINING_ROLLOUTS_SUMMARY) or []
        )
        return sorted(
            (
                item
                for item in items
                if isinstance(item, dict)
                and item.get("training_run_id") == training_run_id
            ),
            key=lambda item: int(item.get("rollout_id", 0) or 0),
        )
