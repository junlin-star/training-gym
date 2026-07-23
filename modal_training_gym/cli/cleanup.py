"""training-gym cleanup — delete old failed run metadata from the shared volume."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.step_timing import (
    aggregated_training_step_timing_keys,
    legacy_step_time_keys,
    training_timing_attempt_closed_key,
    training_timing_event_batch_keys,
)
from modal_training_gym.utils.metadata import (
    MetadataStore,
    _step_times_dict,
    vol_get_summary_items,
    vol_put_summary_items,
    vol_remove,
)

TERMINAL_STATUSES = frozenset({TrainingRunStatus.FAILED, TrainingRunStatus.CANCELLED})
TIMING_CLEANUP_BATCH_SIZE = 128


def _training_timing_cleanup_keys(run: TrainingRun) -> list[str | tuple[Any, ...]]:
    config = run.config if isinstance(run.config, Mapping) else {}
    recipe = config.get("recipe")
    try:
        num_steps = (
            int(recipe.get("num_rollout") or 0) if isinstance(recipe, Mapping) else 0
        )
    except (TypeError, ValueError):
        num_steps = 0
    try:
        training_attempts = int((run.metadata or {}).get("attempt_count") or 1)
    except (TypeError, ValueError):
        training_attempts = 1

    keys: list[str | tuple[Any, ...]] = []
    for training_attempt in range(1, max(training_attempts, 1) + 1):
        keys.append(
            training_timing_attempt_closed_key(
                run.training_run_id,
                training_attempt,
            )
        )
        if num_steps > 0:
            keys.extend(
                training_timing_event_batch_keys(
                    run.training_run_id,
                    training_attempt,
                    num_steps,
                )
            )
            keys.extend(
                aggregated_training_step_timing_keys(
                    run.training_run_id,
                    training_attempt,
                    num_steps,
                )
            )
    if num_steps > 0:
        keys.extend(legacy_step_time_keys(run.training_run_id, num_steps))
    return keys


async def _clear_step_times(keys: list[str | tuple[Any, ...]]) -> int:
    timing_event_store = _step_times_dict()
    deleted = 0
    for offset in range(0, len(keys), TIMING_CLEANUP_BATCH_SIZE):
        batch = keys[offset : offset + TIMING_CLEANUP_BATCH_SIZE]
        values = await asyncio.gather(
            *(timing_event_store.pop.aio(key, None) for key in batch)
        )
        deleted += sum(value is not None for value in values)
    return deleted


async def _clear_run_step_times(runs: list[TrainingRun]) -> int:
    deleted = 0
    for run in runs:
        deleted += await _clear_step_times(_training_timing_cleanup_keys(run))
    return deleted


def cleanup(*, older_than_days: int = 7, dry_run: bool = False) -> None:
    cutoff = int(time.time()) - older_than_days * 86400

    print(f"Finding failed/cancelled runs older than {older_than_days} days …")

    raw_runs = vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) or []
    runs: list[TrainingRun] = []
    for raw in raw_runs:
        if not isinstance(raw, dict):
            continue
        if "training_run_id" not in raw and "run_id" in raw:
            raw["training_run_id"] = raw["run_id"]
        try:
            runs.append(TrainingRun.model_validate(raw))
        except Exception:
            continue

    targets = [
        r
        for r in runs
        if r.status in TERMINAL_STATUSES
        and (r.created_at or r.started_at or 0) < cutoff
        and (r.created_at or r.started_at or 0) > 0
    ]

    if not targets:
        print("Nothing to clean up.")
        return

    target_ids = {r.training_run_id for r in targets}
    print(
        f"{'Would delete' if dry_run else 'Deleting'} {len(targets)} terminal run(s):\n"
    )
    for r in sorted(targets, key=lambda r: r.created_at or 0):
        age_days = (time.time() - (r.created_at or r.started_at)) / 86400
        print(f"  {r.training_run_id}  ({age_days:.0f}d ago, {r.status.value})")

    if dry_run:
        print("\nDry run — no changes made. Remove --dry-run to delete.")
        return

    deleted_runs = 0
    deleted_rollouts = 0
    deleted_tokens = 0

    try:
        deleted_timing_entries = asyncio.run(_clear_run_step_times(targets))
    except Exception as exc:
        print(f"Failed to clear step times; no run metadata was removed: {exc}")
        return

    for r in targets:
        rid = r.training_run_id
        if vol_remove(MetadataStore.TRAINING_RUNS, rid):
            deleted_runs += 1
        vol_remove(MetadataStore.FRAMEWORK_STATUS_TOKENS, rid)
        vol_remove(MetadataStore.TRAINING_LATEST_ROLLOUTS, rid)
        deleted_tokens += 1

    rollout_summary = (
        vol_get_summary_items(MetadataStore.TRAINING_ROLLOUTS_SUMMARY) or []
    )
    kept_rollout_items = []
    for item in rollout_summary:
        if item.get("training_run_id") in target_ids:
            key = item.get("summary_key") or ""
            if key:
                vol_remove(MetadataStore.TRAINING_ROLLOUTS, key)
                deleted_rollouts += 1
        else:
            kept_rollout_items.append(item)
    if deleted_rollouts:
        vol_put_summary_items(
            MetadataStore.TRAINING_ROLLOUTS_SUMMARY, kept_rollout_items
        )

    run_summary = vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) or []
    kept_run_items = [
        item for item in run_summary if item.get("training_run_id") not in target_ids
    ]
    if len(kept_run_items) != len(run_summary):
        vol_put_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY, kept_run_items)

    print(
        f"\nDone: {deleted_runs} run(s), {deleted_rollouts} rollout(s), "
        f"{deleted_tokens} token(s), and {deleted_timing_entries} timing "
        "entries removed."
    )
