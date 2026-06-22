"""training-gym cleanup — delete old failed run metadata from the shared volume."""

from __future__ import annotations

import time

from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items,
    vol_put_summary_items,
    vol_remove,
)


def cleanup(*, older_than_days: int = 7, dry_run: bool = False) -> None:
    cutoff = int(time.time()) - older_than_days * 86400
    stale_cutoff = int(time.time()) - 24 * 3600

    # Mark stale "running" runs (no update in >24h) as failed — these are
    # orphaned runs whose container was killed without a graceful shutdown.
    raw_runs_all = vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) or []
    stale_runs: list[TrainingRun] = []
    for raw in raw_runs_all:
        if not isinstance(raw, dict):
            continue
        if "training_run_id" not in raw and "run_id" in raw:
            raw["training_run_id"] = raw["run_id"]
        try:
            r = TrainingRun.model_validate(raw)
        except Exception:
            continue
        if r.status == TrainingRunStatus.RUNNING and (r.updated_at or r.started_at or 0) < stale_cutoff:
            stale_runs.append(r)

    if stale_runs:
        print(f"Marking {len(stale_runs)} stale pending run(s) as failed (no update in >24h):")
        for r in stale_runs:
            age_h = (time.time() - (r.updated_at or r.started_at or r.created_at)) / 3600
            print(f"  {r.training_run_id}  (last update {age_h:.0f}h ago)")
            if not dry_run:
                r.status = TrainingRunStatus.FAILED
                r.ended_at = r.updated_at or int(time.time())
                if r.completed_at is None:
                    r.completed_at = r.ended_at
                r.duration_seconds = max(0, r.ended_at - r.started_at) if r.started_at else None
                try:
                    r.save()
                except Exception as exc:
                    print(f"    WARNING: failed to save: {exc}")
        if dry_run:
            print("  (dry run — no changes made)")
        print()

    print(f"Finding failed runs older than {older_than_days} days …")

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
        if r.status == TrainingRunStatus.FAILED
        and (r.created_at or r.started_at or 0) < cutoff
        and (r.created_at or r.started_at or 0) > 0
    ]

    if not targets:
        print("Nothing to clean up.")
        return

    target_ids = {r.training_run_id for r in targets}
    print(f"{'Would delete' if dry_run else 'Deleting'} {len(targets)} failed run(s):\n")
    for r in sorted(targets, key=lambda r: r.created_at or 0):
        age_days = (time.time() - (r.created_at or r.started_at)) / 86400
        print(f"  {r.training_run_id}  ({age_days:.0f}d ago)")

    if dry_run:
        print("\nDry run — no changes made. Remove --dry-run to delete.")
        return

    deleted_runs = 0
    deleted_rollouts = 0
    deleted_tokens = 0

    for r in targets:
        rid = r.training_run_id
        if vol_remove(MetadataStore.TRAINING_RUNS, rid):
            deleted_runs += 1
        vol_remove(MetadataStore.FRAMEWORK_STATUS_TOKENS, rid)
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
        item
        for item in run_summary
        if item.get("training_run_id") not in target_ids
    ]
    if len(kept_run_items) != len(run_summary):
        vol_put_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY, kept_run_items)

    print(
        f"\nDone: {deleted_runs} run(s), {deleted_rollouts} rollout(s), "
        f"{deleted_tokens} token(s) removed."
    )
