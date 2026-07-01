from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from modal_training_gym.common.run import TrainingRun, TrainingRunStatus


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
) -> None:
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
