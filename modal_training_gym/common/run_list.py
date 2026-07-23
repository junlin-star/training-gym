"""The list-page projection of a full training-run summary."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from modal_training_gym.common.run_summary import FrameworkProgress, RunSummary


def _field(
    title: str,
    *,
    filterable: bool = False,
    timestamp: bool = False,
) -> FieldInfo:
    return Field(
        title=title,
        json_schema_extra={
            "filterable": filterable,
            "timestamp": timestamp,
        },
    )


class RunListItem(BaseModel):
    """Fields shown for one run on list-oriented clients."""

    run_id: str = _field("Run")
    status: str = _field("Status", filterable=True)
    stage: str = _field("Stage")
    model: str = _field("Model", filterable=True)
    dataset: str = _field("Dataset", filterable=True)
    recipe: str = _field("Recipe", filterable=True)
    group: str = _field("Group", filterable=True)
    created_at: int = _field("Created", timestamp=True)
    last_updated_at: int = _field("Last updated", timestamp=True)


_STAGE_LABELS = {
    "initializing": "Initializing",
    "download_model": "Downloading model",
    "convert_model": "Converting model",
    "prepare_dataset": "Preparing dataset",
    "initialize_rollouts": "Initializing rollouts",
    "generate_rollouts": "Generating rollouts",
    "evaluate_rollouts": "Evaluating rollouts",
    "compute_log_probs": "Computing log probs",
    "optimizer_step": "Optimizer step",
    "weight_sync": "Weight sync",
    "offload_rollout": "Offload rollout",
    "offload_train": "Offload train",
    "checkpoint_save": "Saving checkpoint",
    "training": "Training",
}
_QUEUEABLE_STAGES = {"download_model", "convert_model"}


def _status(summary: RunSummary) -> str:
    raw = summary.status.strip().lower()
    if summary.train_result is not None or raw == "completed":
        return "completed"
    if raw in {"cancelled", "stopped", "failed"}:
        return raw
    return "pending"


def _stage(status: str, progress: FrameworkProgress | None) -> str:
    normalized = status.strip().lower()
    if not normalized:
        return ""
    label = _STAGE_LABELS.get(normalized, normalized.replace("_", " ").title())
    if (
        normalized in _QUEUEABLE_STAGES
        and progress is not None
        and progress.is_active is False
    ):
        return f"Queuing for GPU — {label}"
    return label


def build_run_list_item(summary: RunSummary) -> RunListItem:
    """Generate the dashboard/CLI list fields from an existing summary."""
    return RunListItem(
        run_id=summary.run_id,
        status=_status(summary),
        stage=_stage(summary.framework_status, summary.framework_progress),
        model=summary.model,
        dataset=summary.dataset,
        recipe=summary.recipe,
        group=summary.group_id,
        created_at=summary.created_at,
        last_updated_at=summary.updated_at,
    )


def run_list_field_metadata() -> dict[str, dict[str, object]]:
    """Return ordered display metadata directly from RunListItem."""
    return {
        name: {
            "label": field.title or name.replace("_", " ").title(),
            **(
                field.json_schema_extra
                if isinstance(field.json_schema_extra, dict)
                else {}
            ),
        }
        for name, field in RunListItem.model_fields.items()
    }


def filter_run_summaries(
    summaries: Iterable[RunSummary],
    *,
    filters: dict[str, str] | None = None,
    since: int | None = None,
    limit: int | None = None,
) -> list[RunSummary]:
    """Filter summaries by their generated list fields and update recency."""
    metadata = run_list_field_metadata()
    active_filters = {
        name: value.strip().casefold()
        for name, value in (filters or {}).items()
        if metadata.get(name, {}).get("filterable") and value.strip()
    }

    selected: list[tuple[RunSummary, RunListItem]] = []
    for summary in summaries:
        item = build_run_list_item(summary)
        if since is not None and max(item.created_at, item.last_updated_at) < since:
            continue
        if any(
            str(getattr(item, name)).casefold() != expected
            for name, expected in active_filters.items()
        ):
            continue
        selected.append((summary, item))

    selected.sort(
        key=lambda pair: (pair[1].last_updated_at, pair[1].run_id),
        reverse=True,
    )
    if limit is not None:
        selected = selected[:limit]
    return [summary for summary, _item in selected]
