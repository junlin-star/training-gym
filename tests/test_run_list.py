from __future__ import annotations

from modal_training_gym.common.run_list import (
    filter_run_summaries,
    run_list_field_metadata,
)
from modal_training_gym.common.run_summary import RunSummary


def _summary(**overrides) -> RunSummary:
    values = {
        "training_run_id": "run-1",
        "run_id": "run-1",
        "status": "running",
        "display_status": "pending",
        "display_stage": "Training",
        "framework_status": "training",
        "model": "org/model",
        "dataset": "org/data",
        "recipe": "slime",
        "group_id": "nightly",
        "created_at": 100,
        "updated_at": 200,
    }
    values.update(overrides)
    return RunSummary(**values)


def test_schema_metadata_drives_columns_and_filters():
    fields = run_list_field_metadata()

    assert list(fields) == [
        "run_id",
        "display_status",
        "display_stage",
        "model",
        "dataset",
        "recipe",
        "group_id",
        "created_at",
        "updated_at",
    ]
    assert {name for name, metadata in fields.items() if metadata["filterable"]} == {
        "display_status",
        "model",
        "dataset",
        "recipe",
        "group_id",
    }


def test_filtering_uses_projection_values_and_update_recency():
    older = _summary(run_id="older", training_run_id="older")
    newer = _summary(
        run_id="newer",
        training_run_id="newer",
        status="failed",
        display_status="failed",
        created_at=250,
        updated_at=300,
    )

    assert filter_run_summaries(
        [older, newer],
        filters={"display_status": "FAILED"},
    ) == [newer]
    assert filter_run_summaries([older, newer], since=225, limit=1) == [newer]
