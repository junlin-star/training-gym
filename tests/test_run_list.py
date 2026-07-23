from __future__ import annotations

from modal_training_gym.common.run_list import (
    build_run_list_item,
    filter_run_summaries,
    run_list_field_metadata,
)
from modal_training_gym.common.run_summary import (
    FrameworkProgress,
    RunSummary,
    TrainResultSummary,
)


def _summary(**overrides) -> RunSummary:
    values = {
        "training_run_id": "run-1",
        "run_id": "run-1",
        "status": "running",
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


def test_projection_generates_dashboard_list_fields_from_summary():
    item = build_run_list_item(
        _summary(
            framework_status="download_model",
            framework_progress=FrameworkProgress(is_active=False),
        )
    )

    assert item.model_dump() == {
        "run_id": "run-1",
        "status": "pending",
        "stage": "Queuing for GPU — Downloading model",
        "model": "org/model",
        "dataset": "org/data",
        "recipe": "slime",
        "group": "nightly",
        "created_at": 100,
        "last_updated_at": 200,
    }


def test_projection_treats_train_result_as_completed_without_mutating_summary():
    summary = _summary(
        train_result=TrainResultSummary(training_run_id="run-1"),
    )

    assert summary.status == "running"
    assert build_run_list_item(summary).status == "completed"


def test_schema_metadata_drives_columns_and_filters():
    fields = run_list_field_metadata()

    assert list(fields) == [
        "run_id",
        "status",
        "stage",
        "model",
        "dataset",
        "recipe",
        "group",
        "created_at",
        "last_updated_at",
    ]
    assert {name for name, metadata in fields.items() if metadata["filterable"]} == {
        "status",
        "model",
        "dataset",
        "recipe",
        "group",
    }


def test_filtering_uses_projection_values_and_update_recency():
    older = _summary(run_id="older", training_run_id="older")
    newer = _summary(
        run_id="newer",
        training_run_id="newer",
        status="failed",
        created_at=250,
        updated_at=300,
    )

    assert filter_run_summaries([older, newer], filters={"status": "FAILED"}) == [newer]
    assert filter_run_summaries([older, newer], since=225, limit=1) == [newer]
