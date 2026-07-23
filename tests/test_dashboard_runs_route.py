from __future__ import annotations

from fastapi.testclient import TestClient

from modal_training_gym import _dashboard
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.step_timing import (
    advance_synchronous_timing_watermark,
    aggregated_training_step_timing_key,
    build_aggregated_training_step_timing,
)
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _save_records() -> None:
    TrainingRun(
        training_run_id="run-route-1",
        modal_app_id="ap-route",
        framework=Framework.SLIME,
        config={
            "model": {"model_name": "Qwen/Qwen3-4B"},
            "dataset": {"hf_repo": "openai/gsm8k"},
            "recipe": {"gpu_type": "H100"},
        },
        created_at=100,
        started_at=100,
        updated_at=150,
        metadata={"group_id": "route-group"},
        substep_times={
            "1": {
                "train_model": {
                    "start": 120.0,
                    "duration_s": 2.0,
                    "intervals": [
                        {
                            "step_id": 0,
                            "start": 120.0,
                            "duration_s": 2.0,
                            "training_role": "actor",
                        }
                    ],
                }
            }
        },
    ).save()
    TrainResult(
        app_name="route-app",
        framework=Framework.SLIME,
        training_run_id="run-route-1",
        checkpoint_dir="/checkpoints/run-route-1",
    ).save()


def test_runs_route_returns_typed_joined_summaries(fake_volume, monkeypatch, tmp_path):
    _save_records()
    stored_runs = metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert stored_runs is not None
    assert (
        stored_runs[0]["substep_times"]["1"]["train_model"]["intervals"][0][
            "training_role"
        ]
        == "actor"
    )

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert len(response.json()) == 1
    summary = response.json()[0]
    assert summary["training_run_id"] == "run-route-1"
    assert summary["run_id"] == "run-route-1"
    assert summary["status"] == "running"
    assert summary["model"] == "Qwen/Qwen3-4B"
    assert summary["dataset"] == "openai/gsm8k"
    assert summary["recipe"] == "slime"
    assert summary["group_id"] == "route-group"
    assert summary["has_train_result"] is True
    assert summary["train_result"]["checkpoint_dir"] == ("/checkpoints/run-route-1")
    assert "step_times" not in summary
    assert "substep_times" not in summary


def test_framework_status_updates_canonical_run_without_rewriting_summary(
    fake_volume, monkeypatch, tmp_path
):
    run_id = "run-live-status"
    TrainingRun(
        training_run_id=run_id,
        framework=Framework.SLIME,
        config={},
        metadata={"attempt_count": 1},
    ).save()
    metadata.vol_put(
        MetadataStore.FRAMEWORK_STATUS_TOKENS,
        run_id,
        {"token": "test-token"},
    )
    summary_before = metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)

    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/framework-status",
            headers={"Authorization": "Bearer test-token"},
            json={
                "training_run_id": run_id,
                "training_attempt": 1,
                "phase": "training",
                "progress_current": 2,
                "progress_total": 4,
                "progress_unit": "step",
            },
        )
        runs_response = client.get("/api/runs")

    assert response.status_code == 200
    assert (
        metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
        == summary_before
    )

    canonical_run = TrainingRun.from_id(run_id)
    assert canonical_run.framework_status.value == "training"
    assert canonical_run.metadata["framework_progress"]["current"] == 2

    assert runs_response.status_code == 200
    summary = runs_response.json()[0]
    assert summary["framework_status"] == "training"
    assert summary["framework_progress"]["current"] == 2
    assert summary["framework_progress"]["total"] == 4
    assert "step_times" not in summary
    assert "substep_times" not in summary


def test_runs_route_keeps_runs_when_train_result_store_fails(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    original = metadata.vol_get_summary_items_healed

    def fail_results(store):
        if store is MetadataStore.TRAIN_RESULTS_SUMMARY:
            raise RuntimeError("result store unavailable")
        return original(store)

    monkeypatch.setattr(metadata, "vol_get_summary_items_healed", fail_results)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["has_train_result"] is False


def test_runs_route_falls_back_when_live_canonical_read_fails(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()

    def fail_canonical_read(_cls, _run_id, *, is_async=False):
        raise RuntimeError("canonical store unavailable")

    monkeypatch.setattr(TrainingRun, "from_id", classmethod(fail_canonical_read))

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["training_run_id"] for run in response.json()] == ["run-route-1"]


def test_runs_route_isolates_invalid_run_records(fake_volume, monkeypatch, tmp_path):
    _save_records()
    runs = metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert runs is not None
    invalid_run = {
        **runs[0],
        "training_run_id": "invalid-run",
        "step_times": {"1": {"phase": {"not": "an integer"}}},
    }
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_RUNS_SUMMARY, [invalid_run, runs[0]]
    )

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["training_run_id"] for run in response.json()] == ["run-route-1"]


def test_step_timing_deltas_filter_steps_and_fall_back_to_canonical(
    fake_volume, monkeypatch, tmp_path
):
    run_id = "run-timing-delta"
    step_times = {
        str(step): {
            "start": step * 10,
            "end": step * 10 + step,
            "duration_s": step,
            "full_step_duration_s": step + 0.5,
        }
        for step in (1, 2)
    }
    substep_times = {
        str(step): {
            "train_model": {
                "start": step * 10.0,
                "duration_s": float(step),
                "intervals": [],
            }
        }
        for step in (1, 2)
    }
    run = TrainingRun(
        training_run_id=run_id,
        framework=Framework.SLIME,
        config={},
        metadata={"attempt_count": 1},
    )
    advance_synchronous_timing_watermark(
        run,
        1,
        completed_through_step=2,
    )
    run.save()

    timing_store = {
        aggregated_training_step_timing_key(run_id, 1, step): (
            build_aggregated_training_step_timing(
                run_id,
                1,
                step,
                step_times[str(step)],
                substep_times[str(step)],
                source="live",
            )
        )
        for step in (1, 2)
    }
    monkeypatch.setattr(_dashboard, "_step_times_dict", lambda: timing_store)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get(
            f"/api/runs/{run_id}/step-timings",
            params={"training_attempt": 1, "after_step": 1},
        )

        assert response.status_code == 200
        assert response.json() == {
            "training_attempt": 1,
            "persisted_through_step": 2,
            "step_times": {"2": step_times["2"]},
            "substep_times": {"2": substep_times["2"]},
        }

        step_one_key = aggregated_training_step_timing_key(run_id, 1, 1)
        step_one_timing = timing_store.pop(step_one_key)
        response = client.get(
            f"/api/runs/{run_id}/step-timings",
            params={"training_attempt": 1, "after_step": 0},
        )
        assert response.status_code == 200
        assert response.json() == {
            "training_attempt": 1,
            "persisted_through_step": 0,
            "step_times": {},
            "substep_times": {},
        }

        resumed_run = TrainingRun.from_id(run_id)
        resumed_run.metadata = {
            **(resumed_run.metadata or {}),
            "resumed_from_checkpoint": True,
            "resume_from_iteration": 0,
        }
        resumed_run.save()
        response = client.get(
            f"/api/runs/{run_id}/step-timings",
            params={"training_attempt": 1, "after_step": 0},
        )
        assert response.status_code == 200
        assert response.json() == {
            "training_attempt": 1,
            "persisted_through_step": 2,
            "step_times": {"2": step_times["2"]},
            "substep_times": {"2": substep_times["2"]},
        }

        timing_store[step_one_key] = step_one_timing

        timing_store.clear()
        canonical_run = TrainingRun.from_id(run_id)
        canonical_run.metadata = {
            key: value
            for key, value in (canonical_run.metadata or {}).items()
            if key not in {"resumed_from_checkpoint", "resume_from_iteration"}
        }
        canonical_run.step_times = step_times
        canonical_run.substep_times = substep_times
        canonical_run.save()

        response = client.get(
            f"/api/runs/{run_id}/step-timings",
            params={"training_attempt": 1, "after_step": 1},
        )

        assert response.status_code == 200
        assert response.json() == {
            "training_attempt": 1,
            "persisted_through_step": 2,
            "step_times": {"2": step_times["2"]},
            "substep_times": {"2": substep_times["2"]},
        }

        canonical_run.metadata = {
            **(canonical_run.metadata or {}),
            "attempt_count": 2,
        }
        canonical_run.save()

        stale_response = client.get(
            f"/api/runs/{run_id}/step-timings",
            params={"training_attempt": 1, "after_step": 2},
        )
        assert stale_response.status_code == 200
        assert stale_response.json() == {
            "training_attempt": 2,
            "persisted_through_step": 0,
            "step_times": {},
            "substep_times": {},
        }

        future_response = client.get(
            f"/api/runs/{run_id}/step-timings",
            params={"training_attempt": 3, "after_step": 0},
        )
        assert future_response.status_code == 503
