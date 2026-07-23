from __future__ import annotations

from fastapi.testclient import TestClient

from modal_training_gym import _dashboard
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun
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
    assert (
        summary["substep_times"]["1"]["train_model"]["intervals"][0]["training_role"]
        == "actor"
    )


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
