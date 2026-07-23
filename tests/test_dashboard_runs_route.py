from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from modal_training_gym import _dashboard
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore


class _SnapshotKeys:
    def __init__(self, values: dict[tuple[str, str, int, int], object]) -> None:
        self.values = values

    async def aio(self):
        for key in self.values:
            yield key


class _SnapshotGet:
    def __init__(self, values: dict[tuple[str, str, int, int], object]) -> None:
        self.values = values

    async def aio(self, key: tuple[str, str, int, int]) -> object:
        return self.values[key]


class _SnapshotPut:
    def __init__(
        self,
        values: dict[tuple[str, str, int, int], object],
        on_put: Callable[[], None] | None,
    ) -> None:
        self.values = values
        self.on_put = on_put

    async def aio(self, key: tuple[str, str, int, int], value: object) -> None:
        self.values[key] = value
        if self.on_put is not None:
            self.on_put()


class _SnapshotPop:
    def __init__(self, values: dict[tuple[str, str, int, int], object]) -> None:
        self.values = values

    async def aio(
        self, key: tuple[str, str, int, int], default: object = None
    ) -> object:
        return self.values.pop(key, default)


class _SnapshotStore:
    def __init__(
        self,
        values: dict[tuple[str, str, int, int], object],
        *,
        on_put: Callable[[], None] | None = None,
    ) -> None:
        self.keys = _SnapshotKeys(values)
        self.get = _SnapshotGet(values)
        self.put = _SnapshotPut(values, on_put)
        self.pop = _SnapshotPop(values)
        self.values = values


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _save_records(*, attempt_count: int = 1) -> None:
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
        metadata={"group_id": "route-group", "attempt_count": attempt_count},
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


def test_runs_route_overlays_live_async_timing_intervals(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    snapshot = {
        "displayed_training_attempt": 1,
        "archived_timing_attempts": {},
        "timing_event_counts": {0: 2},
        "step_times": {
            "1": {"start": 100, "end": 110, "duration_s": 10},
        },
        "substep_times": {
            "1": {
                "forward_backward": {
                    "start": 103.0,
                    "duration_s": 4.0,
                    "intervals": [
                        {
                            "start": 103.0,
                            "duration_s": 4.0,
                            "step_id": 0,
                            "training_role": "actor",
                            "timeline_lane": "training",
                            "parent_phase": "training",
                            "display_name": "Forward / backward",
                        }
                    ],
                }
            }
        },
    }
    snapshot_store = _SnapshotStore(
        {("run-route-1", "timing_snapshot", 1, 0): snapshot}
    )
    monkeypatch.setattr(
        metadata,
        "_step_timing_snapshots_dict",
        lambda: snapshot_store,
    )

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    summary = response.json()[0]
    assert summary["train_result"]["checkpoint_dir"] == "/checkpoints/run-route-1"
    assert summary["substep_times"]["1"]["forward_backward"]["intervals"] == [
        {
            "start": 103.0,
            "duration_s": 4.0,
            "step_id": 0,
            "training_role": "actor",
            "timeline_lane": "training",
            "parent_phase": "training",
            "display_name": "Forward / backward",
        }
    ]


def test_async_timing_notification_ignores_an_older_attempt(
    fake_volume, monkeypatch, tmp_path
):
    _save_records(attempt_count=2)
    metadata.vol_put(
        MetadataStore.FRAMEWORK_STATUS_TOKENS,
        "run-route-1",
        {"token": "timing-token"},
    )

    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/async-step-times",
            json={
                "training_run_id": "run-route-1",
                "training_attempt": 1,
                "completed_rollout_id": 0,
            },
            headers={"Authorization": "Bearer timing-token"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ignored", "reason": "attempt_changed"}


def test_async_timing_notification_discards_snapshot_if_attempt_changes(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    metadata.vol_put(
        MetadataStore.FRAMEWORK_STATUS_TOKENS,
        "run-route-1",
        {"token": "timing-token"},
    )

    def start_next_attempt() -> None:
        run = TrainingRun.from_id("run-route-1")
        assert isinstance(run, TrainingRun)
        run.metadata = {**(run.metadata or {}), "attempt_count": 2}
        run.save()

    snapshot_store = _SnapshotStore({}, on_put=start_next_attempt)
    monkeypatch.setattr(
        metadata,
        "_step_timing_snapshots_dict",
        lambda: snapshot_store,
    )
    monkeypatch.setattr(metadata, "_step_times_dict", lambda: _SnapshotStore({}))

    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/async-step-times",
            json={
                "training_run_id": "run-route-1",
                "training_attempt": 1,
                "completed_rollout_id": 0,
            },
            headers={"Authorization": "Bearer timing-token"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ignored", "reason": "attempt_changed"}
    assert snapshot_store.values == {}


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
