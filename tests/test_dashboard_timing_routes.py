"""Timing read routes: measured records, missing rollouts, legacy runs, cache."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modal_training_gym import _dashboard
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common import step_timing
from modal_training_gym.common.step_timing import PROTOCOL, RoleTimingRecord
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore

RUN_ID = "run-timing-1"
TOKEN = "test-status-token-run-timing-1"


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _save_run(substep_times: dict | None = None) -> None:
    TrainingRun(
        training_run_id=RUN_ID,
        modal_app_id="ap-timing",
        framework=Framework.SLIME,
        config={"model": {"model_name": "Qwen/Qwen3-4B"}},
        created_at=100,
        started_at=100,
        updated_at=150,
        substep_times=substep_times,
    ).save()
    metadata.vol_put(MetadataStore.FRAMEWORK_STATUS_TOKENS, RUN_ID, {"token": TOKEN})


def _save_record(rollout_id: int, role: str, total: float) -> None:
    RoleTimingRecord(
        training_run_id=RUN_ID,
        rollout_id=rollout_id,
        role=role,
        lane_start_unix_s=1000.0,
        phases={
            "train_models": {
                "count": 2,
                "total_duration_s": total,
                "longest_duration_s": total / 2,
                "first_start_s": 0.5,
                "last_end_s": 0.5 + total,
                "invocations": [(0.5, 0.5 + total / 2), (0.5 + total / 2, 0.5 + total)],
            }
        },
    ).save()


def test_capability_route_identifies_the_protocol(fake_volume, monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/timing-events")
    assert response.json() == {"protocol": PROTOCOL}


def test_batch_returns_a_lane_per_role_and_leaves_unmeasured_rollouts_empty(
    fake_volume, monkeypatch, tmp_path
):
    """A rollout still in flight must read as absent, not as another's timing."""
    _save_run()
    _save_record(0, "driver", 4.0)
    _save_record(0, "actor", 2.0)
    _save_record(1, "driver", 3.0)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get(f"/api/runs/{RUN_ID}/timings?rollout_ids=0,1,2")

    timings = response.json()
    assert set(timings) == {"0", "1", "2"}
    assert set(timings["0"]["roles"]) == {"driver", "actor"}
    assert timings["0"]["roles"]["driver"]["phases"]["train_models"] == {
        "count": 2,
        "total_duration_s": 4.0,
        "longest_duration_s": 2.0,
        "first_start_s": 0.5,
        "last_end_s": 4.5,
        "invocations": [[0.5, 2.5], [2.5, 4.5]],
    }
    assert timings["2"] == {"roles": {}}


def test_a_measured_run_is_never_backfilled_from_legacy_timing(
    fake_volume, monkeypatch, tmp_path
):
    _save_run(
        substep_times={"2": {"generate_rollouts": {"start": 10.0, "duration_s": 1.0}}}
    )
    _save_record(0, "driver", 4.0)

    with _client(monkeypatch, tmp_path) as client:
        timings = client.get(f"/api/runs/{RUN_ID}/timings?rollout_ids=0,1").json()

    assert timings["1"] == {"roles": {}}


def test_a_pre_cutover_run_renders_from_its_legacy_blob(
    fake_volume, monkeypatch, tmp_path
):
    # Legacy steps are one-based; rollout rows are zero-based.
    _save_run(
        substep_times={"1": {"generate_rollouts": {"start": 10.0, "duration_s": 1.5}}}
    )

    with _client(monkeypatch, tmp_path) as client:
        timings = client.get(f"/api/runs/{RUN_ID}/timings?rollout_ids=0").json()

    phases = timings["0"]["roles"]["driver"]["phases"]
    assert phases["generate_rollouts"]["total_duration_s"] == 1.5
    assert phases["generate_rollouts"]["count"] == 1


def test_posting_a_record_invalidates_the_cached_rollout(
    fake_volume, monkeypatch, tmp_path
):
    """The lane grows while a rollout runs, so a cached empty must not stick."""
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        assert client.get(f"/api/runs/{RUN_ID}/timings/0").json() == {"roles": {}}

        record = RoleTimingRecord(
            training_run_id=RUN_ID,
            rollout_id=0,
            role="driver",
            lane_start_unix_s=1000.0,
            phases={
                "train_models": {
                    "count": 1,
                    "total_duration_s": 1.0,
                    "longest_duration_s": 1.0,
                    "first_start_s": 0.0,
                    "last_end_s": 1.0,
                }
            },
        )
        posted = client.post(
            "/api/timing-events",
            json=record.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert posted.status_code == 200

        timings = client.get(f"/api/runs/{RUN_ID}/timings/0").json()

    assert timings["roles"]["driver"]["phases"]["train_models"]["count"] == 1


@pytest.mark.parametrize(
    "query, expected",
    [("", {}), ("rollout_ids=", {}), ("rollout_ids=abc", {})],
)
def test_a_request_for_no_rollouts_reads_nothing(
    fake_volume, monkeypatch, tmp_path, query, expected
):
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        assert client.get(f"/api/runs/{RUN_ID}/timings?{query}").json() == expected


def test_a_batch_lists_the_volume_once(fake_volume, monkeypatch, tmp_path):
    """Volume listing is rate limited, so a page of rollouts is one listing."""
    _save_run()
    for rollout_id in range(5):
        _save_record(rollout_id, "driver", 1.0)

    listings = []
    original = step_timing.vol_list_prefix
    monkeypatch.setattr(
        step_timing,
        "vol_list_prefix",
        lambda store, prefix: listings.append(prefix) or original(store, prefix),
    )

    with _client(monkeypatch, tmp_path) as client:
        timings = client.get(f"/api/runs/{RUN_ID}/timings?rollout_ids=0,1,2,3,4").json()

    assert len(listings) == 1
    assert all(timings[str(i)]["roles"]["driver"] for i in range(5))


def test_an_oversized_batch_is_refused(fake_volume, monkeypatch, tmp_path):
    _save_run()
    ids = ",".join(str(i) for i in range(201))  # one past TIMING_MAX_BATCH
    with _client(monkeypatch, tmp_path) as client:
        response = client.get(f"/api/runs/{RUN_ID}/timings?rollout_ids={ids}")
    assert response.status_code == 400
