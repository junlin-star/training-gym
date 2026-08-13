from __future__ import annotations

from modal_training_gym.common import step_timing
from modal_training_gym.common.step_timing import RoleTimingRecord


def _phase(start: float, end: float, *, invocations=None) -> dict:
    return {
        "count": 1,
        "busy_duration_s": end - start,
        "longest_duration_s": end - start,
        "first_start_s": start,
        "last_end_s": end,
        "invocations": invocations or [[start, end]],
    }


def test_measured_run_times_excludes_not_in_step_phases(monkeypatch):
    monkeypatch.setattr(
        step_timing,
        "load_run",
        lambda _run_id: {
            0: [
                {
                    "role": "driver",
                    "lane_start_unix_s": 100.0,
                    "phases": {
                        "train_models": _phase(10.1234, 20.0),
                        "evaluate_rollouts": _phase(0.0, 5.0),
                        "checkpoint_save": _phase(20.0, 23.0),
                    },
                }
            ]
        },
    )

    step_times, _substep_times = step_timing.measured_run_times("run")

    assert step_times == {"1": {"duration_s": 9.877}}


def test_role_timing_record_ignores_legacy_created_at():
    record = RoleTimingRecord.model_validate(
        {
            "training_run_id": "run",
            "role": "driver",
            "created_at": 123,
        }
    )

    assert "created_at" not in record.model_dump()


def test_phase_timing_accepts_legacy_total_duration_key():
    record = RoleTimingRecord.model_validate(
        {
            "training_run_id": "run",
            "role": "driver",
            "phases": {
                "train_models": {
                    "count": 1,
                    "total_duration_s": 2.0,
                    "longest_duration_s": 2.0,
                    "first_start_s": 0.0,
                    "last_end_s": 2.0,
                }
            },
        }
    )

    phase = record.phases["train_models"]
    assert phase.busy_duration_s == 2.0
    assert "total_duration_s" not in phase.model_dump()
