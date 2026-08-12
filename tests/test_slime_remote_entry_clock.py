from __future__ import annotations

import pytest

from modal_training_gym.frameworks.slime import launcher


def test_operator_pool_remote_entry_clock_is_exact_and_receipt_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.time, "time_ns", lambda: 123)
    digest = "a" * 64
    clock = launcher._capture_remote_entry_clock(
        {"DRIFT_ASYNC_RL_OPERATOR_POOL_RECEIPT_SHA256": digest}
    )
    assert clock == {
        "DRIFT_ASYNC_RL_REMOTE_ENTRY_EPOCH_NS": "123",
        "DRIFT_ASYNC_RL_REMOTE_ENTRY_RECEIPT_SHA256": digest,
    }


def test_remote_entry_clock_is_propagated_and_cannot_be_overwritten() -> None:
    clock = {
        "DRIFT_ASYNC_RL_REMOTE_ENTRY_EPOCH_NS": "123",
        "DRIFT_ASYNC_RL_REMOTE_ENTRY_RECEIPT_SHA256": "a" * 64,
    }
    result = launcher._remote_entry_runtime_env(
        {
            "USER_SETTING": "retained",
            "DRIFT_ASYNC_RL_REMOTE_ENTRY_EPOCH_NS": "forged",
        },
        clock,
    )
    assert result == {"USER_SETTING": "retained", **clock}
    assert launcher._capture_remote_entry_clock({}) == {}
    with pytest.raises(ValueError, match="malformed"):
        launcher._capture_remote_entry_clock(
            {"DRIFT_ASYNC_RL_OPERATOR_POOL_RECEIPT_SHA256": "not-a-sha"}
        )


def test_remote_execution_identity_binds_app_and_function_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "current_function_call_id", lambda: "fc-call")
    assert launcher._remote_execution_identity("ap-app") == {
        "TRAINING_GYM_MODAL_APP_ID": "ap-app",
        "TRAINING_GYM_FUNCTION_CALL_ID": "fc-call",
    }
    monkeypatch.setattr(launcher, "current_function_call_id", lambda: "")
    with pytest.raises(RuntimeError, match="identity is unavailable"):
        launcher._remote_execution_identity("ap-app")
