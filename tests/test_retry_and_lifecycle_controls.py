from __future__ import annotations

import pytest
from modal_proto import api_pb2

from modal_training_gym.common import modal_lifecycle
from modal_training_gym.frameworks.slime.launcher import (
    _modal_retry_policy,
    _validate_attempt_limit,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


def _recipe(**overrides):
    return SlimeRecipe(
        gpu_type="H200",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=4,
        rollout_max_response_len=256,
        rollout_temperature=1.0,
        save_interval=1,
        **overrides,
    )


def test_zero_retries_uses_no_modal_retry_policy() -> None:
    assert _modal_retry_policy(0) is None
    policy = _modal_retry_policy(2)
    assert policy is not None
    assert policy.max_retries == 2
    with pytest.raises(ValueError, match="nonnegative"):
        _modal_retry_policy(-1)


def test_logical_attempt_limit_fails_before_ray_boundary() -> None:
    _validate_attempt_limit(1, 1)
    _validate_attempt_limit(100, None)
    with pytest.raises(RuntimeError, match="before Ray bootstrap"):
        _validate_attempt_limit(2, 1)


def test_recipe_validates_positive_max_attempts() -> None:
    assert _recipe(max_retries=0, max_attempts=1).max_attempts == 1
    with pytest.raises(ValueError, match="max_attempts must be positive"):
        _recipe(max_attempts=0)


def test_best_effort_stop_wraps_strict_stop(monkeypatch, capsys) -> None:
    def fail(_app_id: str) -> None:
        raise ConnectionError("control plane unavailable")

    monkeypatch.setattr(modal_lifecycle, "request_stop_app", fail)
    modal_lifecycle.stop_app("ap-exact")
    assert "could not auto-stop app ap-exact" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (api_pb2.APP_STATE_DETACHED, True),
        (api_pb2.APP_STATE_STOPPING, True),
        (api_pb2.APP_STATE_STOPPED, False),
        (api_pb2.APP_STATE_DISABLED, False),
        (api_pb2.APP_STATE_UNSPECIFIED, None),
        (10_000, None),
    ],
)
def test_live_status_requires_an_explicit_live_or_terminal_state(
    monkeypatch,
    state: int,
    expected: bool | None,
) -> None:
    monkeypatch.setattr(
        modal_lifecycle, "get_app_lifecycle_state", lambda _app_id: state
    )
    assert modal_lifecycle.app_live_status("ap-exact") is expected


@pytest.mark.parametrize("app_id", ["", "app-123", "../ap-123", "ap-123/child"])
def test_strict_stop_rejects_non_exact_app_id(app_id: str) -> None:
    with pytest.raises(ValueError, match="exact Modal app ID"):
        modal_lifecycle.request_stop_app(app_id)
