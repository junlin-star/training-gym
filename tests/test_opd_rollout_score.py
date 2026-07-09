"""OPD / cross-tokenizer rewards must not render as 0.0 on the dashboard.

``custom_rm`` returns the teacher ``/generate`` JSON as ``sample.reward`` until
post-process. Gym's rollout reporter snapshots samples in that window, so hooks
must stash ``metadata["shaped_reward"] = float(...)`` and score extraction
reads that instead of treating the dict as ``0.0``.
"""

from types import SimpleNamespace

from modal_training_gym.frameworks.slime.sample_extraction import (
    _sample_score,
    _sample_to_dict,
)


def test_sample_score_prefers_numeric_reward():
    sample = SimpleNamespace(reward=0.91, metadata={"shaped_reward": 0.1})
    assert _sample_score(sample) == 0.91


def test_sample_score_falls_back_to_shaped_reward_when_reward_is_opd_dict():
    sample = SimpleNamespace(
        reward={"text": "", "meta_info": {"input_token_logprobs": []}},
        metadata={"shaped_reward": 0.825, "task_passed": True},
    )
    assert _sample_score(sample) == 0.825


def test_sample_score_returns_zero_for_dict_without_shaped_reward():
    sample = SimpleNamespace(
        reward={"meta_info": {"input_token_logprobs": []}},
        metadata={},
    )
    assert _sample_score(sample) == 0.0
    assert _sample_score(SimpleNamespace(reward=None, metadata=None)) == 0.0


def test_sample_to_dict_uses_shaped_reward_for_opd():
    sample = SimpleNamespace(
        prompt="p",
        response="r",
        reward={"meta_info": {"input_token_logprobs": []}},
        metadata={"shaped_reward": 0.71, "task_passed": False, "response_length": 12},
        response_length=12,
    )
    out = _sample_to_dict(sample)
    assert out["score"] == 0.71
    assert out["metadata"]["shaped_reward"] == 0.71
    assert out["metadata"]["task_passed"] is False
