"""OPD / cross-tokenizer rewards must not render as 0.0 on the dashboard.

``custom_rm`` returns the teacher ``/generate`` JSON as ``sample.reward`` until
post-process. Gym's rollout reporter snapshots samples in that window, so hooks
must stash ``metadata["shaped_reward"] = float(...)`` and score extraction
maps that onto gym :class:`~modal_training_gym.common.sample.Sample.score`.
"""

from modal_training_gym.common.sample import Sample
from modal_training_gym.frameworks.slime.sample_extraction import (
    _sample_score,
    _sample_to_dict,
)

_OPD_TEACHER_PAYLOAD = {"text": "", "meta_info": {"input_token_logprobs": []}}


def test_sample_score_prefers_numeric_reward():
    sample = Sample(metadata={"shaped_reward": 0.1})
    assert _sample_score(sample, reward=0.91) == 0.91


def test_sample_score_falls_back_to_shaped_reward():
    sample = Sample(metadata={"shaped_reward": 0.825, "task_passed": True})
    assert _sample_score(sample) == 0.825


def test_sample_score_returns_zero_without_shaped_reward():
    assert _sample_score(Sample()) == 0.0


def test_sample_to_dict_maps_shaped_reward_onto_gym_sample():
    raw = {
        "prompt": "p",
        "response": "r",
        "reward": _OPD_TEACHER_PAYLOAD,
        "metadata": {
            "shaped_reward": 0.71,
            "task_passed": False,
            "response_length": 12,
        },
        "response_length": 12,
    }
    sample = Sample.model_validate(_sample_to_dict(raw))
    assert sample.score == 0.71
    assert sample.prompt == "p"
    assert sample.response == "r"
    assert sample.metadata["shaped_reward"] == 0.71
    assert sample.metadata["task_passed"] is False
    assert sample.metadata["response_length"] == 12
