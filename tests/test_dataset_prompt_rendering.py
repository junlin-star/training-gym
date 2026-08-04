"""What `HuggingFaceDataset.prepare()` puts in the `messages` column.

A framework builds the rollout prompt by chat-templating that column, so
anything in it is something the model gets to read. The ground truth therefore
belongs in `label_key` and nowhere else — an assistant turn carrying the answer
turns an RL run into a copying exercise that scores near-perfectly and produces
no gradient (every GRPO group agrees), which is exactly how it hid.
"""

import datasets

from modal_training_gym.common.dataset import HuggingFaceDataset


class _QA(HuggingFaceDataset):
    hf_repo = "unused/qa"
    input_column = "question"
    output_column = "answer"
    label_key = "label"
    prompt_template = "{input}\n\nPut your final answer in \\boxed{{}}."


def _rows(**kwargs):
    ds = datasets.Dataset.from_list(
        [
            {
                "question": "Jack collects 2L of 20% salt water. How many ml of salt?",
                "answer": "400",
            }
        ]
    )
    return _QA(**kwargs)._format_for_training(ds)[0]


def test_answer_is_not_in_the_prompt():
    row = _rows()

    assert [m["role"] for m in row["messages"]] == ["user"]
    assert "400" not in row["messages"][0]["content"]
    # The reward still needs the ground truth; it just travels out of band.
    assert row["label"] == "400"


def test_no_assistant_turn_is_ever_emitted():
    for kwargs in ({}, {"system_prompt": "You are terse."}):
        row = _rows(**kwargs)

        assert all(m["role"] != "assistant" for m in row["messages"])


def test_system_prompt_still_leads():
    row = _rows(system_prompt="You are terse.")

    assert [m["role"] for m in row["messages"]] == ["system", "user"]
    assert row["messages"][0]["content"] == "You are terse."


def test_prompt_template_is_applied_to_the_user_turn():
    row = _rows()

    assert row["messages"][0]["content"].endswith("Put your final answer in \\boxed{}.")
