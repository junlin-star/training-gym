"""Disaggregated Qwen3-4B GRPO on GSM8K via stitch.

Runnable example for :class:`StitchRecipe` — the disaggregated counterpart of the
colocated slime GSM8K tutorial, with the same one-call shape as the other
recipes. A Modal Flash pool of SGLang servers comes up with the app and serves
rollouts, self-syncing to the sparse weight deltas the trainer publishes to a
Volume bulletin board.

Run from the repo root::

    uv run python -m modal_training_gym.frameworks.stitch.examples.qwen3_4b_gsm8k
"""

from __future__ import annotations

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models.qwen3_4b import Qwen3_4B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.stitch_recipe import Qwen3_4b_Stitch_Recipe


class GSM8K(HuggingFaceDataset):
    """GSM8K pre-formatted for slime's math reward (chat ``messages`` + gold ``label``)."""

    hf_repo = "zhuzilin/gsm8k"
    hf_split = "train"
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True


training_run = TrainConfig(
    model=Qwen3_4B(),
    dataset=GSM8K(),
    recipe=Qwen3_4b_Stitch_Recipe(
        wandb=WandbConfig(project="training-gym", group="stitch-qwen3-4b-gsm8k"),
    ),
)


if __name__ == "__main__":
    result = training_run.train()
    print(f"Checkpoints: {result.checkpoint_dir}")
