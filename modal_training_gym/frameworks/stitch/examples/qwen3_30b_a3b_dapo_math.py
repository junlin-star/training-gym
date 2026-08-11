"""Disaggregated Qwen3-30B-A3B NVFP4 GRPO on DAPO-math via stitch.

The port of stitch's ``miles_disagg/configs/qwen3_30b_a3b_nvfp4_46`` cookbook
config: a 1×8 B200 miles trainer doing NVFP4 QAT on the routed experts and
publishing sparse deltas to the bulletin board, with rollouts served by a Flash
pool of single-B200 SGLang replicas that apply those deltas in place.

One call brings up everything, including the served baseline: ``train()`` runs the
model download, the dataset prep, and ``prepare_checkpoints`` — which materializes
the trainer's BF16 masters and converts the pool's NVFP4 baseline with the same
quantizer the trainer exports with — before starting the run.

Run from the repo root::

    uv run python -m modal_training_gym.frameworks.stitch.examples.qwen3_30b_a3b_dapo_math
"""

from __future__ import annotations

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models.qwen3_30b import Qwen3_30B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.stitch_recipe import (
    Qwen3_30B_A3B_Stitch_Recipe,
)


class DAPOMath(HuggingFaceDataset):
    """DAPO-math-17k, pre-formatted for the deepscaler math reward."""

    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = "prompt"
    output_column = "label"
    output_format = "jsonl"
    apply_chat_template = True


training_run = TrainConfig(
    model=Qwen3_30B(),
    dataset=DAPOMath(),
    recipe=Qwen3_30B_A3B_Stitch_Recipe(
        wandb=WandbConfig(project="training-gym", group="stitch-qwen3-30b-a3b-nvfp4"),
    ),
)


if __name__ == "__main__":
    result = training_run.train()
    print(f"Checkpoints: {result.checkpoint_dir}")
