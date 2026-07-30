"""Disaggregated Moonlight-16B-A3B GRPO on DAPO-math via stitch.

The larger rung of the stitch examples: same one-call shape as
``qwen3_4b_gsm8k``, but a 2×8×H200 multi-node trainer with expert parallelism
publishing MoE deltas to the bulletin board, and rollouts served by a Flash pool
of single-H200 SGLang replicas (multi-latent attention keeps the KV cache small
enough for one GPU at 16B). Mirrors stitch's ``moonlight`` cookbook config.

Run from the repo root::

    uv run python -m modal_training_gym.frameworks.stitch.examples.moonlight_16b_dapo_math
"""

from __future__ import annotations

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models.moonlight_16b import Moonlight_16B_A3B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.stitch_recipe import (
    Moonlight_16B_A3B_Stitch_Recipe,
)


class DAPOMath(HuggingFaceDataset):
    """DAPO-math-17k pre-formatted for slime's math reward."""

    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = "prompt"
    output_column = "label"
    output_format = "jsonl"
    apply_chat_template = True


training_run = TrainConfig(
    model=Moonlight_16B_A3B(),
    dataset=DAPOMath(),
    recipe=Moonlight_16B_A3B_Stitch_Recipe(
        wandb=WandbConfig(project="training-gym", group="stitch-moonlight-16b-dapo"),
    ),
)


if __name__ == "__main__":
    result = training_run.train()
    print(f"Checkpoints: {result.checkpoint_dir}")
