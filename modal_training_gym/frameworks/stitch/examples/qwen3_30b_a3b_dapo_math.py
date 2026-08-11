"""Disaggregated Qwen3-30B-A3B GRPO on DAPO-math via stitch.

The BF16 port of stitch's ``miles_disagg/configs/qwen3_30b_a3b_nvfp4_46``
cookbook config, on the slime path this launcher wires: a 1×8×H200 TP4/EP8
trainer publishing sparse MoE deltas to the bulletin board, rollouts served by a
Flash pool of single-H200 SGLang replicas that apply those deltas in place. Same
one-call shape as the other stitch examples.

NVFP4 — the subject of the upstream config — is not ported: it needs the miles
trainer plus a separately quantized served base, and ``build_stitch_app``
launches the slime trainer only.

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
    """DAPO-math-17k pre-formatted for slime's math reward."""

    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = "prompt"
    output_column = "label"
    output_format = "jsonl"
    apply_chat_template = True


training_run = TrainConfig(
    model=Qwen3_30B(),
    dataset=DAPOMath(),
    recipe=Qwen3_30B_A3B_Stitch_Recipe(
        wandb=WandbConfig(project="training-gym", group="stitch-qwen3-30b-a3b-dapo"),
    ),
)


if __name__ == "__main__":
    result = training_run.train()
    print(f"Checkpoints: {result.checkpoint_dir}")
