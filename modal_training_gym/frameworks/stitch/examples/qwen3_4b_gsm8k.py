"""Disaggregated Qwen3-4B GRPO on GSM8K via stitch (deploy-based).

Runnable example for :class:`StitchRecipe` — the disaggregated counterpart of the
colocated slime GSM8K tutorial. A persistent Modal Flash pool serves rollouts
and self-syncs to sparse weight deltas the trainer publishes to a Volume
bulletin board.

Run from the repo root::

    # one-time: cache the base model + materialize the dataset
    uv run modal run -m modal_training_gym.frameworks.stitch.examples.qwen3_4b_gsm8k::download_model
    uv run modal run -m modal_training_gym.frameworks.stitch.examples.qwen3_4b_gsm8k::prepare_dataset

    # bring up the persistent trainer + Flash rollout pool
    uv run modal deploy -m modal_training_gym.frameworks.stitch.examples.qwen3_4b_gsm8k

    # (optional) confirm the pool serves at the base weight version
    uv run modal run -m modal_training_gym.frameworks.stitch.examples.qwen3_4b_gsm8k::smoke_flash_pool

    # spawn a training run on the deployed app
    uv run modal run -m modal_training_gym.frameworks.stitch.examples.qwen3_4b_gsm8k::launch_train
"""

from __future__ import annotations

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models.qwen3_4b import Qwen3_4B
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.frameworks.stitch import build_stitch_app
from modal_training_gym.frameworks.stitch.launcher import (
    smoke_flash_pool as _smoke_flash_pool,
)
from modal_training_gym.frameworks.stitch.launcher import spawn_training_run
from modal_training_gym.train_recipes.stitch_recipe import Qwen3_4b_Stitch_Recipe


class GSM8K(HuggingFaceDataset):
    """GSM8K pre-formatted for slime's math reward (chat ``messages`` + gold ``label``)."""

    hf_repo = "zhuzilin/gsm8k"
    hf_split = "train"
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True


model = Qwen3_4B()
dataset = GSM8K()
recipe = Qwen3_4b_Stitch_Recipe(
    wandb=WandbConfig(project="training-gym", group="stitch-qwen3-4b-gsm8k"),
)

app = build_stitch_app(model=model, dataset=dataset, recipe=recipe)


@app.local_entrypoint()
def launch_train() -> None:
    """Spawn a training run on the deployed app."""
    spawn_training_run(app_name=app.name, recipe=recipe, model=model, dataset=dataset)


@app.local_entrypoint()
def smoke_flash_pool(weight_version: int = 0, timeout_seconds: int = 1800) -> None:
    """Confirm the deployed Flash pool serves at ``weight_version``."""
    _smoke_flash_pool(
        app_name=app.name,
        model=model,
        recipe=recipe,
        weight_version=weight_version,
        timeout_seconds=timeout_seconds,
    )
