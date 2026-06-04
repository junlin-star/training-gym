"""A/B test for weight-sync performance: HfWeightIteratorDirect vs Bridge.

Root cause
==========
The Megatron → SGLang weight sync on Qwen3.6-35B-A3B (MoE, 256 experts)
takes >1 hr because the launcher blanked ``megatron_to_hf_mode`` to ``""``
for pre-conversion models, forcing the slow ``HfWeightIteratorDirect``
path.  That path serializes ~70 GB through 140 sequential 512 MB chunks
(CPU serialize → Gloo gather → Ray IPC → SGLang load per chunk ≈ 25 s
each ≈ 58 min total).

Fix
===
Keep ``megatron_to_hf_mode="bridge"`` (the recipe default) so slime uses
``HfWeightIteratorBridge`` / ``AutoBridge.export_hf_weights``, which
batches expert conversion and avoids the per-parameter EP all-gather +
chunk pipeline.

Experiment
==========
Two entry points that share identical config except for
``megatron_to_hf_mode``:

    # Baseline — current (slow) behavior
    uv run modal run scripts/ml/debug_weight_sync.py::train_baseline

    # Fix — bridge mode
    uv run modal run scripts/ml/debug_weight_sync.py::train_fixed

Both runs log ``[weight_sync] ... finished in Xs`` to Modal container
stdout (injected by ``patch_weight_sync_timing``).  wandb project
``weight-sync-debug`` groups runs by ``baseline`` / ``fixed`` for easy
comparison.

The dataset is a synthetic 10-row stub (no AWS creds needed).  Each run
does ``num_rollout=3`` training steps to get 3 weight-sync samples per
condition.
"""

import json
import os
import re

import modal

from modal_training_gym import (
    SlimeRecipe,
    TrainConfig,
)
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.models import Qwen3_6_35B

# ── Tiny synthetic dataset (no AWS creds needed) ────────────────────────────
SYSTEM_PROMPT = "You are a chemistry expert. Respond with a number 0-1."
USER_PROMPT = (
    "Is this reaction feasible?\n\nReaction SMILES: CC>>CC\n\nProbability (0-1):"
)

_PROB_RE = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")


class SyntheticRFMDataset(DatasetConfig):
    """10-row JSONL stub — just enough to trigger training steps + syncs."""

    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    output_format = "jsonl"
    no_think: bool = True

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._write_stub(path)
        if eval_paths:
            for eval_path in eval_paths.values():
                os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                self._write_stub(eval_path)

    def _write_stub(self, path: str) -> None:
        with open(path, "w") as f:
            for i in range(10):
                user_content = (
                    f"/no_think {USER_PROMPT}" if self.no_think else USER_PROMPT
                )
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "label": str(i % 2),
                }
                f.write(json.dumps(record) + "\n")


async def rfm_reward(args, sample, **kwargs) -> float:
    """GRPO reward: 1 - |pred - true|."""
    text = sample.response
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        return 0.0
    pred = 0.5
    for raw in reversed(_PROB_RE.findall(text)):
        try:
            val = float(raw)
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            pred = val
            break
    true = 1.0 if str(getattr(sample, "label", "")).strip() in ("1", "true") else 0.0
    return 1.0 - abs(pred - true)


# ── Config builder ──────────────────────────────────────────────────────────
def _build_config(*, variant: str) -> TrainConfig:
    """Build a minimal training config for the weight-sync experiment.

    ``variant`` is ``"baseline"`` (force HfWeightIteratorDirect via
    ``megatron_to_hf_mode=""``) or ``"fixed"`` (keep bridge mode).
    """
    model = Qwen3_6_35B()
    dataset = SyntheticRFMDataset()

    recipe = SlimeRecipe.get_base_recipe(model)
    if recipe is None:
        raise ValueError(f"No base slime recipe for {model.model_name}")

    recipe.custom_rm_function = rfm_reward
    recipe.wandb = None  # timing comes from stdout [weight_sync] logs
    recipe.image_overlay = lambda img: img.pip_install("typing-extensions>=4.13.0")

    # Terse rollout — only a digit, not a thinking trace.
    recipe.rollout_max_response_len = 64
    recipe.rollout_temperature = 1.0

    # EP8 everywhere (same override as modal_rl_rfm.py).
    recipe.rollout_num_gpus_per_engine = 8
    recipe.sglang_ep_size = 8
    recipe.sglang_dp_size = 8

    # 3 rollouts → 3 weight syncs → enough samples to compare.
    recipe.num_rollout = 3
    recipe.save_interval = 999
    recipe.global_batch_size = 4
    recipe.n_samples_per_prompt = 2

    if variant == "baseline":
        # Force the slow HfWeightIteratorDirect path (old behavior).
        object.__setattr__(recipe, "megatron_to_hf_mode", "")

    return TrainConfig(model=model, dataset=dataset, recipe=recipe)


def _launch(variant: str):
    """Build app and launch detached on Modal."""
    cfg = _build_config(variant=variant)
    app = cfg._build_app()
    print(f"Starting detached {variant} run: {cfg.training_run_id}")
    result = None
    modal_app_id = ""
    with modal.enable_output():
        with app.run(detach=True):
            modal_app_id = app.app_id or ""
            print(f"Detached app dashboard: {modal_app_dashboard_url(modal_app_id)}")
            arch = getattr(cfg.model, "architecture", None)
            if arch and getattr(arch, "needs_pre_conversion", False):
                app.download.remote()
                app.convert_checkpoint.remote()
            result = app.train.remote(
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_dashboard_url(modal_app_id),
            )
    if result is None:
        print(
            "Local client disconnected; detached app keeps running on Modal. "
            f"Track at {modal_app_dashboard_url(modal_app_id)}"
        )
        return None
    print(f"Training complete ({variant}): {result}")
    return result


# ── Modal local entrypoints ─────────────────────────────────────────────────
_cli_app = modal.App("weight-sync-debug-cli")


@_cli_app.local_entrypoint()
def train_baseline():
    """Launch baseline run (HfWeightIteratorDirect, slow path)."""
    return _launch("baseline")


@_cli_app.local_entrypoint()
def train_fixed():
    """Launch fixed run (HfWeightIteratorBridge, fast path)."""
    return _launch("fixed")
