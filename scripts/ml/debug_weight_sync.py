"""Minimal repro for slow weight sync on Qwen3.6-35B-A3B (MoE, 256 experts).

Root cause analysis
===================
The weight sync from Megatron → SGLang takes >1 hr because of the
HfWeightIteratorDirect path combined with this model's extreme expert count.

**Why HfWeightIteratorDirect (not Bridge)?**
Qwen3.6-35B-A3B sets ``megatron_model_type="qwen3.5-35B-A3B"`` on its
ModelArchitecture, which means ``needs_pre_conversion=True``.  The slime
launcher (launcher.py:142) clears ``megatron_to_hf_mode`` to ``""`` for
pre-conversion models, so slime falls back to HfWeightIteratorDirect
instead of the faster HfWeightIteratorBridge.

**What HfWeightIteratorDirect does per sync:**
1.  For every parameter across all 40 layers:
    - Non-expert params: TP all-gather across 2 ranks → small.
    - Expert params (256 experts × gate/up/down projections per layer):
      EP broadcast/all-gather across 8 ranks to reconstruct the full
      expert set from shards (each rank holds 256/8 = 32 experts).
2.  Converts each gathered parameter from Megatron format → HF naming.
3.  Chunks the resulting HF tensors into ``update_weight_buffer_size``
    buckets (default **512 MB**).
4.  For each chunk:
    a. Flatten tensors into FlattenedTensorBucket + CPU serialize via
       MultiprocessingSerializer.
    b. Gloo ``gather_object`` (CPU-side) to a single source rank.
    c. Source rank sends the serialized payload to the SGLang engine via
       Ray IPC (``ipc_engine.update_weights_from_tensor.remote``).
    d. SGLang deserializes, loads weights, returns ObjectRef.
    e. Wait on ObjectRef before freeing GPU tensors for the next chunk.

**Back-of-envelope for this model:**
- Total params ≈ 35 B  →  ~70 GB in BF16.
- 70 GB / 512 MB per chunk  ≈  **140 chunks**.
- Each chunk pays: GPU→CPU copy + serialize + Gloo gather + Ray IPC +
  SGLang load ≈ 20–30 s  (dominated by CPU serialize + Gloo on large
  payloads, plus GPU←→CPU copies for the expert-cache-to-CPU patch).
- 140 × 25 s ≈ **58 min** — consistent with the observed ~57 min/sync.

**Why MoE is uniquely bad here:**
Dense models (e.g. Qwen3-32B at ~32 B params) also go through this path
but have fewer total parameters *and* no EP all-gather overhead per expert
per layer.  The 256-expert MoE has:
- 40 layers × 256 experts × 3 projections = ~30 720 expert parameter
  tensors, each requiring an EP8 all-gather before the chunk pipeline.
- ``_patch_bridge_expert_cache_to_cpu`` moves each merged expert tensor
  to CPU to avoid OOM in colocated mode (SGLang shares the GPU), adding
  GPU→CPU→GPU round-trips.

Potential mitigations
=====================
1. **Increase ``update_weight_buffer_size``** — fewer, larger chunks mean
   fewer serialization + IPC round-trips.  Needs a slime CLI flag or
   monkey-patch (not currently exposed on SlimeRecipe).
2. **Re-enable bridge mode** — ``HfWeightIteratorBridge`` uses
   ``AutoBridge.export_hf_weights`` which may batch expert conversion more
   efficiently.  Requires verifying bridge compatibility with the
   qwen3.5-35B-A3B model type.
3. **Reduce the serialization path** — replace CPU serialize + Gloo gather
   with direct CUDA IPC handles (``torch.multiprocessing`` shared-memory
   tensors or NCCL send/recv), bypassing CPU entirely.
4. **Async / pipelined chunks** — overlap chunk N's SGLang load with chunk
   N+1's gather + serialize (currently sequential).

Repro
=====
This script launches a 1-step slime GRPO run on 1×8×H100 with the same
recipe used in ``modal_rl_rfm.py``.  It exists purely to trigger ONE
weight sync and measure its wall-clock time in the training logs (look for
``[weight_sync]`` or slime's own ``update_weights took ...`` log line).

    uv run modal run scripts/ml/debug_weight_sync.py::train

The dataset is a tiny synthetic stub (10 rows) so data prep is instant.
"""

import json
import os
import re

from modal_training_gym import (
    SlimeRecipe,
    TrainConfig,
    WandbConfig,
)
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import Qwen3_6_35B

# ── Tiny synthetic dataset (no AWS creds needed) ────────────────────────────
SYSTEM_PROMPT = "You are a chemistry expert. Respond with a number 0-1."
USER_PROMPT = (
    "Is this reaction feasible?\n\nReaction SMILES: CC>>CC\n\nProbability (0-1):"
)


class SyntheticRFMDataset(DatasetConfig):
    """10-row JSONL stub — just enough to trigger one training step + sync."""

    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    output_format = "jsonl"
    no_think: bool = True

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
        if eval_paths:
            for ep in eval_paths.values():
                os.makedirs(os.path.dirname(ep), exist_ok=True)
                with open(ep, "w") as f:
                    f.write(json.dumps(record) + "\n")


# ── Reward stub ─────────────────────────────────────────────────────────────
_PROB_RE = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")


def _parse_probability(text: str) -> float:
    for raw in reversed(_PROB_RE.findall(text)):
        try:
            val = float(raw)
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            return val
    return 0.5


async def rfm_reward(args, sample, **kwargs) -> float:
    text = sample.response
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        return 0.0
    pred = _parse_probability(text)
    true = (
        1.0
        if str(getattr(sample, "label", "")).strip().lower() in ("1", "true", "yes")
        else 0.0
    )
    return 1.0 - abs(pred - true)


# ── Training config ─────────────────────────────────────────────────────────
def build_config() -> TrainConfig:
    model = Qwen3_6_35B()
    dataset = SyntheticRFMDataset()

    recipe = SlimeRecipe.get_base_recipe(model)
    if recipe is None:
        raise ValueError(f"No base slime recipe for {model.model_name}")

    recipe.custom_rm_function = rfm_reward
    recipe.wandb = WandbConfig(
        project="weight-sync-debug",
        modal_wandb_secret_name="train-secrets",
    )
    recipe.image_overlay = lambda img: img.pip_install("typing-extensions>=4.13.0")

    # Terse rollout — only a digit, not a thinking trace.
    recipe.rollout_max_response_len = 64
    recipe.rollout_temperature = 1.0

    # ── EP8 everywhere (same override as modal_rl_rfm.py) ────────────────
    # One 8-GPU engine at EP8 so the sync is rank-aligned with Megatron's
    # EP8.  This is the BEST CASE for sync speed — it avoids the EP8→EP4
    # reshard the default 2×4 layout would require.  Still takes ~57 min.
    recipe.rollout_num_gpus_per_engine = 8
    recipe.sglang_ep_size = 8
    recipe.sglang_dp_size = 8

    # Absolute minimum work: 1 rollout, 1 save, small batch.
    recipe.num_rollout = 1
    recipe.save_interval = 999
    recipe.global_batch_size = 2
    recipe.n_samples_per_prompt = 2

    return TrainConfig(model=model, dataset=dataset, recipe=recipe)


def train():
    """Launch 1-step slime run to trigger and time the weight sync.

    Watch the logs for slime's ``update_weights took`` line — that is the
    wall-clock duration of a single Megatron→SGLang weight push.

    Run:
        uv run modal run scripts/ml/debug_weight_sync.py::train
    """
    import modal

    from modal_training_gym.common.modal_urls import modal_app_dashboard_url

    cfg = build_config()
    app = cfg._build_app()
    print(f"Starting weight-sync debug run: {cfg.training_run_id}")
    result = None
    modal_app_id = ""
    with modal.enable_output():
        with app.run(detach=True):
            modal_app_id = app.app_id or ""
            print(f"Dashboard: {modal_app_dashboard_url(modal_app_id)}")
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
            "Client disconnected; the detached app keeps running on Modal.  "
            f"Track at {modal_app_dashboard_url(modal_app_id)}"
        )
        return None
    print(f"Done: {result}")
    return result
