"""8×H100 run of the Qwen3-ASR audio GRPO example, on native training-gym main.

Thin wrapper over train_qwen3_asr.build_train_config — inherits every native-stack
fix (compat shim, qkv_format=bshd, disable_hf_conversion) and scales to a full
8×H100 node.

This uses the SAME data slice + batch dynamics that trained clean at 2-GPU
(n_clips=8, rollout_batch_size=4, global_batch_size=8, micro_batch_size=1) — only
the GPU count (2→8) and run length (num_rollout 8→50) change. A larger/different
slice (n_clips=64) exposed a numerically pathological clip that produced an inf
gradient under GRPO (no sequence packing on the native bridge → micro_batch_size=1
can't dilute it); see NATIVE_STACK_CHANGES.md. Keeping the proven slice avoids it.

  uv run python train_qwen3_asr_scale.py

Batch math (colocate, TP=1 → DP=8): global_batch_size=8 → 1 sample/rank/step;
rollout_batch_size=4 × n=8 = 32 samples/rollout = 4 steps/rollout.
"""

from __future__ import annotations

from train_qwen3_asr import build_train_config

if __name__ == "__main__":
    result = build_train_config(
        n_clips=8,
        actor_num_gpus_per_node=8,
        num_rollout=50,
        rollout_batch_size=4,
        global_batch_size=8,
        exp_name="qwen3-asr-grpo-scale-8gpu",
    ).train()
    print("training_run_id:", result.training_run_id)
