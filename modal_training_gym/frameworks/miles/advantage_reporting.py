"""Torch / Megatron advantage-distribution math for miles' dashboard
reporting.
"""

from __future__ import annotations

import time

from modal_training_gym.common.reporting import (
    _arg_value,
    _enqueue_advantage,
    _positive_int,
    _run_context,
)
from modal_training_gym.frameworks.slime.advantage_reporting import (
    _advantage_samples_payload as _advantage_samples_payload,
)

_warned_compute_failure = False


def _warn_once(exc: Exception) -> None:
    """Log the first advantage-math failure; the reporter itself stays fail-open."""
    global _warned_compute_failure
    if _warned_compute_failure:
        return
    _warned_compute_failure = True
    print(
        f"[training-gym] advantage-distribution reporting disabled: {exc!r}",
        flush=True,
    )


def report_advantage_distribution(
    rollout_id: int,
    args: object,
    rollout_data: object,
) -> None:
    """Emit per-sample advantages (tagged with their GRPO group) for one step.

    ``args`` is miles' argparse namespace and ``rollout_data`` its per-step
    dict of tensors — neither type is importable outside the training
    container, so both are taken as ``object`` and duck-typed.

    Injected into miles' train-side ``log_rollout_data`` (in
    ``miles/backends/training_utils/log_utils.py``) so it fires right after
    advantages are computed. miles itself only logs the *mean* advantage per
    step; this captures the full per-sample distribution.

    Runs on every actor rank but only the TP-rank-0 / last-PP-stage ranks hold
    the reduced advantages, and within those only CP-rank-0 posts (after a CP
    all-reduce makes each sample's mean cover its full response). Each
    surviving rank covers its own data-parallel shard of the step's samples;
    the dashboard merges shards into per-group distributions.
    """
    if not isinstance(rollout_data, dict):
        return
    try:
        import torch
        import torch.distributed as dist
        from miles.backends.training_utils.parallel import get_parallel_state
    except Exception:
        return

    try:
        parallel_state = get_parallel_state()
        if not (parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage):
            return
    except Exception:
        return

    advantages = rollout_data.get("advantages")
    loss_masks = rollout_data.get("loss_masks")
    response_lengths = rollout_data.get("response_lengths")
    total_lengths = rollout_data.get("total_lengths")
    if not advantages or loss_masks is None:
        return
    if response_lengths is None or total_lengths is None:
        return

    n = len(advantages)
    try:
        device = advantages[0].device
        sums = torch.zeros(n, dtype=torch.float64, device=device)
        counts = torch.zeros(n, dtype=torch.float64, device=device)
        failed = torch.zeros(1, dtype=torch.float64, device=device)
        cp_size = parallel_state.cp.size
        cp_rank = parallel_state.cp.rank
    except Exception as exc:
        _warn_once(exc)
        return

    compute_exc: Exception | None = None
    try:
        if cp_size == 1:
            local_masks = loss_masks
        else:
            from miles.backends.training_utils.cp_utils import (
                get_local_response_loss_masks,
            )

            local_masks = get_local_response_loss_masks(
                total_lengths=list(total_lengths),
                response_lengths=list(response_lengths),
                loss_masks=loss_masks,
                qkv_format=getattr(args, "qkv_format", "thd"),
                max_seq_lens=rollout_data.get("max_seq_lens"),
            )

        for i in range(n):
            adv = advantages[i].to(torch.float64)
            mask = local_masks[i].to(torch.float64)
            m = min(adv.numel(), mask.numel())
            sums[i] = (adv[:m] * mask[:m]).sum()
            counts[i] = mask[:m].sum()
    except Exception as exc:
        compute_exc = exc
        failed[0] = 1.0

    try:
        if cp_size > 1:
            # Every CP rank holds a token-shard of the same samples; reduce so
            # each sample's mean is taken over its full response.
            from miles.utils.ft_utils.process_group_utils import GeneralPGUtil

            cp_group = parallel_state.cp.group
            pg_util = GeneralPGUtil.create(cp_group)
            pg_util.all_reduce(sums, cp_group, dist.ReduceOp.SUM)
            pg_util.all_reduce(counts, cp_group, dist.ReduceOp.SUM)
            pg_util.all_reduce(failed, cp_group, dist.ReduceOp.SUM)
    except Exception as exc:
        _warn_once(exc)
        return

    if compute_exc is not None:
        _warn_once(compute_exc)
    if failed.item() > 0:
        return

    if cp_rank != 0:
        return

    raw_rewards = list(rollout_data.get("raw_reward") or [])
    sample_indices = list(rollout_data.get("sample_indices") or range(n))
    n_per = _positive_int(_arg_value(args, "n_samples_per_prompt")) or 1

    samples = _advantage_samples_payload(
        sums.tolist(),
        counts.tolist(),
        sample_indices,
        raw_rewards,
        n_per,
    )
    if not samples:
        return

    try:
        dp_rank = int(parallel_state.effective_dp.rank)
    except Exception:
        dp_rank = 0

    _enqueue_advantage(
        {
            **_run_context(args),
            "rollout_id": int(rollout_id),
            "created_at": int(time.time()),
            "dp_rank": dp_rank,
            "n_samples_per_prompt": int(n_per),
            "samples": samples,
        }
    )
