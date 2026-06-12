"""Patch compute_advantages_and_returns to handle missing tensor references.

When a custom generate function is used with kl_coef=0 and no critic,
the ``can_reuse_log_probs_in_loss`` optimisation in ``train_actor``
skips computing log_probs before advantage calculation.  Standard
rollouts still provide ``rollout_log_probs`` as a fallback, but custom
generate functions do not collect them.  This leaves all three tensor
references (log_probs, rollout_log_probs, values) as None, crashing the
``[torch.zeros_like(x, ...) for x in xs]`` list comprehension with
``TypeError: 'NoneType' object is not iterable``.

This patch makes the None-xs path fall back to creating zero-KL tensors
from ``loss_masks`` shapes.  When context parallelism is active (CP>1),
the fallback tensors are sliced to CP-chunked sizes so that downstream
consumers like ``log_rollout_data`` (which splits concatenated tensors
by CP chunk lengths) see consistent shapes.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path("/root/slime/slime/backends/megatron_utils/loss.py")
src = p.read_text()

old = """\
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]"""

new = """\
        xs = log_probs or rollout_log_probs or values
        if xs is not None:
            kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
        else:
            _dev = loss_masks[0].device if loss_masks else torch.cuda.current_device()
            _cp_size = mpu.get_context_parallel_world_size()
            if _cp_size > 1:
                kl = [
                    slice_log_prob_with_cp(
                        torch.zeros(_rl, dtype=torch.float32, device=_dev),
                        _tl, _rl, args.qkv_format,
                        max_seq_lens[_i] if max_seq_lens else None,
                    )
                    for _i, (_rl, _tl) in enumerate(zip(response_lengths, total_lengths))
                ]
            else:
                kl = [torch.zeros(rl, dtype=torch.float32, device=_dev) for rl in response_lengths]"""

if old in src:
    p.write_text(src.replace(old, new, 1))
