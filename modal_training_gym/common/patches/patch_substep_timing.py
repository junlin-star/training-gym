"""Wrap the driver loop, rollout worker and actor with measured substep timing.

One script for both frameworks: slime and miles have the same lanes and phase
names, only different source anchors, and running the same script at both image
builds is what keeps the record format from drifting apart.

Puts each phase of the training loop inside ``with _tg_rec.phase(...)``
The driver's phases all sit in one loop body, with a local (``_tg_rec``) recorder.
The rollout worker and the actor measure work inside a reward function/inside megatron's train
step, so they open a lane at the entry point and the phases below use the
module-level ``_tg_time_phase``, which finds the active lane in a ``ContextVar`` and does nothing
if there is none.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREAMBLE_MARKER = "PATCHED_TRAINING_GYM_TIMING_PREAMBLE"
RECORDER_MARKER = "PATCHED_TRAINING_GYM_TIMING_RECORDER"


def phase_marker(phase: str) -> str:
    return f"PATCHED_TRAINING_GYM_TIMING_{phase.upper()}"


PREAMBLE = (
    f"# {PREAMBLE_MARKER}: bootstrap substep-timing recorder\n"
    "import sys as _tg_sys\n"
    "if '/root' not in _tg_sys.path:\n"
    "    _tg_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_training_gym.common.timing_recorder import (\n"
    "        RoleRecorder as _TgRecorder,\n"
    "        recording_lane as _tg_role,\n"
    "        recording_lane_on_reporting_rank as _tg_mrec,\n"
    "        time_phase as _tg_time_phase,\n"
    "    )\n"
    "except ImportError:\n"
    "    print('WARNING: modal_training_gym not importable; substep timing off')\n"
    "    from contextlib import contextmanager as _tg_cm\n"
    "\n"
    "    class _TgRecorder:\n"
    "        def __init__(self, role, rollout_id): pass\n"
    "\n"
    "        def __enter__(self): return self\n"
    "\n"
    "        def __exit__(self, *exc): pass\n"
    "\n"
    "        @_tg_cm\n"
    "        def phase(self, name):\n"
    "            yield\n"
    "\n"
    "    @_tg_cm\n"
    "    def _tg_role(role, rollout_id):\n"
    "        yield _TgRecorder(role, rollout_id)\n"
    "\n"
    "    @_tg_cm\n"
    "    def _tg_mrec(rollout_id, role='actor'):\n"
    "        yield _TgRecorder(role, rollout_id)\n"
    "\n"
    "    @_tg_cm\n"
    "    def _tg_time_phase(name):\n"
    "        yield\n"
    "\n"
)


# The framework checkouts this script can patch, in the container.
SLIME_ROOT = Path("/root/slime")
MILES_ROOT = Path("/root/miles")


def replace_once(source: str, old: str, new: str, path: Path) -> str:
    """Replace exactly one occurrence, or raise.

    Short anchors are not unique in these files (``actor_model.update_weights()``
    appears twice, ``rollout_manager.eval.remote`` three times), and silently
    patching the wrong one is worse than failing the build.
    """
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected 1 occurrence of {old!r}, found {count}")
    return source.replace(old, new, 1)


def indent_block(block: str) -> str:
    return "\n".join(f"    {ln}" if ln.strip() else ln for ln in block.splitlines())


def wrap_block(block: str, phase: str, opener: str = "_tg_rec.phase") -> str:
    """Wrap a block in ``with <opener>('<phase>'):``.

    ``opener`` is ``_tg_rec.phase`` where the patch can see the active lane (the
    driver loop) and the module-level ``_tg_time_phase`` where it cannot.

    For a bare ``if`` with no ``else``, only the body is wrapped: wrapping the
    ``if`` itself records a ~0s interval on every step the branch is skipped,
    and a zero-width bar for work that never ran is indistinguishable from work
    that finished instantly. An ``if/else`` is wrapped whole, since wrapping one
    arm alone would not parse. The header can span several lines, and its
    closing ``):`` sits back at the ``if``'s own indent.
    """
    lines = block.splitlines(keepends=True)
    outer = len(lines[0]) - len(lines[0].lstrip(" "))
    head, body = "", block
    if lines[0].lstrip().startswith("if "):
        header = next(i for i, ln in enumerate(lines) if ln.rstrip().endswith(":")) + 1
        has_dedent_to_outer = any(
            ln.strip() and len(ln) - len(ln.lstrip(" ")) == outer
            for ln in lines[header:]
        )
        if not has_dedent_to_outer:
            head, body = "".join(lines[:header]), "".join(lines[header:])
    indent = body[: len(body) - len(body.lstrip(" "))]
    return (
        head
        + f"{indent}# {phase_marker(phase)}\n"
        + f"{indent}with {opener}('{phase}'):\n{indent_block(body)}\n"
    )


def _wrap_driver_loop(src: str, path: Path) -> str:
    """Wrap the driver ``for rollout_id in range(...)`` body in a recording lane."""
    lines = src.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "for rollout_id in range(args.start_rollout_id, args.num_rollout):" in line:
            break
    else:
        raise RuntimeError(f"{path}: driver rollout loop not found")

    loop_indent = line[: len(line) - len(line.lstrip(" "))]
    start = i + 1
    j = start
    while j < len(lines):
        body_line = lines[j]
        if body_line.strip() == "":
            j += 1
            continue
        body_indent = body_line[: len(body_line) - len(body_line.lstrip(" "))]
        if len(body_indent) <= len(loop_indent):
            break
        j += 1

    with_line = f"{loop_indent}    with _tg_role('driver', rollout_id) as _tg_rec:\n"
    marker = f"{loop_indent}    # {RECORDER_MARKER}: driver lane active\n"
    new_body = []
    for body_line in lines[start:j]:
        if body_line.strip():
            new_body.append("    " + body_line)
        else:
            new_body.append(body_line)
    new_lines = lines[: i + 1] + [marker, with_line] + new_body + lines[j:]
    return "".join(new_lines)


# Patches applied to the driver loop body after the rollout-status patcher has run.
_SYNC_PHASE_WRAPS = [
    (
        "        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:\n"
        "            # PATCHED_TRAINING_GYM_EVAL_BEGIN: eval-before-train substep start\n"
        "            _tg_report('evaluate_rollouts', args, rollout_id)\n"
        "            ray.get(rollout_manager.eval.remote(rollout_id))\n",
        "evaluate_rollouts",
    ),
    (
        "        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))\n",
        "generate_rollouts",
    ),
    (
        "        if args.offload_rollout:\n"
        "            # PATCHED_TRAINING_GYM_OFFLOAD_ROLLOUT_STATUS: rollout offload state\n"
        "            _tg_report('offload_rollout', args, rollout_id)\n"
        "            ray.get(rollout_manager.offload.remote())\n",
        "offload_rollout",
    ),
    (
        "        if args.use_critic:\n"
        "            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)\n"
        "            if actor_trains:\n"
        "                # PATCHED_TRAINING_GYM_COMPUTE_LOG_PROBS_STATUS: compute log probs state\n"
        "                _tg_report('compute_log_probs', args, rollout_id)\n"
        "                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))\n"
        "            else:\n"
        "                ray.get(value_refs)\n"
        "        else:\n"
        "            # PATCHED_TRAINING_GYM_COMPUTE_LOG_PROBS_STATUS: compute log probs state\n"
        "            _tg_report('compute_log_probs', args, rollout_id)\n"
        "            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))\n",
        "train_models",
    ),
    (
        "        if release_train or should_run_periodic_action(\n"
        "            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout\n"
        "        ):\n"
        "            # PATCHED_TRAINING_GYM_CHECKPOINT_SAVE_STATUS: checkpoint save state\n"
        "            _tg_report('checkpoint_save', args, rollout_id)\n"
        "            force_sync = release_train or rollout_id == args.num_rollout - 1\n"
        "            if actor_trains:\n"
        "                actor_model.save_model(rollout_id, force_sync=force_sync)\n"
        "            if args.use_critic:\n"
        "                critic_model.save_model(rollout_id, force_sync=force_sync)\n"
        "            if args.rollout_global_dataset:\n"
        "                ray.get(rollout_manager.save.remote(rollout_id))\n",
        "checkpoint_save",
    ),
    (
        "        # PATCHED_TRAINING_GYM_OFFLOAD_TRAIN_STATUS: train offload state\n"
        "        _tg_report('offload_train', args, rollout_id)\n"
        "        offload_train(actor_trains)\n",
        "offload_train",
    ),
    (
        "        # PATCHED_TRAINING_GYM_WEIGHT_SYNC_STATUS: weight sync state\n"
        "        _tg_report('weight_sync', args, rollout_id)\n"
        "        actor_model.update_weights()\n",
        "weight_sync",
    ),
    (
        "        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):\n"
        "            # PATCHED_TRAINING_GYM_EVAL_END: eval-after-train substep start\n"
        "            _tg_report('evaluate_rollouts', args, rollout_id)\n"
        "            ray.get(rollout_manager.eval.remote(rollout_id))\n",
        "evaluate_rollouts_end",
    ),
]


# train_async.py is a separate file with a different loop: no eval before train,
# no rollout offload, no offload_train, and the wait is on a future prefetched
# during the previous step. Same phase names, different anchors.
_ASYNC_PHASE_WRAPS = [
    (
        "        if rollout_data_next_future is not None:\n"
        "            rollout_data_curr_ref = ray.get(rollout_data_next_future)\n",
        "wait_for_rollout",
    ),
    (
        "        if args.use_critic:\n"
        "            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)\n"
        "            if actor_trains:\n"
        "                # PATCHED_TRAINING_GYM_COMPUTE_LOG_PROBS_STATUS: compute log probs state\n"
        "                _tg_report('compute_log_probs', args, rollout_id)\n"
        "                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))\n"
        "            else:\n"
        "                ray.get(value_refs)\n"
        "        else:\n"
        "            # PATCHED_TRAINING_GYM_COMPUTE_LOG_PROBS_STATUS: compute log probs state\n"
        "            _tg_report('compute_log_probs', args, rollout_id)\n"
        "            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))\n",
        "train_models",
    ),
    (
        "        if release_train or should_run_periodic_action(\n"
        "            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout\n"
        "        ):\n"
        "            # PATCHED_TRAINING_GYM_CHECKPOINT_SAVE_STATUS: checkpoint save state\n"
        "            _tg_report('checkpoint_save', args, rollout_id)\n"
        "            force_sync = release_train or rollout_id == args.num_rollout - 1\n"
        "            if actor_trains:\n"
        "                actor_model.save_model(rollout_id, force_sync=force_sync)\n"
        "            if args.use_critic:\n"
        "                critic_model.save_model(rollout_id, force_sync=force_sync)\n"
        "            if args.rollout_global_dataset:\n"
        "                ray.get(rollout_manager.save.remote(rollout_id))\n",
        "checkpoint_save",
    ),
    # Where an async run actually waits for generation from the second step on:
    # this consumes the prefetched future because weights cannot be updated mid
    # generation. Measured apart from the weight update, which is the number
    # async mode is judged on.
    (
        "            # sync generate before update weights to prevent update weight in the middle of generation\n"
        "            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None\n"
        "            rollout_data_next_future = None\n",
        "wait_for_rollout",
    ),
    (
        "            # PATCHED_TRAINING_GYM_WEIGHT_SYNC_STATUS: weight sync state\n"
        "            _tg_report('weight_sync', args, rollout_id)\n"
        "            actor_model.update_weights()\n",
        "weight_sync",
    ),
    (
        "        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):\n"
        "            # PATCHED_TRAINING_GYM_EVAL_END: eval-after-train substep start\n"
        "            _tg_report('evaluate_rollouts', args, rollout_id)\n"
        "            ray.get(rollout_manager.eval.remote(rollout_id))\n",
        "evaluate_rollouts_end",
    ),
]

SLIME_ENTRYPOINTS = {
    "train.py": _SYNC_PHASE_WRAPS,
    "train_async.py": _ASYNC_PHASE_WRAPS,
}


@dataclass(frozen=True)
class PackageTarget:
    """One file in the slime *package* that measures a non-driver lane.

    ``scope`` opens the recorder: the first line of the function that knows the
    rollout id, the line ending its body, and the ``with`` header to insert.
    ``blocks`` are the phases, wrapped with the module-level ``_tg_time_phase``
    because they may sit in a different module from the scope entirely --
    ``forward_backward`` is in ``model.py``, its recorder in ``actor.py``.
    """

    path: str
    scope: tuple[str, str, str] | None
    blocks: tuple[tuple[str, str], ...]


# The rollout worker: one lane per generate call. ``generate`` runs an asyncio
# loop, so its phases overlap -- which is why a phase records its own summed
# duration alongside first start and last end.
ROLLOUT_TARGET = PackageTarget(
    path="slime/ray/rollout.py",
    scope=(
        "    def generate(self, rollout_id):\n",
        "        return self._split_train_data_by_dp(data)\n",
        "with _tg_role('rollout', rollout_id):",
    ),
    blocks=(
        # Named apart from the driver's ``generate_rollouts``, which is the same
        # work seen from the caller: one name per lane keeps the run-level phase
        # summary from counting generation twice.
        (
            "generate_samples",
            "        data, metrics = self._get_rollout_data(rollout_id=rollout_id)\n",
        ),
        (
            "reward_post_process",
            "        raw_rewards, rewards = self._post_process_rewards(samples)\n",
        ),
    ),
)

# Rewards run on slime's background event-loop thread (`utils/async_utils.py`
# submits with `asyncio.run_coroutine_threadsafe`), not on the thread that
# opened the rollout lane -- and they still see it: that call schedules task
# creation with `call_soon_threadsafe`, which copies the *submitting* thread's
# context, and every task spawned below inherits it.
#
# `batched_async_rm` is only reached by the group and fan-out paths (under
# `--group-rm`, or a custom generate function returning several samples). The
# default path scores one sample at a time through `async_rm`, so measuring
# only here left the lane empty on every ordinary run, including one with a
# custom reward function; REWARD_SAMPLE_TARGET covers that path.
REWARD_TARGET = PackageTarget(
    path="slime/rollout/rm_hub/__init__.py",
    scope=None,
    blocks=(
        (
            "reward",
            "    if args.custom_rm_path is not None:\n"
            "        # Ensure the custom reward function is implemented in batch mode\n"
            "        rm_function = load_function(args.custom_rm_path)\n"
            "        return await rm_function(args, samples, **kwargs)\n"
            "    tasks = [async_rm(args, sample, **kwargs) for sample in samples]\n"
            "    rewards = await asyncio.gather(*tasks)\n"
            "    return rewards\n",
        ),
    ),
)

# The default reward path: one sample, scored where it was generated. Same lane
# and phase name as the batched site -- a run uses one path or the other, and
# what the phase answers ("how much of this rollout went to scoring") does not
# depend on which.
REWARD_SAMPLE_TARGET = PackageTarget(
    path="slime/rollout/sglang_rollout.py",
    scope=None,
    blocks=(
        (
            "reward",
            "        if sample.reward is None:\n"
            '            with trace_span(sample, "reward_model"):\n'
            "                sample.reward = await async_rm(args, sample)\n",
        ),
    ),
)

# The actor and the critic are one class and one lane each, told apart by
# ``self.role`` -- so the header is an expression, not a literal, and one patch
# instruments both.
ACTOR_TARGET = PackageTarget(
    path="slime/backends/megatron_utils/actor.py",
    scope=(
        "    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):\n",
        "        return result\n",
        "with _tg_mrec(rollout_id, 'critic' if self.role == 'critic' else 'actor'):",
    ),
    blocks=(
        # Called up to four times per step (ref, teacher, old actor, actor).
        (
            "compute_log_probs",
            '        with timer(f"{store_prefix}log_probs"):\n'
            "            return forward_only(\n"
            "                get_log_probs_and_entropy,\n"
            "                self.args,\n"
            "                self.model,\n"
            "                data_iterator,\n"
            "                num_microbatches,\n"
            "                store_prefix=store_prefix,\n"
            "                use_rollout_top_p_replay=True,\n"
            "            )\n",
        ),
    ),
)

# No scope: ``train_one_step`` runs inside the actor's ``train`` on the same
# thread, so it inherits that lane. Both phases are per *step*, not per
# microbatch -- ``forward_backward_func`` loops over microbatches internally.
TRAIN_STEP_TARGET = PackageTarget(
    path="slime/backends/megatron_utils/model.py",
    scope=None,
    blocks=(
        (
            "forward_backward",
            "    losses_reduced = forward_backward_func(\n"
            "        forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),\n"
            "        data_iterator=data_iterator,\n"
            "        model=model,\n"
            "        num_microbatches=num_microbatches,\n"
            "        seq_length=args.seq_length,\n"
            "        micro_batch_size=args.micro_batch_size,\n"
            "        decoder_seq_length=args.decoder_seq_length,\n"
            "        forward_only=False,\n"
            "    )\n",
        ),
        # Body of ``if valid_step:``, so a skipped step (NaN grads) records
        # nothing instead of a 0s bar claiming an instant optimizer step.
        (
            "optimizer_step",
            "    if valid_step:\n"
            "        # Update parameters.\n"
            "        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()\n"
            "\n"
            "        # Update learning rate. Use the per-step global_batch_size when dynamic\n"
            "        # batching is on so the scheduler's samples-seen counter tracks reality.\n"
            "        assert update_successful\n"
            "        opt_param_scheduler.step(increment=step_global_batch_size)\n",
        ),
    ),
)

SLIME_PACKAGE_TARGETS: tuple[PackageTarget, ...] = (
    ROLLOUT_TARGET,
    REWARD_TARGET,
    REWARD_SAMPLE_TARGET,
    ACTOR_TARGET,
    TRAIN_STEP_TARGET,
)


# ---------- miles ----------
#
# Same lanes and phase names as slime, different source. Two differences drive
# every anchor below: miles' driver loop is ``async``, so the phases wrap
# ``await`` expressions rather than ``ray.get`` calls, and no status patcher
# runs before this one, so the anchors are the pristine upstream lines.

_MILES_SYNC_PHASE_WRAPS = [
    (
        "        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:\n"
        "            await rollout_manager.eval.remote(rollout_id)\n",
        "evaluate_rollouts",
    ),
    (
        "        rollout_data_ref = await rollout_manager.generate.remote(rollout_id)\n",
        "generate_rollouts",
    ),
    (
        "        if args.offload_rollout:\n"
        "            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]\n"
        '            if "kv_cache" in args.offload_rollout_level:\n'
        "                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)\n"
        '            if "weight" in args.offload_rollout_level:\n'
        "                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)\n"
        "            await rollout_manager.offload.remote(tags=offload_tags)\n",
        "offload_rollout",
    ),
    (
        "        if args.use_critic:\n"
        "            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_ref))\n"
        "            if rollout_id >= args.num_critic_only_steps:\n"
        "                await actor_model.train(rollout_id, rollout_data_ref)\n"
        "            await critic_task\n"
        "        else:\n"
        "            await actor_model.train(rollout_id, rollout_data_ref)\n",
        "train_models",
    ),
    (
        "        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):\n"
        "            await save(rollout_id)\n",
        "checkpoint_save",
    ),
    ("        await offload_train()\n", "offload_train"),
    # The onload calls belong to the weight update: the rollout engines cannot
    # take new weights until they are back on the GPU.
    (
        "        if args.offload_rollout:\n"
        "            await rollout_manager.onload_weights.remote()\n"
        "        await actor_model.update_weights()\n"
        "        if args.offload_rollout:\n"
        "            await rollout_manager.onload_kv.remote()\n",
        "weight_sync",
    ),
    (
        "        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):\n"
        "            await rollout_manager.eval.remote(rollout_id)\n",
        "evaluate_rollouts_end",
    ),
]

_MILES_ASYNC_PHASE_WRAPS = [
    (
        "        if rollout_data_next_future is not None:\n"
        "            rollout_data_curr_ref = await rollout_data_next_future\n",
        "wait_for_rollout",
    ),
    (
        "        if args.use_critic:\n"
        "            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_curr_ref))\n"
        "            if rollout_id >= args.num_critic_only_steps:\n"
        "                await actor_model.train(rollout_id, rollout_data_curr_ref)\n"
        "            await critic_task\n"
        "        else:\n"
        "            await actor_model.train(rollout_id, rollout_data_curr_ref)\n",
        "train_models",
    ),
    (
        "        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):\n"
        "            await actor_model.save_model(\n"
        "                rollout_id,\n"
        "                force_sync=rollout_id == args.num_rollout - 1,\n"
        "            )\n"
        "            if args.use_critic:\n"
        "                await critic_model.save_model(\n"
        "                    rollout_id,\n"
        "                    force_sync=rollout_id == args.num_rollout - 1,\n"
        "                )\n"
        "            if args.rollout_global_dataset:\n"
        "                await rollout_manager.save.remote(rollout_id)\n",
        "checkpoint_save",
    ),
    # Where an async run actually waits for generation: the prefetched future is
    # consumed here because weights cannot be updated mid generation. Measured
    # apart from the weight update, which is the number async mode is judged on.
    (
        "            # sync generate before update weights to prevent update weight in the middle of generation\n"
        "            rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None\n"
        "            rollout_data_next_future = None\n",
        "wait_for_rollout",
    ),
    # Indented: the bring-up call before the loop is at module-function level.
    ("            await actor_model.update_weights()\n", "weight_sync"),
    (
        "        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):\n"
        "            await rollout_manager.eval.remote(rollout_id)\n",
        "evaluate_rollouts_end",
    ),
]

MILES_ENTRYPOINTS = {
    "train.py": _MILES_SYNC_PHASE_WRAPS,
    "train_async.py": _MILES_ASYNC_PHASE_WRAPS,
}

MILES_PACKAGE_TARGETS: tuple[PackageTarget, ...] = (
    PackageTarget(
        path="miles/ray/rollout/rollout_manager.py",
        scope=(
            "    async def generate(self, rollout_id):\n",
            '        return split_train_data_by_dp(self.args, data, self.train_parallel_config["dp_size"])\n',
            "with _tg_role('rollout', rollout_id):",
        ),
        blocks=(
            (
                "generate_samples",
                "        data, metadata, metrics = await self._get_rollout_data(rollout_id=rollout_id)\n",
            ),
            (
                "reward_post_process",
                "        data = convert_samples_to_train_data(\n"
                "            self.args,\n"
                "            data,\n"
                "            metadata=metadata,\n"
                "            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,\n"
                "            custom_reward_post_process_func=self.custom_reward_post_process_func,\n"
                "        )\n",
            ),
        ),
    ),
    PackageTarget(
        path="miles/rollout/rm_hub/__init__.py",
        scope=None,
        blocks=(
            (
                "reward",
                "    if args.custom_rm_path is not None:\n"
                "        # Ensure the custom reward function is implemented in batch mode\n"
                "        rm_function = load_function(args.custom_rm_path)\n"
                "        return await rm_function(args, samples, **kwargs)\n"
                "    tasks = [async_rm(args, sample, **kwargs) for sample in samples]\n"
                "    rewards = await asyncio.gather(*tasks)\n"
                "    return rewards\n",
            ),
        ),
    ),
    PackageTarget(
        path="miles/rollout/sglang_rollout.py",
        scope=None,
        blocks=(
            (
                "reward",
                "        if sample.reward is None:\n"
                "            sample.reward = await async_rm(args, sample)\n",
            ),
        ),
    ),
    PackageTarget(
        path="miles/backends/megatron_utils/actor.py",
        scope=(
            "    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:\n",
            "            return self.train_actor(rollout_id, rollout_data)\n",
            "with _tg_mrec(rollout_id, 'critic' if self.role == 'critic' else 'actor'):",
        ),
        blocks=(
            (
                "compute_log_probs",
                '        with timer(f"{store_prefix}log_probs"):\n'
                "            return forward_only(\n"
                "                get_log_probs_and_entropy,\n"
                "                self.args,\n"
                "                self.model,\n"
                "                data_iterator,\n"
                "                num_microbatches,\n"
                "                store_prefix=store_prefix,\n"
                "            )\n",
            ),
        ),
    ),
    PackageTarget(
        path="miles/backends/megatron_utils/model.py",
        scope=None,
        blocks=(
            (
                "forward_backward",
                "    losses_reduced = forward_backward_func(\n"
                "        forward_step_func=forward_step,\n"
                "        data_iterator=data_iterator,\n"
                "        model=model,\n"
                "        num_microbatches=num_microbatches,\n"
                "        seq_length=args.seq_length,\n"
                "        micro_batch_size=args.micro_batch_size,\n"
                "        decoder_seq_length=args.decoder_seq_length,\n"
                "        forward_only=False,\n"
                "    )\n",
            ),
            (
                "optimizer_step",
                "    if not disable_optimizer and valid_step:\n"
                "        # Update parameters.\n"
                "        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()\n"
                "\n"
                "        # Update learning rate.\n"
                "        assert update_successful\n"
                "        opt_param_scheduler.step(increment=args.global_batch_size)\n",
            ),
        ),
    ),
)


def wrap_scope(src: str, scope: tuple[str, str, str], path: Path) -> str:
    """Put a function body inside the recorder's ``with``.

    Anchored on its first and last lines rather than its whole text, because
    the phases inside it have already been rewritten by the time this runs.
    """
    signature, last_line, header = scope
    if src.count(signature) != 1:
        raise RuntimeError(
            f"{path}: expected 1 occurrence of {signature.strip()!r}, "
            f"found {src.count(signature)}"
        )
    head, _, rest = src.partition(signature)
    body, sep, tail = rest.partition(last_line)
    if not sep:
        raise RuntimeError(f"{path}: scope end not found: {last_line.strip()!r}")
    indent = " " * (len(signature) - len(signature.lstrip(" ")) + 4)
    # The match has to be the function's own last line: an earlier line with
    # the same text would close the lane early, and the rest of the function
    # would keep running, measured by nothing.
    after = next((ln for ln in tail.splitlines() if ln.strip()), "")
    if after.startswith(indent):
        raise RuntimeError(
            f"{path}: {last_line.strip()!r} is not the end of "
            f"{signature.strip()!r}; {after.strip()!r} follows it"
        )
    return (
        head
        + signature
        + f"{indent}# {RECORDER_MARKER}\n"
        + f"{indent}{header}\n"
        + indent_block(body + last_line)
        + "\n"
        + tail
    )


def patch_package_file(root: Path, target: PackageTarget) -> None:
    """Instrument one file in the framework package.

    A missing file is fatal here, unlike the entrypoints: these paths are
    inside the pinned framework checkout, so a missing one means the layout
    moved and the lane would silently never appear.
    """
    path = root / target.path
    if not path.exists():
        raise RuntimeError(f"{path}: not found; {root.name} layout changed")

    src = path.read_text()
    if PREAMBLE_MARKER in src:
        print(f"{target.path} already patched for substep timing")
        return

    for phase, block in target.blocks:
        src = replace_once(src, block, wrap_block(block, phase, "_tg_time_phase"), path)
    if target.scope is not None:
        src = wrap_scope(src, target.scope, path)  # last: it reindents the body
    src = PREAMBLE + src

    missing = [phase for phase, _ in target.blocks if phase_marker(phase) not in src]
    if missing:
        raise RuntimeError(f"{path}: phases not instrumented: {missing}")
    if target.scope is not None and RECORDER_MARKER not in src:
        raise RuntimeError(f"{path}: recorder scope not instrumented")
    compile(src, str(path), "exec")

    path.write_text(src)
    print(f"Patched {target.path} for substep timing ({len(target.blocks)} phases)")


def _patch_file(path: Path, wraps: list[tuple[str, str]]) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping substep timing patch")
        return

    src = path.read_text()
    if PREAMBLE_MARKER in src:
        print(f"{path.name} already patched for substep timing")
        return

    src = PREAMBLE + src
    for old, phase in wraps:
        src = replace_once(src, old, wrap_block(old, phase), path)
    src = _wrap_driver_loop(src, path)  # last: it reindents the loop body

    missing = [phase for _, phase in wraps if phase_marker(phase) not in src]
    if missing:
        raise RuntimeError(f"{path}: phases not instrumented: {missing}")
    compile(src, str(path), "exec")

    path.write_text(src)
    print(f"Patched {path.name} for substep timing ({len(wraps)} phases)")


def main() -> None:
    """Patch whichever framework checkout this image has.

    Both launchers run this same script at image build, so the lanes, phase
    names and record format cannot drift between slime and miles.
    """
    for root, entrypoints, package_targets in (
        (SLIME_ROOT, SLIME_ENTRYPOINTS, SLIME_PACKAGE_TARGETS),
        (MILES_ROOT, MILES_ENTRYPOINTS, MILES_PACKAGE_TARGETS),
    ):
        if not root.is_dir():
            continue
        for name, wraps in entrypoints.items():
            _patch_file(root / name, wraps)
        for target in package_targets:
            patch_package_file(root, target)


if __name__ == "__main__":
    main()
