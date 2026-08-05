"""Wrap slime's driver loop, rollout worker and actor with measured substep timing.

Puts each phase of the training loop inside ``with _tg_rec.phase(...)``
The driver's phases all sit in one loop body, with a local (``_tg_rec``) recorder.
The rollout worker and the actor measure work inside a reward function/inside megatron's train
step, so they open a lane at the entry point and the phases below use the
module-level ``_tg_time_phase``, which finds the active lane in a ``ContextVar`` and does nothing
if there is none.
"""

from __future__ import annotations

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
    "    from modal_training_gym.frameworks.slime.phase_reporting import (\n"
    "        RoleRecorder as _TgRecorder,\n"
    "        recording_lane as _tg_role,\n"
    "        recording_lane_on_reporting_rank as _tg_mrec,\n"
    "        time_phase as _tg_time_phase,\n"
    "    )\n"
    "except ImportError:\n"
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


# Files inside /root/slime to patch.
PACKAGE_TARGETS = ["train.py", "train_async.py"]


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
    arm alone would not parse.
    """
    lines = block.splitlines(keepends=True)
    outer = len(lines[0]) - len(lines[0].lstrip(" "))
    has_dedent_to_outer = any(
        ln.strip() and len(ln) - len(ln.lstrip(" ")) == outer for ln in lines[1:]
    )
    if lines[0].lstrip().startswith("if ") and not has_dedent_to_outer:
        head, body = lines[0], "".join(lines[1:])
    else:
        head, body = "", block
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
_DRIVER_PHASE_WRAPS = [
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
        "evaluate_rollouts",
    ),
]


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping substep timing patch")
        return

    src = path.read_text()
    if PREAMBLE_MARKER in src:
        print(f"{path.name} already patched for substep timing")
        return

    src = PREAMBLE + src

    failed: list[str] = []
    for old, phase in _DRIVER_PHASE_WRAPS:
        try:
            src = replace_once(src, old, wrap_block(old, phase), path)
        except RuntimeError:
            failed.append(phase)

    try:
        src = _wrap_driver_loop(src, path)
    except RuntimeError as exc:
        print(f"WARNING: {exc}")
        if failed:
            print(f"WARNING: Could not patch {path.name} for: {', '.join(failed)}")
        path.write_text(src)
        print(f"Patched {path.name} with substep timing preamble only")
        return

    if failed:
        print(f"WARNING: Could not patch {path.name} for: {', '.join(failed)}")

    path.write_text(src)
    print(f"Patched {path.name} for substep timing")


def main() -> None:
    for target in PACKAGE_TARGETS:
        _patch_file(Path("/root/slime") / target)


if __name__ == "__main__":
    main()
