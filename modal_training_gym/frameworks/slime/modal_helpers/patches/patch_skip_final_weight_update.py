"""Patch slime train loop to avoid unnecessary final rollout weight sync.

When the last rollout has already trained and saved a checkpoint, updating the
co-located rollout engines is only needed if a final in-loop eval will run. For
large models this extra broadcast can fail under post-checkpoint resource
pressure, even though the training checkpoint is complete.

Executed at image-build time via ``python3 <this file>``.
"""

from pathlib import Path

TARGET = Path("/root/slime/train.py")
MARKER = "PATCHED_SKIP_FINAL_WEIGHT_UPDATE"

OLD = """        offload_train(actor_trains_this_step)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))
"""

NEW = f"""        offload_train(actor_trains_this_step)
        run_eval = should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch)
        needs_rollout_weights = rollout_id < args.num_rollout - 1 or run_eval
        if needs_rollout_weights:  # {MARKER}
            if args.offload_rollout:
                ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()
            if args.offload_rollout:
                ray.get(rollout_manager.onload_kv.remote())

        if run_eval:
            ray.get(rollout_manager.eval.remote(rollout_id))
"""

if not TARGET.exists():
    print(f"WARNING: {TARGET} not found, skipping final-weight-update patch")
else:
    src = TARGET.read_text()
    if MARKER in src:
        print("train.py already patched to skip unnecessary final weight update")
    elif OLD in src:
        TARGET.write_text(src.replace(OLD, NEW, 1))
        print(f"Patched {TARGET}: skipped unnecessary final weight update")
    else:
        print("WARNING: Could not find train.py final weight-update block")
