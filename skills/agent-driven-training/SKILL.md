---
name: agent-driven-training
description: >-
  Use when asked to train, post-train, fine-tune, or improve a model with
  the training gym. Own the full iteration loop: create a fresh config,
  validate one step, run a smoke test, monitor with the CLI, fix failures or
  poor reward, and promote only a healthy run.
---

# Agent-driven training

## 0. Make observability work first

From the repo root:

```bash
uv run training-gym run list --limit 1
```

If this returns `authentication_failed`, immediately run the remedy named in
the error (`uv run training-gym set-proxy-auth` or
`uv run training-gym set-password`) and retry. For Basic Auth, export the same
password as `TRAINING_GYM_DASHBOARD_PASSWORD` in the agent's shell. Do not
bypass the auth problem with dashboard scraping or raw `modal app` commands.

Use the documented CLI directly. Do not call an assumed helper script under a
skill directory.

## 1. Prove one step

Create and launch a fresh configuration for one training step, then monitor its
new run ID:

```bash
uv run training-gym run get <run-id> --verbose
```

Poll `run get` periodically. Advance only when the fresh run completes one
rollout, records a nonempty reward, and has no traceback. If it fails or stops
advancing, inspect:

```bash
uv run training-gym run logs <run-id> --tail 200
uv run training-gym run params <run-id>
```

## 2. Run a smoke test

Launch a **new** run from the same configuration and topology with about 10
steps:

```bash
uv run training-gym run get <run-id> --verbose
```

Continue polling; do not merely wait on the local process. Advance when the run
completes, reward data spans the smoke test, and step times are stable enough
to estimate full-run cost.

- Failed or not advancing: inspect its logs and parameters as above.
- Reward is flat or declining: use `training-gym run trace` to compare
  low-reward samples from early and late steps.
- Learning, but too slow: compare rollout times across several steps. Keep the
  GPU topology fixed; the observed one-GPU probe changed the shared Megatron
  cache layout and forced the later eight-GPU run to reconvert.

Change one setting at a time and repeat the smoke test with a fresh run ID.

## 3. Promote

Launch a fresh full run from the final smoke-tested config. Monitor it with the
same CLI loop until completion. Report the run ID, checkpoint, early-versus-
late reward, task-specific success rate, and whether all apps stopped.

Prefer a fresh full run. For Slime continuation, keep the Hugging Face model
path unchanged and use `recipe.load` rather than
`TrainConfig(checkpoint=...)`. When extending the saved training horizon, also
set:

```python
extra_config={"override_opt_param_scheduler": True}
```
