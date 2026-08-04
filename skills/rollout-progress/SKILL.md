---
name: rollout-progress
description: This skill should be used when the user asks to "check the progress of a run", "what step is run X on", "is my training run stuck", "what is run <id> doing", "grab the app logs for a run", "how far along is the rollout", or otherwise wants to know the live status, current step, current phase, or blocker of a modal-training-gym training run given a run id (or partial id / wandb id / Modal app id).
version: 0.1.0
---

# Rollout Progress

Determine where a modal-training-gym training run is: its current phase, which
training step / rollout it is on, whether it is alive or stuck, and what the
recent Modal app logs show it doing. Use this to answer "how far along is run
X?", "what is it stuck on?", and "what step is it on?".

## How a run is tracked (background)

A training run is keyed by a stable `training_run_id` (e.g.
`elastic-frog-a1b2c3d4e5f6`), which is also the Modal **app name**. Progress is
recorded two ways:

- A `TrainingRun` record in the `training-gym-metadata` Modal Volume holds the
  current `framework_status` (phase), `status` (running/completed/failed),
  `metadata.framework_progress` (`current`/`total` step), the `modal_app_id`,
  timings, and `updated_at`.
- `TrainingRolloutResult` summaries record each completed rollout step
  (`rollout_id`, sample count, mean reward, rollout time).

The Modal **app logs** (`modal app logs <app_id>`) show the live, line-by-line
activity — this is where a hang or error is visible.

## Workflow

Run the helper from the repo root so it uses the project venv and metadata
access:

```bash
uv run .claude/skills/rollout-progress/scripts/run_progress.py <run_id> --tail 200
```

`<run_id>` may be the full id, a prefix, the 8-char wandb id, a substring, or
the Modal app id — the script fuzzy-matches against the run summary list and
lists candidates if ambiguous.

The script prints, in order:

1. **Progress block** — framework, run status, current `phase`, `step
   current/total`, app id/url, created/started/last-update times, per-phase
   timings, app liveness, and a `⚠ POSSIBLY STUCK` hint if the app is live but
   metadata has not updated in >10 min.
2. **Rollouts** — count recorded, latest `rollout_id` (= the latest completed
   step), and the last 5 rollouts' sample count / mean reward / time.
3. **Logs** — tails `modal app logs <app_id>` (last 200 lines by default).

Useful flags: `--tail N` (more/fewer log lines), `--no-logs` (metadata only,
fast), `--follow`/`-f` (live-stream logs instead of tailing).

## Interpreting the result

Answer the user with: **what step** it is on (the `step current/total` line,
corroborated by the latest `rollout_id`), **what phase** it is currently in
(the `phase` field), and **what it is doing / stuck on** (read from the logs
tail).

The `phase` values for the slime (RL) framework, in normal cycle order, are:

| phase | meaning |
|---|---|
| `initializing` / `download_model` / `convert_model` / `prepare_dataset` | one-time setup before the training loop |
| `initialize_rollouts` | spinning up the rollout/inference engines |
| `generate_rollouts` | generating samples (prompts → model responses) for this step |
| `evaluate_rollouts` | running the eval rollout set |
| `compute_log_probs` | computing log-probs over the generated samples |
| `optimizer_step` | the GRPO weight update for this step |
| `weight_sync` | syncing updated weights to the rollout engines |
| `offload_rollout` / `offload_train` | moving model state between GPUs |
| `checkpoint_save` | writing a checkpoint |

(The `miles` framework only reports `initializing`/`download_model`/`convert_model`/`prepare_dataset`/`training`.)

Diagnose "stuck on" by combining signals:

- **Phase + stale `updated_at`** → stuck within that phase. E.g. live app stuck
  in `generate_rollouts` with no update for 20m usually means rollout
  generation is hanging (slow/long generations, a wedged inference engine, or a
  reward/rm endpoint returning 401 — see the proxy-auth note in CLAUDE.md).
- **Log tail** is the ground truth for the blocker. Scan it for: tracebacks /
  exceptions, NCCL or CUDA OOM errors, `401`/auth failures from reward or
  teacher endpoints, repeated retry lines, or a long quiet gap after the last
  meaningful line. Report the most recent error or the last activity line.
- **App not live** while `status == running` → the app died; the last log lines
  hold the cause.
- **No app id yet** → the app has not started; the run is still queuing or in
  early setup. There are no logs to fetch.

When the user wants to watch a live run progress, re-run with `--follow`, or
re-run the metadata-only view (`--no-logs`) periodically to see the step/phase
advance.
