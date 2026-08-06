# Failure signatures

Use this reference when a Training Gym run fails or appears hung.

## Collect evidence

Start with the supported CLI:

```bash
training-gym run get <run-id> --verbose
training-gym run params <run-id>
training-gym run logs <run-id> --tail 200
```

Record the run status, phase, current/total step, last update, app ID, first
actionable error, and last meaningful log line. Fetch more logs with `--since`
or `--search` when the tail omits the beginning of a traceback.

## Match the symptom

| Evidence | Likely cause | Next action |
|---|---|---|
| `ValidationError`, `extra_forbidden`, unknown field, or invalid parallelism | Config or recipe does not match the public schema | Inspect `run params` and the current class definition; correct the invalid field or topology locally |
| `KeyError`, `FileNotFoundError`, missing column, parse exception, or zero prepared rows | Dataset preparation or formatting bug | Reproduce with representative local rows; inspect prepared paths and configured input/output keys |
| CUDA out-of-memory or allocation failure | Batch, token, model, or parallelism layout exceeds memory | Identify whether rollout or training OOMed; reduce the relevant batch/token load or change a justified parallelism setting |
| NCCL timeout, Ray actor death, worker disconnect, or repeated placement retries | Distributed runtime, node, or capacity problem | Check whether tasks are active and whether one rank failed first; retry transient placement failures without changing training semantics |
| HTTP 401 from a reward or teacher endpoint | An explicitly protected served endpoint lacks proxy credentials | Verify that the endpoint is intentionally protected and forward credentials to the remote worker; the Training Gym dashboard itself is open by default |
| App is not live while run status remains `running` | App exited before metadata finalized | Use the final logs as ground truth and report the stale metadata separately |

Do not apply a table fix mechanically. Confirm that the cited evidence matches
the failing component.

## Distinguish slow from hung

`run logs --follow` is useful only over a bounded observation window. Log
silence alone does not prove a hang: scheduling, image build, model conversion,
long GPU kernels, and evaluation can be quiet.

Corroborate at least two signals:

- phase and `updated_at` remain unchanged beyond the expected duration,
- no new meaningful logs arrive,
- the Modal app has no active task or its workers have exited, or
- logs show repeated retries with no completed work.

On a final step matching `eval_interval`, Slime can spend many minutes in
`evaluate_rollouts` or appear stale in `weight_sync`. Live SGLang generation
traffic plus an active Modal task indicates a slow held-out eval, not a hang.

## Cleanup and relaunch

There is currently no `training-gym run kill` command. If the authorized task
includes stopping or fixing the run, obtain the Modal app ID from `run get`,
then:

```bash
modal app stop <app-id>
modal app list --json
```

Confirm the old app stopped before relaunching against shared volumes. Preserve
the evidence, change one setting at a time, and use a fresh run ID.

For a diagnosis-only or status-only request, report the cause and stop; do not
kill or relaunch without authorization.
