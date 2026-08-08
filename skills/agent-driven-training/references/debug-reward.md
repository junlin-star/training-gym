# Debug reward

Use this reference when reward is flat, declining, saturated, unexpectedly
volatile, or improving suspiciously fast.

## Characterize the trajectory

```bash
training-gym run get <run-id> --verbose
```

Compare the baseline, early steps, recent steps, sample counts, and variance.
Do not infer learning from the final value alone.

- Flat near chance: the model may not be learning, the signal may be sparse,
  or extraction may return zero broadly.
- Declining: inspect regressions, truncation, instability, and whether the
  optimization target matches the task metric.
- Near the ceiling from the first rollout: the base model may already solve the
  task, the evaluation set may be too easy, or the reward may leak or overmatch.
- Abrupt jump to perfect reward: inspect for answer leakage, permissive parsing,
  duplicated samples, and reward hacking before treating it as success.

Ceiling thresholds are task-relative. Require enough steps and samples to
separate saturation from normal variance.

## Decide whether to stop an active run

Set an early efficacy checkpoint proportional to the run length; for example,
reassess a 150-step run across roughly steps 10–40. If enough samples show that
reward remains flat outside normal noise, declines, or is otherwise
uninformative, stop the run instead of waiting for completion. Do not stop on a
single noisy point, but do not keep a healthy yet ineffective job alive merely
because it has not failed.

Use `training-gym run get <run-id> --verbose` to obtain the Modal app ID, then
stop it:

```bash
modal app stop <app-id>
```

Diagnose the trajectory, change one parameter, and launch a fresh smoke test
before promoting again.

## Select and download traces

Estimate the export before downloading:

```bash
training-gym run trace <run-id> --out ./traces --dry-run --json
```

Select representative baseline, low-reward, transition, and recent steps:

```bash
training-gym run trace <run-id> --out ./traces --step 0,3,9 --yes
jq '.steps | sort_by(.mean_reward)' ./traces/<run-id>/manifest.json
```

`manifest.json` identifies each step file and records its sample count and mean
reward. Inspect structured samples by score:

```bash
jq '.samples | sort_by(.score) | .[:10]' \
  ./traces/<run-id>/step_0003.json
```

Prefer `jq` over text grep for selecting low-score records. Use text search
inside the selected records for repeated errors or suspicious response
patterns.

## Classify samples

Compare both low- and high-reward samples. Classify:

- correct answer scored incorrectly,
- incorrect or malformed answer scored as correct,
- prediction extracted from the prompt/reference instead of the response,
- response truncation or missing final answer,
- repeated tool-call or environment failures,
- parser mismatch with otherwise valid answers,
- repeated templates, copied references, or other reward-hacking behavior,
- infrastructure errors represented as task failures.

Recompute the reward locally for representative samples using the exact
prompt, response, and reference fields. Add fixture cases for every discovered
false positive or false negative.

## Decide the next experiment

- Reward implementation bug: fix it and rerun the one-step proof.
- Task is already saturated: make the task or evaluation more discriminative;
  do not launch a full horizon to produce an uninformative curve.
- Sparse but valid signal: adjust one reward or sampling choice and repeat the
  smoke test.
- Model behavior problem: change one training setting and compare fresh runs
  over equivalent steps and samples.

Never silently redefine the task metric merely to make the curve rise.
