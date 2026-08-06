# Continue a training run

Use this reference before extending training from an existing checkpoint.

## Decide whether to continue

Prefer a fresh full run from the final smoke-tested config. Continue only when
the checkpoint is healthy and preserving optimizer/training state is useful.
Do not continue a run whose reward is saturated, corrupted, or based on a
broken reward function.

Inspect:

```bash
training-gym run get <run-id> --verbose
training-gym run params <run-id>
```

Record the source checkpoint, model identity, recipe, completed horizon,
topology, scheduler settings, and the reason for continuation.

## Slime continuation

Keep the original Hugging Face model path unchanged and set `recipe.load` to
the saved training checkpoint. Do not use `TrainConfig(checkpoint=...)` as a
substitute for Slime's load field.

When extending beyond the optimizer scheduler's saved horizon, set:

```python
extra_config={"override_opt_param_scheduler": True}
```

Preserve compatible model architecture and parallelism. If changing topology
or conversion layout is necessary, validate compatibility with a one-step
continuation before committing to the extended horizon.

## Validate

Launch continuation as a fresh run ID. Prove one step and verify:

- the intended checkpoint was loaded,
- the resumed step/horizon is correct,
- optimizer and scheduler behavior are expected,
- reward and loss are finite and plausible, and
- no conversion or state-shape error appears.

Then use the normal smoke and promotion gates. Report both the source run ID
and continuation run ID.
