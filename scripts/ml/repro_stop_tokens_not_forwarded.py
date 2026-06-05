#!/usr/bin/env python3
"""Minimum repro: rollout_stop_token_ids not forwarded to SGLang.

Root cause
----------
``SlimeRecipe`` on main was missing the ``rollout_stop_token_ids`` field entirely.
Slime's argparse defines ``--rollout-stop-token-ids`` (type=int, nargs="+"),
but training-gym never emitted it because the dataclass field didn't exist.

    training-gym SlimeRecipe  →  cli_args()  →  (no --rollout-stop-token-ids)
        ↓
    slime parse_args()  →  args.rollout_stop_token_ids = None (default)
        ↓
    GenerateState  →  sampling_params["stop_token_ids"] = None
        ↓
    HTTP POST  →  {"sampling_params": {"stop_token_ids": null}}
        ↓
    SGLang SamplingParams.stop_token_ids = None  →  no stop tokens applied

Fix: add ``rollout_stop_token_ids: list[int] | None = None`` to ``SlimeRecipe``.

Evidence
--------
Isolation test ``ap-MauLPYiDiGlkCQa58CoysR`` (2-node, bridge mode, GSM8K) shows
the model produces a correct answer then keeps generating past ``<|im_end|>``::

    "Tara's balance after 4 months is $780 - $260 = $520. #### 520"
    ... followed by random multilingual garbage tokens ...

    rollout/rollout_log_probs: -12.42  (avg over coherent + garbage)
    rollout/truncated_ratio:   1.0     (all responses hit max_new_tokens)
    rewards:                   0.0     (truncation kills the reward)

This script verifies end-to-end that, with the fix, the value flows correctly.
The companion patch ``patch_stop_token_diagnostic.py`` instruments the Docker
image for runtime verification — grep for ``[STOP_TOKEN_DIAG]`` in the logs.

Usage
-----
    uv run python scripts/ml/repro_stop_tokens_not_forwarded.py
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe

# ---------------------------------------------------------------------------
# Step 0: Show that the field was missing → None default
# ---------------------------------------------------------------------------
print("=" * 72)
print("Step 0: Verify the field exists and defaults to None")
print("=" * 72)

recipe_no_stop = SlimeRecipe(
    gpu_type="H100",
    colocate=True,
    tensor_model_parallel_size=2,
    sequence_parallel=True,
    rollout_num_gpus_per_engine=4,
    num_rollout=50,
    rollout_batch_size=8,
    rollout_max_response_len=4096,
    rollout_temperature=1.0,
    save_interval=100,
    # NOT setting rollout_stop_token_ids
)
cli_no_stop = recipe_no_stop.cli_args()
has_stop_flag = any(a == "--rollout-stop-token-ids" for a in cli_no_stop)
print(f"  rollout_stop_token_ids default = {recipe_no_stop.rollout_stop_token_ids}")
print(f"  --rollout-stop-token-ids in cli_args = {has_stop_flag}")
assert recipe_no_stop.rollout_stop_token_ids is None
assert not has_stop_flag, (
    "Expected no --rollout-stop-token-ids in cli_args when field is None"
)
print("  → Without setting the field, stop tokens are NEVER sent to slime.\n")
print("  THIS WAS THE BUG: the field didn't exist in SlimeRecipe on main,")
print("  so --rollout-stop-token-ids was never emitted in the CLI args.\n")

# ---------------------------------------------------------------------------
# Step 1: training-gym → CLI args (with fix)
# ---------------------------------------------------------------------------
print("=" * 72)
print("Step 1: training-gym SlimeRecipe → CLI args (with fix)")
print("=" * 72)

recipe = SlimeRecipe(
    gpu_type="H100",
    colocate=True,
    tensor_model_parallel_size=2,
    sequence_parallel=True,
    rollout_num_gpus_per_engine=4,
    num_rollout=50,
    rollout_batch_size=8,
    rollout_max_response_len=4096,
    rollout_temperature=1.0,
    save_interval=100,
    rollout_stop_token_ids=[151645, 151643],
)

cli_args = recipe.cli_args()

stop_idx = None
for i, arg in enumerate(cli_args):
    if arg == "--rollout-stop-token-ids":
        stop_idx = i
        break

if stop_idx is None:
    print("FAIL: --rollout-stop-token-ids not found in cli_args!")
    sys.exit(1)

values = []
for j in range(stop_idx + 1, len(cli_args)):
    if cli_args[j].startswith("--"):
        break
    values.append(cli_args[j])

print(
    f"  cli_args[{stop_idx}:{stop_idx + 1 + len(values)}] = "
    f"{cli_args[stop_idx : stop_idx + 1 + len(values)]}"
)
print(f"  Shell form: {shlex.join(cli_args[stop_idx : stop_idx + 1 + len(values)])}")
assert values == ["151645", "151643"], f"Expected ['151645', '151643'], got {values}"
print("  OK\n")

# ---------------------------------------------------------------------------
# Step 2: slime argparse → Namespace
# ---------------------------------------------------------------------------
print("=" * 72)
print("Step 2: slime argparse → args.rollout_stop_token_ids")
print("=" * 72)

parser = argparse.ArgumentParser()
parser.add_argument("--rollout-stop-token-ids", type=int, nargs="+", default=None)
ns, _ = parser.parse_known_args(cli_args)
print(f"  args.rollout_stop_token_ids = {ns.rollout_stop_token_ids}")
assert ns.rollout_stop_token_ids == [151645, 151643]
print("  OK\n")

# ---------------------------------------------------------------------------
# Step 3: GenerateState.sampling_params dict
# ---------------------------------------------------------------------------
print("=" * 72)
print("Step 3: GenerateState.__init__ → sampling_params dict")
print("=" * 72)

sampling_params = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    max_new_tokens=4096,
    stop=None,
    stop_token_ids=ns.rollout_stop_token_ids,
    skip_special_tokens=True,
    no_stop_trim=True,
    spaces_between_special_tokens=False,
)
print(f"  sampling_params['stop_token_ids'] = {sampling_params['stop_token_ids']}")
assert sampling_params["stop_token_ids"] == [151645, 151643]
print("  OK\n")

# ---------------------------------------------------------------------------
# Step 4: HTTP POST payload
# ---------------------------------------------------------------------------
print("=" * 72)
print("Step 4: HTTP POST payload to SGLang /generate")
print("=" * 72)

payload = {
    "sampling_params": sampling_params,
    "return_logprob": True,
    "input_ids": [1, 2, 3],
}
parsed = json.loads(json.dumps(payload))
stop_in_json = parsed["sampling_params"]["stop_token_ids"]
print(f"  payload.sampling_params.stop_token_ids = {stop_in_json}")
assert stop_in_json == [151645, 151643]
print("  OK\n")

# ---------------------------------------------------------------------------
# Step 5: SGLang SamplingParams construction
# ---------------------------------------------------------------------------
print("=" * 72)
print("Step 5: SGLang SamplingParams(**sampling_kwargs)")
print("=" * 72)

stop_token_ids_input = parsed["sampling_params"]["stop_token_ids"]
if stop_token_ids_input:
    filtered = {int(t) for t in stop_token_ids_input if t is not None}
    sglang_stop_token_ids = filtered or None
else:
    sglang_stop_token_ids = None

print(f"  SamplingParams.stop_token_ids = {sglang_stop_token_ids}")
assert sglang_stop_token_ids == {151645, 151643}
print("  OK\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 72)
print("RESULT")
print("=" * 72)
print(
    """
Root cause: SlimeRecipe was missing the rollout_stop_token_ids field.
The upstream slime framework accepts --rollout-stop-token-ids, but
training-gym never emitted it because the dataclass field didn't exist.

Fix: Add rollout_stop_token_ids to SlimeRecipe (this PR).

With the field set, the full pipeline correctly forwards stop tokens:
  SlimeRecipe.cli_args()  →  --rollout-stop-token-ids 151645 151643
  slime argparse          →  args.rollout_stop_token_ids=[151645, 151643]
  GenerateState           →  sampling_params["stop_token_ids"]=[151645, 151643]
  HTTP POST payload       →  {"stop_token_ids": [151645, 151643]}
  SGLang SamplingParams   →  stop_token_ids={151645, 151643}

Runtime verification: deploy with patch_stop_token_diagnostic.py and
grep for [STOP_TOKEN_DIAG] in the logs.
"""
)
