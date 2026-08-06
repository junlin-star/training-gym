"""Send Gemma-4 VL prompts to SGLang as text so SGLang does the image expansion.

``miles/rollout/sglang_rollout.py: generate`` runs the HF processor locally and
sends the resulting ``input_ids`` to SGLang. For Gemma-4 those ids already carry
one ``<|image|>`` token per vision patch, and SGLang re-validates them against
the raw image it was handed, so every request 400s::

    The total number of <|image|> tokens in the prompts should be the same as the
    number of images passed. Found [252] <|image|> tokens and [1] images per sample.

Turning the chat template off is not an escape: ``call_processor`` then receives a
list of message dicts and ``Gemma4Processor.validate_inputs`` does ``sample.count``
on a dict (``'dict' object has no attribute 'count'``). Both settings fail, so the
prompt has to reach SGLang unexpanded instead.

slime solves this by sending ``text`` rather than ``input_ids`` whenever a
single-turn request carries images, with the comment "so SGLang expands the image
placeholders with its own processor rules"
(``slime/rollout/sglang_rollout.py``). This mirrors that branch.

**Scoped to Gemma-4 on purpose.** The condition is general -- every image model
sets ``image_data`` -- but miles and slime disagree by design about who expands
media placeholders. ``miles/backends/training_utils/mm_data.py`` states miles'
contract: "The rollout side emits one media placeholder/sentinel per media item;
training expands it." Kimi-VL/K2.5 and Inkling rely on that, and Qwen3-VL already
works on the ``input_ids`` path. Flipping them all is an upstream design decision,
not ours, so the branch tests the processor class and every other model keeps the
original code path byte for byte.

Gemma-4 needs no training-side expansion in return: ``_collect_multimodal_grid_inputs``
keys on ``"grid_thws"``, which Gemma-4's processor does not emit, so
``expand_multimodal_rollout_data_in_place`` returns early -- the same no-op slime
has by never expanding at all.

Report upstream; miles has no Gemma-4 VL support and none is planned (roadmap #797
lists "gemma-4" and "VL series" as separate, and VL series means Qwen3-VL).

Idempotent. Run at image build:  python patch_gemma4_vl_rollout_text.py
"""

import pathlib

MARKER = "PATCHED_GEMMA4_VL_ROLLOUT_TEXT"

TARGET = pathlib.Path("/root/miles/miles/rollout/sglang_rollout.py")

OLD = """    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids
"""

NEW = f"""    # Use existing tokens for multi-turn or tokenize the new prompt
    if (
        payload.get("image_data")
        and len(sample.response) == 0
        and type(getattr(state, "processor", None)).__name__.startswith("Gemma4")
    ):
        # {MARKER}: Gemma-4 only. Its processor pre-expands <|image|> per patch and
        # SGLang rejects that against the single raw image, so hand SGLang the text
        # and let it expand -- what slime does for every multimodal request. Other
        # models fall through to the original branches unchanged, because miles
        # expects them to keep one placeholder for mm_data.py to expand later.
        payload["text"] = sample.prompt
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids
    elif len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids
"""

if not TARGET.exists():
    print(f"{TARGET} not found; skipping Gemma-4 VL rollout text patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("Gemma-4 VL rollout text patch already applied")
    raise SystemExit(0)

if OLD not in src:
    raise SystemExit(
        "Gemma-4 VL rollout text patch did not match; miles' sglang_rollout.py "
        "payload construction has changed. Re-check it before shipping."
    )

# The injected branch reads `state.processor`, a local of the enclosing
# generate(). Matching OLD only proves the *surrounding* lines are unchanged, so
# check the binding too: if upstream renames it, the patch would still apply and
# the branch would just never fire -- sending pre-expanded input_ids again and
# putting VL back to silently failing on every request.
if "state = GenerateState(args)" not in src:
    raise SystemExit(
        "Gemma-4 VL rollout text patch expects a local `state = GenerateState(args)` "
        "in miles' sglang_rollout.py; it is gone, so the injected branch would never "
        "fire. Re-check how the processor is reached before shipping."
    )

TARGET.write_text(src.replace(OLD, NEW, 1))
print("Patched Gemma-4 VL rollout to send text instead of pre-expanded input_ids")
