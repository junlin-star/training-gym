"""Return bare logits from Gemma4-VL's forward so slime's loss function can use them.

``Gemma4VLModel.forward`` ends with ``return (outputs, loss_mask)``, but slime's
training step passes whatever the model returns straight to ``loss_function`` as
``logits``, so a tuple arrives where a tensor is expected and
``policy_loss_function`` raises ``AttributeError: 'tuple' object has no attribute
'dtype'``.

Dropping the second element loses nothing: slime passes the loss mask *into* the model
and reads its own copy from ``batch``, so the returned one is redundant. At
``context_parallel_size > 1`` it would be the CP-sliced mask and slime would have to
consume it instead, which is part of why the recipe pins cp=1. Report upstream.

Idempotent. Run at image build:  python patch_gemma4_vl_forward_return.py
"""

import pathlib

MARKER = "PATCHED_GEMMA4_VL_FORWARD_RETURN"

OLD_RETURN = "        return (outputs, loss_mask)"
NEW_RETURN = f"""        # {MARKER}: slime's forward_step feeds this value directly to
        # loss_function as `logits`, and keeps its own loss mask in the batch, so
        # return the bare tensor instead of megatron-bridge's (logits, loss_mask).
        return outputs"""


def main() -> None:
    targets = list(
        pathlib.Path("/usr/local/lib").glob(
            "python3.*/dist-packages/megatron/bridge/models/gemma_vl/modeling_gemma4_vl.py"
        )
    )
    if not targets:
        print("WARNING: modeling_gemma4_vl.py not found; skipping forward-return patch")
        return

    for path in targets:
        src = path.read_text()
        if MARKER in src:
            print("compat: gemma4_vl forward return already unwrapped:", path)
            continue
        if OLD_RETURN not in src:
            print(
                "WARNING: could not find '(outputs, loss_mask)' return in",
                path,
                "(upstream may have changed; forward-return patch NOT applied)",
            )
            continue
        path.write_text(src.replace(OLD_RETURN, NEW_RETURN, 1))
        print("compat: unwrapped gemma4_vl forward return:", path)


if __name__ == "__main__":
    main()
