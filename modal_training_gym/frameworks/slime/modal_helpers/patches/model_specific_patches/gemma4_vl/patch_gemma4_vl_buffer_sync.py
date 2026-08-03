"""Let bridge-mode weight sync carry a model's registered buffers, not just expert bias.

Both collectors in slime's ``update_weight/common.py`` drop every buffer except one,
under an upstream ``TODO shall we handle (almost) all buffers like Megatron Bridge``.
Gemma-4 cannot survive that: its checkpoint stores 33 trained buffers — one
``layer_scalar`` per decoder layer, the vision tower's ``std_bias`` / ``std_scale``,
and ``embed_scale`` — which megatron-bridge enumerates as conversion tasks, so the
sync asserts ``weight_dict_key='vp_stages.0.vision_tower.std_bias' not in
new_weight_dict``.

Skipping the unsupplied tasks instead is worse than the crash: SGLang loads dummy
weights and takes everything from this sync, so a skipped buffer silently keeps its
default init. Doing that left all 30 ``layer_scalar`` values wrong and the model
generating confident garbage that reads as a bad model rather than a bad transfer.

Supplying them is additive: ``megatron_local_weights`` is only looked up by
``_process_conversion_tasks``, never iterated to generate work, and buffers are
identical across ranks, so no other model or collective is affected. The upstream
assertion stays as the tripwire for the next unsupplied tensor. Report upstream.

Idempotent. Run at image build:  python patch_gemma4_vl_buffer_sync.py
"""

import pathlib

MARKER = "PATCHED_GEMMA4_VL_BUFFER_SYNC"

TARGET = pathlib.Path(
    "/root/slime/slime/backends/megatron_utils/update_weight/common.py"
)

OLD_FILTER = """        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
"""

NEW_FILTER = f"""        for name, buffer in model_module.named_buffers():
            # {MARKER}: yield every buffer, not just expert_bias. Gemma-4 stores 33
            # trained buffers (per-layer layer_scalar, vision std_bias/std_scale,
            # embed_scale); withholding them leaves SGLang on its default init.
"""


def main() -> None:
    if not TARGET.exists():
        print("WARNING: update_weight/common.py not found; skipping buffer-sync patch")
        return

    src = TARGET.read_text()
    if MARKER in src:
        print("compat: update_weight/common.py already yields all buffers")
        return

    # Two collectors share the same filter: _named_params_and_buffers_vanilla (used by
    # bridge-mode weight sync) and _named_params_and_buffers_global (used by the HF
    # checkpoint saver). Both need it removed.
    count = src.count(OLD_FILTER)
    if count == 0:
        print(
            "WARNING: could not find the expert_bias buffer filter in "
            "update_weight/common.py (upstream may have changed; buffer-sync patch "
            "NOT applied)"
        )
        return

    TARGET.write_text(src.replace(OLD_FILTER, NEW_FILTER))
    print(f"compat: patched {count} buffer filter(s) in update_weight/common.py")


if __name__ == "__main__":
    main()
