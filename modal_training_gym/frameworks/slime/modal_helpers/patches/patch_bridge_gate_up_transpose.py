"""Patch qwen35_vl_bridge.py to add transpose_on_export=True on gate_up_proj.

With EP > 1, TransformerEngine stores expert weights in [in, out] layout
(transposed from standard PyTorch [out, in]).  The bridge's grouped-export
accumulator auto-detects the mismatch and transposes, but ONLY when the
mapping has ``transpose_on_export=True``.

In the current Megatron-Bridge ``qwen35_vl_bridge.py``:
- ``down_proj``  (FusedExpertMapping)       → has ``transpose_on_export=True``  ✓
- ``gate_up_proj`` (FusedGatedExpertMapping) → MISSING the flag               ✗

The sibling ``qwen3_vl_bridge.py`` was fixed (commit ad27e2c4) but
``qwen35_vl_bridge.py`` was not updated, causing corrupted expert weights
and gibberish model outputs.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py"
)
if not p.exists():
    print("WARNING: qwen35_vl_bridge.py not found, skipping patch")
else:
    src = p.read_text()
    marker = "PATCHED_GATE_UP_TRANSPOSE"

    if marker in src:
        print("qwen35_vl_bridge.py already patched for gate_up_proj transpose")
    else:
        OLD = (
            "FusedGatedExpertMapping(\n"
            '                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",\n'
            '                    hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",\n'
            "                )"
        )
        NEW = (
            "FusedGatedExpertMapping(  # {marker}\n"
            '                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",\n'
            '                    hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",\n'
            "                    transpose_on_export=True,\n"
            "                )"
        ).format(marker=marker)

        if OLD in src:
            new_src = src.replace(OLD, NEW, 1)
            p.write_text(new_src)
            print(
                "Patched qwen35_vl_bridge.py: added transpose_on_export=True to gate_up_proj FusedGatedExpertMapping"
            )
        else:
            print(
                "WARNING: Could not find target FusedGatedExpertMapping string in qwen35_vl_bridge.py"
            )
