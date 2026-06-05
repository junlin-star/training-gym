"""Patch slime's Qwen3.5 HF conversion for bridge checkpoint names.

The Qwen3.6-35B-A3B bridge checkpoint stores Megatron keys below a
``language_model.`` namespace, while slime's Qwen3.5 converter expects the
namespace-less Megatron names. It can also carry unused vision placeholders from
the VLM bridge. Normalize these names before conversion.

Executed at image-build time via ``python3 <this file>``.
"""

from pathlib import Path

TARGET = Path("/root/slime/slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py")
MARKER = "PATCHED_QWEN35_BRIDGE_NAMES"

if not TARGET.exists():
    print(f"WARNING: {TARGET} not found, skipping Qwen3.5 conversion-name patch")
else:
    src = TARGET.read_text()
    needle = (
        '    """Convert Qwen3.5 model parameters from Megatron to HuggingFace format.'
    )
    if MARKER in src:
        print("qwen3_5.py already patched for bridge checkpoint names")
    elif needle in src:
        insert = (
            f"    # {MARKER}\n"
            '    if name.startswith("module.module.language_model."):\n'
            '        name = "module.module." + name[len("module.module.language_model.") :]\n'
            '    if name.startswith("module.module.vision_model."):\n'
            "        return []\n"
            "\n"
        )
        src = src.replace(needle, insert + needle, 1)
        TARGET.write_text(src)
        print(f"Patched {TARGET}: normalized Qwen3.5 bridge names")
    else:
        print("WARNING: Could not find Qwen3.5 converter insertion point")
