"""Patch Megatron bridge to skip None tasks in stream_weights_megatron_to_hf.

When using bridge mode for text-only MoE models (e.g. Qwen3.6-35B-A3B),
the Qwen35VLMoEBridge generates conversion tasks for vision-model parameters
that don't exist in the text-only variant.  These tasks are None, which
crashes at ``task.param_weight`` with an ``AttributeError``.

This patch adds a None check to skip such tasks gracefully.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/megatron/bridge/models/conversion/model_bridge.py"
)
if not p.exists():
    print("WARNING: model_bridge.py not found, skipping patch")
else:
    src = p.read_text()
    marker = "PATCHED_NONE_TASK_CHECK"
    if marker in src:
        print("model_bridge.py already patched for None task check")
    else:
        # Use regex to detect actual indentation of the target line.
        m = re.search(
            r"^( +)(if isinstance\(task\.param_weight, DTensor\):)",
            src,
            re.MULTILINE,
        )
        if m:
            indent = m.group(1)
            old_line = m.group(0)
            new_line = (
                f"{indent}if task is None:  # {marker}\n"
                f"{indent}    continue\n"
                f"{old_line}"
            )
            src = src.replace(old_line, new_line, 1)
            p.write_text(src)
            print(
                "Patched model_bridge.py to skip None tasks in stream_weights_megatron_to_hf"
            )
        else:
            print("WARNING: Could not find target string in model_bridge.py")
