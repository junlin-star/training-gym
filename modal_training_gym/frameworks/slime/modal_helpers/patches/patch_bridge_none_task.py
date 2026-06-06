"""Patch Megatron bridge to skip None conversion tasks.

When using bridge mode for text-only MoE models (e.g. Qwen3.6-35B-A3B),
the Qwen35VLMoEBridge generates conversion tasks for vision-model parameters
that don't exist in the text-only variant.  These tasks are None, which
crashes at ``task.param_weight`` with an ``AttributeError``.

This patch adds None checks to skip such tasks gracefully, handling ALL
matching occurrences in the file via re.sub.

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

        def _add_none_guard(m: re.Match) -> str:
            indent = m.group(1)
            original = m.group(0)
            return (
                f"{indent}if task is None:  # {marker}\n"
                f"{indent}    continue\n"
                f"{original}"
            )

        new_src, count = re.subn(
            r"^( +)(if isinstance\(task\.param_weight, DTensor\):)",
            _add_none_guard,
            src,
            flags=re.MULTILINE,
        )
        new_src, load_count = re.subn(
            r"^( +)(if task\.megatron_module is None:)",
            _add_none_guard,
            new_src,
            flags=re.MULTILINE,
        )
        count += load_count
        if count > 0:
            p.write_text(new_src)
            print(
                f"Patched model_bridge.py: added None task guard at {count} location(s)"
            )
        else:
            print("WARNING: Could not find target string in model_bridge.py")
