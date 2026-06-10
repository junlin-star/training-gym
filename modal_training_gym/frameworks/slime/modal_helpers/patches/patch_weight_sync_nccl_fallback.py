"""Patch slime's weight updater to fall back to NCCL when CUDA IPC is unavailable.

When ``colocate=True``, slime selects ``UpdateWeightFromTensor`` which uses CUDA
IPC (``torch.cuda.ipc_get_device_handle`` / ``MultiprocessingSerializer``) to
share GPU memory between the Megatron training process and SGLang rollout engines.

CUDA IPC requires ``CAP_IPC_LOCK`` (Linux capability for memory pinning), which
is only available in containers started with ``clustered(rdma=True)``.  Without
this capability, the IPC path silently falls back to a CPU-mediated copy that is
~100x slower (~30s/it vs ~0.25s/it for 132 weight updates).

This patch modifies ``UpdateWeightFromTensor.connect_rollout_engines`` to detect
whether ``CAP_IPC_LOCK`` is available.  If not, it sets ``colocate_engine_nums=0``
so all engines are routed through the NCCL broadcast path
(``update_weights_from_distributed``), which achieves full NVLink speed without
requiring any special capabilities or device passthrough.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

p = pathlib.Path(
    "/root/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py"
)
if not p.exists():
    print("WARNING: update_weight_from_tensor.py not found, skipping patch")
else:
    src = p.read_text()
    marker = "PATCHED_NCCL_FALLBACK"
    if marker in src:
        print("update_weight_from_tensor.py already patched for NCCL fallback")
    else:
        target = "self.use_distribute = len(rollout_engines) > colocate_engine_nums"
        if target not in src:
            print(
                "WARNING: Could not find target pattern in update_weight_from_tensor.py"
            )
        else:
            # 1. Insert _has_ipc_lock helper before the class definition.
            helper_fn = '''
def _has_ipc_lock():  # {marker}
    """Check if CAP_IPC_LOCK (bit 14) is in the effective capability set."""
    import os
    if os.environ.get("SLIME_FORCE_NCCL_WEIGHT_SYNC"):
        return False
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    cap_hex = int(line.split(":")[1].strip(), 16)
                    return bool(cap_hex & (1 << 14))  # CAP_IPC_LOCK
    except OSError:
        pass
    return True  # assume available if can't check

'''.replace("{marker}", marker)

            # Find the class definition and insert the helper before it.
            class_match = re.search(r"^class UpdateWeightFromTensor", src, re.MULTILINE)
            if class_match:
                insert_pos = class_match.start()
                src = src[:insert_pos] + helper_fn + src[insert_pos:]
            else:
                print("WARNING: Could not find class UpdateWeightFromTensor")

            # 2. Insert the CAP_IPC_LOCK check before self.use_distribute.
            # Detect indentation of the target line.
            indent_match = re.search(r"^( +)" + re.escape(target), src, re.MULTILINE)
            if indent_match:
                indent = indent_match.group(1)
                replacement = (
                    f"{indent}# {marker}: Fall back to NCCL when CUDA IPC is unavailable.\n"
                    f"{indent}if colocate_engine_nums > 0 and not _has_ipc_lock():\n"
                    f"{indent}    colocate_engine_nums = 0\n"
                    f"\n"
                    f"{indent}{target}"
                )
                # Replace only the indented occurrence (inside the method).
                src = re.sub(
                    r"^( +)" + re.escape(target),
                    replacement,
                    src,
                    count=1,
                    flags=re.MULTILINE,
                )

            p.write_text(src)
            print(
                "Patched update_weight_from_tensor.py: "
                "added NCCL fallback when CAP_IPC_LOCK unavailable"
            )
