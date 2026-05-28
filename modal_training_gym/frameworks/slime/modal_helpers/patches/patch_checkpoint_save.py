"""Patch Megatron/PyTorch checkpoint save for hybrid MoE models.

The ``inline_container.cc`` "unexpected pos" error occurs when PyTorch's
zip writer tracks a file position that diverges from what miniz expects.
Root cause: ``_mcore_to_torch_sharded_object`` in Megatron's ``torch.py``
returns ``BytesIO`` objects with the cursor at the end.  Downstream code
(plan size calculation, ``getbuffer()``) *usually* tolerates this, but
certain PyTorch versions and write-path combinations do not.

This patch applies two fixes:
1. Seek every ``BytesIO`` returned by ``_mcore_to_torch_sharded_object``
   back to position 0 so all consumers see consistent data.
2. Wrap ``MCoreSavePlanner.transform_object`` to seek(0) any ``BytesIO``
   before the planner hands it to the writer, as a defence-in-depth
   measure.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

# ── 1. Patch torch.py: seek(0) in _mcore_to_torch_sharded_object ────────────

torch_py = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py"
)
src = torch_py.read_text()
patched = False

# The function creates a BytesIO via torch.save() and returns it with
# the cursor at end-of-stream.  Add a seek(0) before the return.
old_shobj = "return serialized_data"
marker = "PATCHED_SEEK_SHOBJ"
if old_shobj in src and marker not in src:
    new_shobj = f"serialized_data.seek(0)  # {marker}\n    return serialized_data"
    src = src.replace(old_shobj, new_shobj, 1)
    patched = True

# ── 2. Patch MCoreSavePlanner.transform_object to seek(0) BytesIO ───────────
# Handle both NVIDIA Megatron (has docstring) and THUDM slime fork (bare return)

marker2 = "PATCHED_TRANSFORM_SEEK"
if marker2 not in src:
    # NVIDIA Megatron version with docstring
    old_transform_nvidia = (
        '"""Make no transformations - bytes objects are already serialized."""\n'
        "        return object"
    )
    # THUDM slime fork version — no docstring, inline
    old_transform_slime = (
        "def transform_object(self, write_item: WriteItem, object: Any): return object"
    )
    seek_body = (
        "def transform_object(self, write_item: WriteItem, object: Any):\n"
        "        import io as _io  # " + marker2 + "\n"
        "        if isinstance(object, _io.BytesIO):\n"
        "            object.seek(0)\n"
        "        return object"
    )
    if old_transform_nvidia in src:
        new_transform = (
            '"""Make no transformations - bytes objects are already serialized."""\n'
            "        import io as _io  # " + marker2 + "\n"
            "        if isinstance(object, _io.BytesIO):\n"
            "            object.seek(0)\n"
            "        return object"
        )
        src = src.replace(old_transform_nvidia, new_transform, 1)
        patched = True
    elif old_transform_slime in src:
        src = src.replace(old_transform_slime, seek_body, 1)
        patched = True

if patched:
    torch_py.write_text(src)
    print("Patched torch.py for checkpoint save BytesIO handling")
else:
    print("WARNING: Could not patch torch.py — target strings not found")

# ── 3. Patch filesystem_async.py: seek(0) BytesIO before writing ────────────

fs_async = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
)
if fs_async.exists():
    fs_src = fs_async.read_text()
    # In prepare_write_data, BytesIO objects are resolved via
    # planner.resolve_data().  Ensure they are seeked to 0.
    old_bytes_resolve = (
        "planner.resolve_data(item)) "
        "for item in bucket if item.type == WriteItemType.BYTE_IO"
    )
    marker3 = "PATCHED_BYTES_SEEK"
    if old_bytes_resolve in fs_src and marker3 not in fs_src:
        # Wrap the resolve call to seek(0) the result
        new_bytes_resolve = (
            "(lambda _b: (_b.seek(0), _b)[1])(planner.resolve_data(item))) "
            "for item in bucket if item.type == WriteItemType.BYTE_IO"
            "  # " + marker3
        )
        fs_src = fs_src.replace(old_bytes_resolve, new_bytes_resolve, 1)
        fs_async.write_text(fs_src)
        print("Patched filesystem_async.py for BytesIO seek(0)")
    else:
        print("INFO: filesystem_async.py — target string not found or already patched")
