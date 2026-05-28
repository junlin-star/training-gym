"""Patch PyTorch's ``_write_item`` to avoid ``inline_container.cc`` crash.

The ``unexpected pos N vs M`` error in ``ostream_write_func`` occurs
because ``torch.save(tensor, stream)`` creates a ``PyTorchStreamWriter``
that tracks its own ``current_pos_`` starting from 0, but the underlying
file stream may be at a non-zero position from earlier BytesIO writes.
The miniz ``file_ofs`` and ``current_pos_`` can then diverge.

Fix: for tensor items, ``torch.save`` into an **isolated** ``BytesIO``
buffer first, then write the raw bytes to the file stream.  This avoids
any ``PyTorchStreamWriter`` being attached to the shared output stream.

Also seeks ``_mcore_to_torch_sharded_object`` return values to position 0
as a secondary defence.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re
import sys

patched_any = False

# ── 1. Patch _write_item in torch/distributed/checkpoint/filesystem.py ──────
#
# Replace the ``torch.save(data, stream)`` call for tensors with a
# two-step write: serialise into an isolated BytesIO, then copy bytes.

fs_py = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/torch/distributed/checkpoint/filesystem.py"
)
if fs_py.exists():
    fssrc = fs_py.read_text()
    MARKER_FS = "PATCHED_WRITE_ITEM_ISOLATED"
    if MARKER_FS not in fssrc:
        # Find the _write_item function and replace torch.save(data, ...) call
        # for tensors with an isolated BytesIO save.
        #
        # The function has this general pattern (both 2.5 and 2.7+):
        #   else:
        #       assert isinstance(data, torch.Tensor)  # or AssertionError variant
        #       ...
        #       torch.save(data, ...)
        #       ...close() # optional, 2.7+ only
        #
        # We match the torch.save line and replace it with the isolated version.
        # We handle both the 2.5 pattern (no transforms) and 2.7+ pattern
        # (with transforms / transform_to).

        # Try 2.7+ pattern first: torch.save(data, transform_to)
        m = re.search(
            r"^( +)(torch\.save\(data, transform_to\))\s*\n"
            r"\1(transform_to\.close\(\))",
            fssrc,
            re.MULTILINE,
        )
        if m:
            indent = m.group(1)
            old_block = m.group(0)
            new_block = (
                f"{indent}import io as _iso_io  # {MARKER_FS}\n"
                f"{indent}_iso_buf = _iso_io.BytesIO()\n"
                f"{indent}torch.save(data, _iso_buf)\n"
                f"{indent}transform_to.write(_iso_buf.getvalue())\n"
                f"{indent}transform_to.close()"
            )
            fssrc = fssrc.replace(old_block, new_block, 1)
            patched_any = True
            print("Patched filesystem.py _write_item (2.7+ pattern)")
        else:
            # Try 2.5 pattern: torch.save(data, cast(IO[bytes], stream))
            m2 = re.search(
                r"^( +)(torch\.save\(data, cast\(IO\[bytes\], stream\)\))",
                fssrc,
                re.MULTILINE,
            )
            if m2:
                indent = m2.group(1)
                old_line = m2.group(0)
                new_block = (
                    f"{indent}import io as _iso_io  # {MARKER_FS}\n"
                    f"{indent}_iso_buf = _iso_io.BytesIO()\n"
                    f"{indent}torch.save(data, _iso_buf)\n"
                    f"{indent}stream.write(_iso_buf.getvalue())"
                )
                fssrc = fssrc.replace(old_line, new_block, 1)
                patched_any = True
                print("Patched filesystem.py _write_item (2.5 pattern)")
            else:
                # Fallback: match any torch.save(data, ...) inside _write_item
                m3 = re.search(
                    r"^( +)(torch\.save\(data, \w+\))",
                    fssrc,
                    re.MULTILINE,
                )
                if m3:
                    indent = m3.group(1)
                    target_var = re.search(
                        r"torch\.save\(data, (\w+)\)", m3.group(2)
                    ).group(1)
                    old_line = m3.group(0)
                    new_block = (
                        f"{indent}import io as _iso_io  # {MARKER_FS}\n"
                        f"{indent}_iso_buf = _iso_io.BytesIO()\n"
                        f"{indent}torch.save(data, _iso_buf)\n"
                        f"{indent}{target_var}.write(_iso_buf.getvalue())"
                    )
                    fssrc = fssrc.replace(old_line, new_block, 1)
                    patched_any = True
                    print("Patched filesystem.py _write_item (fallback pattern)")
                else:
                    print(
                        "WARNING: could not find torch.save in _write_item",
                        file=sys.stderr,
                    )
        if patched_any:
            fs_py.write_text(fssrc)

# ── 2. Patch torch.py: seek(0) in _mcore_to_torch_sharded_object ────────────

torch_py = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py"
)
if torch_py.exists():
    src = torch_py.read_text()
    marker = "PATCHED_SEEK_SHOBJ"
    m = re.search(r"^( +)(return serialized_data)\b", src, re.MULTILINE)
    if m and marker not in src:
        indent = m.group(1)
        old_line = m.group(0)
        new_line = (
            f"{indent}serialized_data.seek(0)  # {marker}\n"
            f"{indent}return serialized_data"
        )
        src = src.replace(old_line, new_line, 1)
        torch_py.write_text(src)
        patched_any = True
        print("Patched torch.py: seek(0) in _mcore_to_torch_sharded_object")

if not patched_any:
    print("WARNING: No patches applied for checkpoint save", file=sys.stderr)
