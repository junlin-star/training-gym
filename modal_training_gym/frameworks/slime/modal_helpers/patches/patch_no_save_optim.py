"""Patch slime to honor Megatron's no_save_optim flag.

slime passes optimizer objects into Megatron's save_checkpoint() even when
--no-save-optim is set.  Clear those objects first so Megatron writes a
model-only checkpoint.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path("/root/slime/slime/backends/megatron_utils/model.py")
src = p.read_text()
old = """    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
"""
new = """    if getattr(args, "no_save_optim", False):
        optimizer = None
        opt_param_scheduler = None

    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
"""
marker = 'getattr(args, "no_save_optim", False)'
if marker not in src:
    if old not in src:
        raise RuntimeError("Could not patch slime model.py no_save_optim handling")
    src = src.replace(old, new, 1)
    p.write_text(src)
    print("Patched slime model.py to honor no_save_optim")
else:
    print("slime model.py already honors no_save_optim")
