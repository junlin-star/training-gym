"""Add wall-clock timing instrumentation to slime's weight-sync path.

Wraps ``UpdateWeightFromTensor.update_weights`` (colocated mode) and
``UpdateWeightFromDistributed.update_weights`` (distributed mode) so
every sync prints a ``[weight_sync]`` log line with total elapsed time.
These show up in Modal container logs and can be parsed / forwarded to
wandb for cross-run comparison.

Executed at image-build time via ``python3 <this file>``.
"""

from __future__ import annotations

import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Locate the file(s) that contain the weight-update classes.
# ---------------------------------------------------------------------------
SEARCH_ROOTS = [
    "/root/slime/slime",
    "/usr/local/lib/python3.12/dist-packages/slime",
]

_TARGETS: dict[str, Path] = {}  # class_name -> file_path

for root in SEARCH_ROOTS:
    for fpath in glob.glob(f"{root}/**/*.py", recursive=True):
        try:
            text = Path(fpath).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for cls in ("UpdateWeightFromTensor", "UpdateWeightFromDistributed"):
            if f"class {cls}" in text and "def update_weights" in text:
                _TARGETS.setdefault(cls, Path(fpath))

if not _TARGETS:
    print(
        "WARNING: patch_weight_sync_timing — could not find weight-update "
        "classes to instrument; timing will not be reported."
    )
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 2. For each class found, wrap `update_weights` with timing.
#
# Strategy: inject a tiny timing wrapper *before* the method definition.
# We rename the original to `_orig_update_weights` and add a replacement
# that calls it while measuring elapsed time.
# ---------------------------------------------------------------------------
TIMING_SNIPPET = """\

# --- [training-gym] weight-sync timing instrumentation ---
import time as _ws_time

_ws_orig_{tag} = {cls}.update_weights

def _ws_timed_{tag}(self) -> None:
    _t0 = _ws_time.monotonic()
    try:
        return _ws_orig_{tag}(self)
    finally:
        _elapsed = _ws_time.monotonic() - _t0
        _min = _elapsed / 60
        print(
            f"[weight_sync] {{type(self).__name__}}.update_weights "
            f"finished in {{_elapsed:.1f}}s ({{_min:.2f}} min)"
        )

{cls}.update_weights = _ws_timed_{tag}
# --- end weight-sync timing instrumentation ---
"""

patched_files: set[Path] = set()
for cls_name, fpath in _TARGETS.items():
    src = fpath.read_text()
    tag = cls_name.lower()
    marker = f"_ws_orig_{tag}"
    if marker in src:
        print(f"patch_weight_sync_timing: {fpath} already patched for {cls_name}")
        continue

    snippet = TIMING_SNIPPET.format(cls=cls_name, tag=tag)
    src += snippet
    fpath.write_text(src)
    patched_files.add(fpath)
    print(f"patch_weight_sync_timing: instrumented {cls_name} in {fpath}")

if not patched_files:
    print("patch_weight_sync_timing: nothing to patch (already instrumented)")
