"""Image-build-time patch encoding.

Each framework keeps ``patch_*.py`` scripts in its own
``modal_helpers/patches/`` directory.  The helper below reads a script
and base64-encodes it so the launcher can embed it in an
``Image.run_commands`` call without quoting issues.
"""

from __future__ import annotations

import base64
from pathlib import Path

# Patches that target Megatron itself (``/root/Megatron-LM``), which the slime and
# miles images both ship. They are framework-agnostic, so they live here rather than
# under either framework's tree, and both launchers encode them from this constant so
# there is exactly one copy.
MEGATRON_PATCHES = Path(__file__).parent / "megatron_patches"


def encode_patch(name: str, patches_dir: Path) -> str:
    """Return base64-encoded contents of ``<patches_dir>/<name>.py``."""
    return base64.b64encode((patches_dir / f"{name}.py").read_bytes()).decode()
