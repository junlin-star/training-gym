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
# miles images both ship — they are not slime-specific despite living under the
# slime tree for historical reasons. Both launchers encode them from this constant
# so there is exactly one copy; moving the files would invalidate the slime image
# cache for no behavioral gain.
MEGATRON_PATCHES = (
    Path(__file__).parent.parent / "frameworks" / "slime" / "modal_helpers" / "patches"
)


def encode_patch(name: str, patches_dir: Path) -> str:
    """Return base64-encoded contents of ``<patches_dir>/<name>.py``."""
    return base64.b64encode((patches_dir / f"{name}.py").read_bytes()).decode()
