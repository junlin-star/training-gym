"""Image-build-time patches for Megatron/PyTorch checkpoint handling.

Each ``patch_*.py`` file is a self-contained script that rewrites a
target source file inside the container image.  The helper below
reads the script and base64-encodes it so the launcher can embed it
in an ``Image.run_commands`` call without quoting issues.
"""

from __future__ import annotations

import base64
from pathlib import Path

_DIR = Path(__file__).parent


def encode_patch(name: str) -> str:
    """Return base64-encoded contents of ``patches/<name>.py``."""
    return base64.b64encode((_DIR / f"{name}.py").read_bytes()).decode()
