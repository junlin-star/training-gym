"""slime framework package.

``build_slime_app`` is exported lazily: importing it pulls in the launcher, which
imports ``modal``. The transcription rollout in this package runs *inside* the
slime training image (where ``modal`` isn't installed) and is imported by slime via
``importlib.import_module``, which runs this ``__init__``. Eagerly importing the
launcher here would drag ``modal`` into that import and crash the rollout — so we
defer it via ``__getattr__`` instead.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_slime_app"]


def __getattr__(name: str) -> Any:
    if name == "build_slime_app":
        from .launcher import build_slime_app

        return build_slime_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
