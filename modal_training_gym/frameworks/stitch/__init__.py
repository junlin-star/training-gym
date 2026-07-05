"""Stitch framework package — disaggregated SLIME training with Modal Flash rollout.

``build_stitch_app`` is re-exported lazily so importing a sibling module doesn't
pull in the launcher, which imports ``modal`` — unavailable in the slime training
image.
"""

from __future__ import annotations

__all__ = ["build_stitch_app"]


def __getattr__(name: str):
    if name == "build_stitch_app":
        from .launcher import build_stitch_app

        return build_stitch_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
