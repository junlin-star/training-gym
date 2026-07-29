"""Upstream versions the disaggregated stitch flow is pinned to.

Three moving parts have to agree: the ``stitch`` control plane, the slime fork
that publishes deltas into it, and the SGLang fork that applies them behind
``/stage_weight_update``. Each is pinned to an exact commit (not a branch tip)
because the fetch/checkout is a cached image layer — a moving tip would leave a
stale build silently in place.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── stitch ─────────────────────────────────────────────────────────────────────
# The control plane: Store/Engine/Pool ports plus the trainer publish helpers and
# the rollout-service runtime the sidecar runs. modal-projects repos are public,
# so a build-time pip install needs no token.
STITCH_REPO_URL = "https://github.com/modal-projects/stitch.git"
STITCH_REPO_REF = "c4e4fb949a16a705455c7df7411f4df6044bc765"

# ── slime fork ─────────────────────────────────────────────────────────────────
# Carries the generic HTTP rollout endpoint + publish-only disk-delta hooks the
# disagg flow drives; the stock ``slimerl/slime`` image does not.
SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
SLIME_REPO_URL = "https://github.com/modal-projects/slime.git"
SLIME_REPO_REF = "11bb0fa48aa37d5c54fe297143c6bc1d40f311bf"
SLIME_ROOT = "/root/slime"


@dataclass(frozen=True)
class SGLangRuntime:
    """An SGLang source overlay and the ABI-compatible base image it is copied over."""

    image: str
    repository: str
    branch: str
    commit: str


# ── SGLang fork ────────────────────────────────────────────────────────────────
# Asynchronous weight staging (``/stage_weight_update``), correct quantized weight
# loading, and the optional CPU delta cache. See the fork's SGLANG_FORK.md for the
# patch stack and how to re-port onto a newer SGLang release.
DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.16",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.16",
    commit="1a4a4fd6b54ab332c3b0b17a0383037390939587",
)
