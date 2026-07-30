"""Moonlight-16B-A3B model spec as a concrete HFModelConfiguration subclass."""

from __future__ import annotations

from .base import HFModelConfiguration


class Moonlight_16B_A3B(HFModelConfiguration):
    """Moonlight-16B-A3B-Instruct (16B total, ~3B active) MoE model from Moonshot.

    DeepSeek-V3 architecture (multi-latent attention, 64 routed + 2 shared
    experts, sigmoid router with expert bias), i.e. a small rung of the same
    ladder as Kimi K2 — which is why slime ships a dedicated model script
    (``scripts/models/moonlight.sh``) for it rather than plain
    ``ModelArchitecture`` args. Recipes therefore set ``slime_model_script`` and
    leave ``architecture`` unset; this class only names the HF repo and how to
    fetch it.
    """

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
