"""Weights & Biases run metadata.

Pure data — each framework config writes its own converter from this to its
specific CLI flags (e.g. SlimeRecipe emits `--wandb-project`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration shared across all frameworks.

    ## Fields

    project : str
        W&B project name. Default ``""``.
    entity : str
        W&B entity/team slug. Optional; when omitted, preflight resolves the
        default entity for the configured API key.
    group : str
        W&B group tag for organizing related runs. Default ``""``.
    exp_name : str
        W&B run display name. Default ``""``.
    key : str
        W&B API key. Usually injected via ``WANDB_API_KEY`` at launch
        time rather than hardcoded. Default ``""``.
    disable_random_suffix : bool
        When ``True``, suppresses the random suffix that W&B appends to
        run names. Default ``True``.
    modal_wandb_secret_name : str
        Name of the Modal secret containing the W&B API key. Default ``"wandb-secret"``.
    """

    project: str = ""
    entity: str = ""
    group: str = ""
    exp_name: str = ""
    key: str = ""
    disable_random_suffix: bool = True
    modal_wandb_secret_name: str = "wandb-secret"


def preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """
    Returns the resolved W&B entity for constructing deep-links to individual runs.
    """
    key = resolve_wandb_api_key(wandb_cfg)
    if not key:
        raise RuntimeError(
            "W&B logging is enabled (recipe.wandb=...) but no WANDB_API_KEY is "
            f"available — add it to the Modal secret "
            f"'{wandb_cfg.modal_wandb_secret_name}' (or set wandb.key=), or drop "
            "wandb= from the recipe to disable logging."
        )

    import wandb

    project = wandb_cfg.project or "uncategorized"
    entity = wandb_cfg.entity or os.environ.get("WANDB_ENTITY", "")
    try:
        wandb.login(key=key, verify=True, relogin=True)
        probe = wandb.init(
            project=project,
            entity=entity or None,
            name="_preflight",
            settings=wandb.Settings(silent=True, init_timeout=60),
        )
        entity = probe.entity
        probe_path = f"{probe.entity}/{probe.project}/{probe.id}"
        wandb.finish()
        try:
            wandb.Api(api_key=key).run(probe_path).delete()
        except Exception:
            pass  # a leftover empty "_preflight" run is harmless
    except Exception as exc:
        raise RuntimeError(
            f"W&B pre-flight failed for project '{project}': {exc}\n"
            f"The W&B key in Modal secret '{wandb_cfg.modal_wandb_secret_name}' can't "
            "log there (bad/expired key, or no write access to its entity). Point "
            "recipe.wandb at a project/entity you can write to, fix the secret, or "
            "drop wandb= to disable logging."
        ) from exc
    return entity


def resolve_wandb_api_key(wandb_cfg: WandbConfig | None) -> str:
    """Resolve W&B authentication without mutating a recipe/config object."""
    if wandb_cfg is None:
        return ""
    return os.environ.get("WANDB_API_KEY", "") or (wandb_cfg.key or "")


def install_wandb_api_key_in_process(wandb_cfg: WandbConfig | None) -> bool:
    """Install W&B authentication in this container process.

    Ray daemons inherit the Modal Function's environment, so installing the
    credential before ``ray start`` lets subsequently spawned workers inherit
    it without serializing the secret into Ray Job ``runtime_env`` metadata.
    The Modal Secret already wins through :func:`resolve_wandb_api_key`;
    ``wandb_cfg.key`` remains a backwards-compatible fallback.
    """
    api_key = resolve_wandb_api_key(wandb_cfg)
    if not api_key:
        return False
    os.environ["WANDB_API_KEY"] = api_key
    return True


def build_wandb_runtime_env(
    wandb_cfg: WandbConfig | None,
    *,
    run_id: str = "",
    entity: str = "",
) -> dict[str, str]:
    """Build the non-secret W&B portion of a Ray runtime environment.

    Authentication is deliberately absent. Call
    :func:`install_wandb_api_key_in_process` on every cluster node before
    starting Ray so the credential is inherited from the Modal Function
    environment rather than stored in Ray Job control-plane metadata.
    """
    env: dict[str, str] = {}
    if run_id:
        env["WANDB_RUN_ID"] = run_id
        env["WANDB_RESUME"] = "allow"
    if entity:
        env["WANDB_ENTITY"] = entity
    return env
