"""Recipe for disaggregated GRPO training on Modal via stitch.

A stitch run is two halves that meet at a weight-delta bulletin board, so the
recipe is two fields::

    StitchRecipe(
        train=SlimeStitchTrainRecipe(...),   # the actor cluster that publishes
        serve=StitchServeRecipe(...),        # the Flash pool that applies
    )

The trainer half is a :class:`SlimeRecipe` — stitch is trainer-agnostic (its
cookbook has a ``miles_disagg`` path beside ``slime_disagg``), so the trainer is
a field rather than a base class, and its flags are maintained in one place
instead of copied. The serving half wraps the same
:class:`~modal_training_gym.deploy_recipes.sglang_recipe.recipe.SglangRecipe`
the deploy path uses.

Cross-half settings (rollout parallelism, the delta transport contract, W&B) are
*derived*, never set twice: a mismatch would otherwise only surface as a stalled
rollout twenty minutes into a Modal run.

This is the training-gym packaging of the ``stitch`` ``slime_disagg`` cookbook
(https://github.com/modal-projects/stitch/tree/main/cookbook/slime_disagg): the
``stitch`` library supplies the bulletin protocol + sidecar + slime hooks, and
this recipe + :func:`build_stitch_app` play the role the cookbook's config +
``modal_train.py`` play there.

It launches like the other recipes — ``TrainConfig(model=..., dataset=...,
recipe=StitchRecipe(...)).train()`` — with the Flash rollout pool coming up as
part of the same app.
"""

from __future__ import annotations

import json
from dataclasses import field
from typing import Any

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesConfig
from modal_training_gym.train_recipes.slime_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    JSON_CONFIG_FIELDS,
    SlimeRecipe,
)
from modal_training_gym.train_recipes.stitch_recipe.serve import StitchServeRecipe
from modal_training_gym.train_recipes.stitch_recipe.train import (
    HOOK_CONFIG_FIELDS,
    SlimeStitchTrainRecipe,
)

__all__ = [
    "CHECKPOINTS_PATH",
    "DATA_PATH",
    "HF_CACHE_PATH",
    "HOOK_CONFIG_FIELDS",
    "YAML_CONFIG_FIELDS",
    "StitchRecipe",
    "StitchServeRecipe",
    "SlimeStitchTrainRecipe",
]

# Fields slime reads as YAML files at runtime. Recipes set them as inline dicts;
# the launcher materializes them to temp YAML files before building the command.
# ``custom_config_path`` rather than ``extra_config``: the trainer half renames it
# when it resolves fields.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class StitchRecipe(BaseTrainRecipe):
    """Disaggregated GRPO on Modal: a publishing trainer and a syncing pool.

    ## Fields

    train : SlimeStitchTrainRecipe
        The actor cluster. Every :class:`SlimeRecipe` field applies, defaulted
        for a publish-only disaggregated run.
    serve : StitchServeRecipe
        The Modal Flash pool of SGLang replicas that serves rollouts and applies
        published deltas in place.
    name : str
        Modal app name. Empty → derived from the model name.
    app_tags : dict
        Extra Modal app tags, merged over the standard training-gym ones.
    wandb : WandbConfig | None
        Applied to the trainer half and to the app's dashboard tags.
    """

    recipe_type: RecipeType = RecipeType.STITCH

    train: SlimeStitchTrainRecipe | MilesConfig = field(
        default_factory=SlimeStitchTrainRecipe
    )
    serve: StitchServeRecipe = field(default_factory=StitchServeRecipe)

    name: str = ""
    app_tags: dict = field(default_factory=dict)
    wandb: WandbConfig | None = None

    @model_validator(mode="after")
    def _resolve_halves(self) -> StitchRecipe:
        """Derive the settings both halves have to agree on, and reject the ones
        that can't be reconciled."""
        train = self.train
        if not isinstance(train, SlimeStitchTrainRecipe):
            raise NotImplementedError(
                "stitch's miles_disagg path is not wired up yet: build_stitch_app "
                "launches the slime trainer only. Use SlimeStitchTrainRecipe."
            )
        if self.wandb is not None:
            train.wandb = self.wandb
        # Rollout parallelism is a property of a replica, so the pool owns it and
        # the trainer follows: slime sizes its request fan-out from this.
        train.rollout_num_gpus_per_engine = self.serve.gpus_per_replica
        # Publish-only: rollouts come from the pool, so the actor cluster holds
        # no engines of its own.
        train.rollout_num_gpus = 0
        train.colocate = False
        # The bulletin board is a shared Modal Volume, and the sidecar reloads
        # from a file it patches on disk. Neither half can opt out alone.
        if train.update_weight_mode != "delta":
            raise ValueError(
                "the stitch pool applies sparse deltas; "
                f"train.update_weight_mode must be 'delta', got {train.update_weight_mode!r}"
            )
        if train.update_weight_transport != "disk":
            raise ValueError(
                "deltas reach the pool through the bulletin Volume; "
                f"train.update_weight_transport must be 'disk', got "
                f"{train.update_weight_transport!r}"
            )
        return self

    @property
    def slime_train(self) -> SlimeStitchTrainRecipe:
        """The trainer half, narrowed to the slime backend the launcher wires."""
        if not isinstance(self.train, SlimeStitchTrainRecipe):
            raise NotImplementedError(
                "stitch's miles_disagg path is not wired up yet: "
                f"got {type(self.train).__name__}"
            )
        return self.train

    # ── Converters (delegated to the trainer half) ──────────────────────────

    @staticmethod
    def _resolve_data_paths(ds: DatasetConfig) -> tuple[str, dict[str, str] | None]:
        return SlimeRecipe._resolve_data_paths(ds)

    def slime_fields(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> dict[str, Any]:
        """Resolved slime CLI fields (name → value), excluding infra + the three
        run-time-injected fields the trainer fills in per launch
        (``rollout_endpoint_url``, ``update_weight_disk_dir``,
        ``custom_config_path``)."""
        return self.slime_train._fields(dataset=dataset, model=model)

    def cli_args(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> list[str]:
        """Flatten :meth:`slime_fields` to a slime CLI argv list. YAML config
        fields (:data:`YAML_CONFIG_FIELDS`) are skipped here — the launcher
        materializes them to files and appends the resolved flags."""
        return fields_to_argv(self.slime_fields(model=model, dataset=dataset))

    def to_payload(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> dict[str, Any]:
        """Plain-data slime args the trainer runs with."""
        train = self.slime_train
        return {
            "fields": self.slime_fields(model=model, dataset=dataset),
            "environment": dict(train.environment),
            "async_mode": train.async_mode,
            "slime_model_script": train.slime_model_script,
        }


def fields_to_argv(fields: dict[str, Any]) -> list[str]:
    """slime CLI argv from a resolved field dict.

    Rules: ``field_name`` → ``--field-name``; ``True`` → bare flag;
    ``False``/``None``/``""`` → omitted; list → ``--flag v1 v2 …``; dict in
    :data:`YAML_CONFIG_FIELDS` → skipped (materialized to a file by the
    launcher); dict in :data:`JSON_CONFIG_FIELDS` → ``--flag '<json>'``.
    """
    out: list[str] = []
    for key, val in fields.items():
        if val is None or val is False or val == "":
            continue
        if key in YAML_CONFIG_FIELDS and isinstance(val, dict):
            continue
        flag = f"--{key.replace('_', '-')}"
        if val is True:
            out.append(flag)
        elif isinstance(val, dict) and key in JSON_CONFIG_FIELDS:
            out += [flag, json.dumps(val)]
        elif isinstance(val, (list, tuple)):
            out += [flag] + [str(v) for v in val]
        else:
            out += [flag, str(val)]
    return out
