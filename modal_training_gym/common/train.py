import dataclasses as _dc
from typing import cast

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.ids import create_hash
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.frameworks.miles import build_miles_app
from modal_training_gym.frameworks.slime import build_slime_app
from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType
from modal_training_gym.train_recipes.miles_recipe import MilesConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


def _merge_recipe(base: SlimeRecipe, overrides: SlimeRecipe) -> SlimeRecipe:
    base_fields = {f.name: getattr(base, f.name) for f in _dc.fields(base)}
    for f in _dc.fields(overrides):
        if f.name not in base_fields:
            continue
        user_val = getattr(overrides, f.name)
        default_val = getattr(base, f.name, None)
        if default_val is None or user_val != default_val:
            base_fields[f.name] = user_val
    return type(base)(**base_fields)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class TrainConfig:
    """Compose dataset, model, and recipe into one training entrypoint."""

    # ── Composed configs (required) ─────────────────────────────────────────
    dataset: DatasetConfig
    model: ModelConfig
    recipe: BaseTrainRecipe
    checkpoint: Checkpoint | None = None
    _stable_id: str | None = _dc.field(default=None, init=False, repr=False)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def training_run_id(self) -> str:
        # Maintain same stable id, cannot change across calls on one TrainConfig.
        if self._stable_id is None:
            self._stable_id = create_hash(
                self.model.model_name,
                self.checkpoint.path if self.checkpoint is not None else "",
                f"{type(self.recipe).__name__}:{self.recipe.recipe_type.value}",
                self.dataset.dataset_id,
                self.model.model_path or "",
            )
        return self._stable_id

    def _build_app(self):
        recipe_type = self.recipe.recipe_type
        if recipe_type == RecipeType.MILES:
            if not isinstance(self.recipe, MilesConfig):
                raise TypeError(
                    f"Recipe type {recipe_type} requires MilesConfig, got {type(self.recipe).__name__}"
                )
            return build_miles_app(
                training_run_id=self.training_run_id,
                miles=cast(MilesConfig, self.recipe),
                model=self.model,
                dataset=self.dataset,
                checkpoint=self.checkpoint,
            )
        if recipe_type == RecipeType.SLIME:
            if not isinstance(self.recipe, SlimeRecipe):
                raise TypeError(
                    f"Recipe type {recipe_type} requires SlimeRecipe, got {type(self.recipe).__name__}"
                )
            base_recipe = SlimeRecipe.get_base_recipe(self.model)
            if base_recipe is not None:
                combined = _merge_recipe(base_recipe, cast(SlimeRecipe, self.recipe))
            else:
                combined = cast(SlimeRecipe, self.recipe)
            return build_slime_app(
                training_run_id=self.training_run_id,
                slime=combined,
                model=self.model,
                dataset=self.dataset,
                checkpoint=self.checkpoint,
            )
        raise ValueError(f"Unknown recipe type: {recipe_type}")

    def train(self) -> TrainResult:
        """Build the app, run training, and return the TrainResult."""
        import modal

        training_run_id = self.training_run_id
        print(f"Starting training run with id: {training_run_id}")

        app = self._build_app()
        result_dict = None
        with modal.enable_output():
            with app.run():
                modal_app_id = app.app_id or ""
                needs_slime_preconv = (
                    isinstance(self.recipe, SlimeRecipe)
                    and self.model
                    and getattr(self.model, "architecture", None)
                    and getattr(self.model.architecture, "needs_pre_conversion", False)
                )
                needs_miles_raw_conversion = (
                    isinstance(self.recipe, MilesConfig)
                    and getattr(self.recipe, "megatron_to_hf_mode", "bridge")
                    != "bridge"
                )
                if needs_slime_preconv or needs_miles_raw_conversion:
                    app.download.remote()
                    app.convert_checkpoint.remote()
                elif isinstance(self.recipe, SlimeRecipe) and not self.recipe.colocate:
                    app.download.remote()
                result_dict = app.train.remote(
                    modal_app_id=modal_app_id,
                    modal_app_url=modal_app_dashboard_url(modal_app_id),
                )
        if result_dict is None:
            raise RuntimeError(
                "Training app exited before returning a result. "
                "If you interrupted the run, restart with `modal run --detach`."
            )
        result = TrainResult(**TrainResult._parse_model_config(result_dict))
        print(f"Training complete: {result.training_run_id}")
        return result
