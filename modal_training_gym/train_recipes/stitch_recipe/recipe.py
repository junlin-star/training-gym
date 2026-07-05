"""StitchRecipe: disaggregated SLIME training with a Modal Flash rollout pool.

Stitch separates the rollout servers from the trainer cluster. Weight updates
flow as sparse deltas through a Modal Volume bulletin board; an SGLang sidecar
on each rollout container applies deltas in order so requests are always served
by matching weights.

StitchRecipe extends SlimeRecipe with the disaggregated-specific configuration
(rollout pool GPU/size, bulletin board volume, sidecar commit mode) while
reusing SlimeRecipe's training-side fields verbatim.
"""

from dataclasses import field
from typing import Any, Literal

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.base import RecipeType
from modal_training_gym.train_recipes.gpu_allocation import (
    validate_megatron_actor_parallelism,
)
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe

# Fields that are StitchRecipe-specific launcher instructions, not slime CLI flags.
_STITCH_SKIP = {
    "rollout_gpu_type",
    "rollout_min_containers",
    "rollout_proxy_regions",
    "delta_volume_name",
    "delta_bulletin_root",
    "sidecar_commit_mode",
    "sidecar_debug_requests",
    "sglang_server_args",
    "sglang_server_concurrency",
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class StitchRecipe(SlimeRecipe):
    """Disaggregated SLIME recipe using Stitch for elastic rollout pools.

    The trainer runs SLIME/Ray on a clustered set of nodes and publishes sparse
    weight deltas to a Modal Volume bulletin board. A separate Modal Flash pool
    of SGLang servers syncs weights from the bulletin board via a sidecar and
    serves rollout traffic through the Flash gateway.

    Inherits all SlimeRecipe training-side fields; adds rollout-pool and
    bulletin-board configuration.
    """

    # ── Override parent required fields with disaggregated defaults ─────────
    # (must provide defaults for all parent required fields)
    gpu_type: str = "H200"
    colocate: bool = False
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus_per_engine: int = 1
    num_rollout: int = 3
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    save_interval: int = 10

    # ── App identity ──────────────────────────────────────────────────────────
    recipe_type: RecipeType = RecipeType.STITCH
    rollout_num_gpus: int | None = 0

    # ── Disaggregated rollout pool ─────────────────────────────────────────
    rollout_gpu_type: str = "H200"
    rollout_min_containers: int = 4
    rollout_proxy_regions: list[str] = field(default_factory=lambda: ["us-east"])
    sglang_server_concurrency: int = 64
    sglang_server_args: dict[str, str] = field(default_factory=dict)

    # ── Bulletin board (weight delta transport) ────────────────────────────
    delta_volume_name: str = ""
    delta_bulletin_root: str = "/delta-bulletin"
    sidecar_commit_mode: Literal["in_place", "quiesce"] = "in_place"
    sidecar_debug_requests: bool = True

    # ── Disaggregated-mode weight sync defaults ────────────────────────────
    update_weight_mode: str = "delta"
    update_weight_transport: str = "disk"
    update_weight_encoding: str = "indices"
    update_weight_delta_encoding: str = "xor"
    update_weight_delta_checksum: str = "xxh3-128"

    @model_validator(mode="after")
    def _validate_gpu_allocation(self) -> "StitchRecipe":
        """Override parent validator: skip rollout GPU checks for disaggregated mode.

        In stitch mode rollout GPUs live in the external Flash pool, so the
        parent's resolve_gpu_allocation() (which requires rollout_num_gpus > 0
        when colocate=False) doesn't apply. We still validate actor parallelism.
        """
        validate_megatron_actor_parallelism(self)
        return self

    @model_validator(mode="after")
    def _validate_stitch_config(self) -> "StitchRecipe":
        if self.colocate:
            raise ValueError(
                "StitchRecipe requires colocate=False (disaggregated mode). "
                "Use SlimeRecipe for colocated training."
            )
        if not self.delta_volume_name:
            model_slug = "default"
            if self.name:
                model_slug = self.name
            object.__setattr__(
                self,
                "delta_volume_name",
                f"stitch-delta-bulletin-{model_slug}",
            )
        return self

    def _fields(
        self,
        dataset: Any = None,
        model: Any = None,
    ) -> dict[str, Any]:
        """Extend SlimeRecipe._fields to exclude stitch-specific launcher fields."""
        fields = super()._fields(dataset=dataset, model=model)
        for key in _STITCH_SKIP:
            fields.pop(key, None)
        return fields
