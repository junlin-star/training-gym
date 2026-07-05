"""StitchRecipe — disaggregated SLIME training with a stitch-managed rollout pool.

Extends :class:`SlimeRecipe` with fields controlling the disaggregated
Modal Flash SGLang rollout pool and the stitch weight-sync bulletin board.
The trainer publishes sparse weight deltas to a shared Modal Volume; rollout
servers sync from that bulletin board via a per-container sidecar.

Reference: https://github.com/modal-projects/stitch/tree/main/cookbook/slime_disagg
"""

from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.base import RecipeType
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class StitchRecipe(SlimeRecipe):
    """Disaggregated SLIME recipe: Flash rollout pool + stitch weight sync.

    The trainer runs SLIME on a Ray cluster (same as :class:`SlimeRecipe`) but
    rollouts are served by a separate Modal Flash pool of SGLang servers.
    Weight deltas are published to a shared Modal Volume and applied host-side
    by each sidecar.
    """

    recipe_type: RecipeType = RecipeType.STITCH

    # ── Disaggregated rollout pool ─────────────────────────────────────────
    # The Flash pool warm-floor size (min containers always running).
    rollout_min_containers: int = 4
    # Flash gateway proxy regions.
    rollout_proxy_regions: list[str] = field(default_factory=lambda: ["us-east"])
    # GPU type for rollout servers (defaults to trainer GPU type if empty).
    rollout_gpu_type: str = ""

    # ── Sidecar configuration ──────────────────────────────────────────────
    # How the sidecar applies weight versions: "in_place" pauses/applies/resumes
    # without draining; "quiesce" drains in-flight requests first.
    sidecar_commit_mode: str = "in_place"
    # Log every versioned sidecar proxy request at INFO for debugging.
    sidecar_debug_requests: bool = False

    # ── Delta bulletin board ───────────────────────────────────────────────
    # Name of the Modal Volume that holds the weight-delta bulletin board.
    # Each StitchRecipe deployment should use a dedicated volume.
    delta_volume_name: str = ""
    # Mount path for the delta volume inside both trainer and server containers.
    delta_bulletin_root: str = "/delta-bulletin"

    # ── SGLang rollout server args ─────────────────────────────────────────
    # Extra args passed to the SGLang server on each rollout replica (e.g.
    # --reasoning-parser, --context-length). Merged on top of structural args.
    sglang_server_extra_args: dict[str, str] = field(default_factory=dict)

    # ── Stitch weight sync (overrides from SlimeRecipe) ────────────────────
    # These defaults configure publish-only disk-delta mode for the trainer:
    # slime writes sparse deltas to the Volume bulletin board; the rollout
    # sidecars apply them host-side.
    colocate: bool = False
    rollout_num_gpus: int | None = 0
    update_weight_mode: str = "delta"
    update_weight_transport: str = "disk"
    update_weight_encoding: str = "indices"
    update_weight_delta_encoding: str = "xor"
    update_weight_delta_checksum: str = "xxh3-128"

    # ── Rollout request gating ─────────────────────────────────────────────
    # Pin rollout requests to a bounded-staleness weight version so unusable
    # (too-stale) rollouts are never generated.
    rollout_request_weight_version_mode: str = "exact"
    rollout_request_weight_version_lag: int = 0
    rollout_request_retry_attempts: int = 240
    rollout_request_retry_sleep: float = 1.0
    # Session affinity header for co-locating GRPO siblings on the same replica.
    rollout_session_affinity_header: str = "Modal-Session-ID"

    # ── Stitch source (pinned for reproducibility) ─────────────────────────
    stitch_repo_url: str = "https://github.com/modal-projects/stitch.git"
    stitch_repo_ref: str = "1486e2e"
    # Fork branch of slime with generic HTTP rollout endpoint and disk-delta
    # hooks. Set to "" to use the base SLIME_IMAGE's bundled slime.
    slime_fork_url: str = "https://github.com/modal-projects/slime.git"
    slime_fork_ref: str = "ebfe153949b1a69c39e92f947ed5d475166dd724"

    def effective_rollout_gpu_type(self) -> str:
        return self.rollout_gpu_type or self.gpu_type
