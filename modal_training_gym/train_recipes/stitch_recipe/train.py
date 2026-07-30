"""The trainer half of a stitch run: slime in publish-only disaggregated mode.

This is a :class:`SlimeRecipe` — same flags, same model/dataset/wandb
converters, same GPU-allocation validation — with the slime_disagg deltas on
top: no local rollout engines, sparse weight deltas published to a bulletin
board instead of an NCCL broadcast, and stitch's request/publish hooks wired in.

Deriving from ``SlimeRecipe`` is the point: a stitch run's trainer *is* a slime
trainer, so its ~100 flags should be maintained in one place. Stitch itself is
trainer-agnostic (its cookbook has a ``miles_disagg`` path too), which is why
:class:`~modal_training_gym.train_recipes.stitch_recipe.recipe.StitchRecipe`
takes the trainer as a field rather than being one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import Any

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.gpu_allocation import (
    GpuAllocation,
    validate_megatron_actor_parallelism,
)
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe
from modal_training_gym.train_recipes.stitch_recipe.pins import (
    SLIME_IMAGE_TAG,
    SLIME_REPO_REF,
    SLIME_REPO_URL,
)

__all__ = ["HOOK_CONFIG_FIELDS", "SlimeStitchTrainRecipe"]

# Fields only stitch's own hooks read, off the slime args namespace. slime defines
# no such CLI flags, and its parser is parse_known_args — passing them as flags
# drops them silently (the hooks then see their fallbacks, e.g. 60 request
# retries instead of the configured budget). The launcher merges these into
# ``custom_config_path`` instead, which slime setattrs onto args.
HOOK_CONFIG_FIELDS = frozenset(
    {
        "rollout_request_weight_version_mode",
        "rollout_request_weight_version_lag",
        "rollout_request_retry_attempts",
        "rollout_request_retry_sleep",
        "rollout_session_affinity_header",
    }
)

# Inherited SlimeRecipe fields that must NOT reach this trainer's command line.
# Either the pinned slime fork doesn't define the flag (its parser would drop it
# silently, so emitting it only misleads), or it configures a local rollout
# engine, which a disaggregated run does not have.
_TRAINER_DROP = frozenset(
    {
        # Local-engine settings: rollouts come from the Flash pool
        # (StitchServeRecipe configures those engines instead).
        "sglang_mem_fraction_static",
        "sglang_enable_dp_attention",
        "sglang_dp_size",
        "sglang_ep_size",
        "sglang_enable_dp_lm_head",
        "sglang_disable_custom_all_reduce",
        "sglang_cuda_graph_bs",
        "sglang_max_running_requests",
        "sglang_tool_call_parser",
        "sglang_reasoning_parser",
        # Full-broadcast weight sync; the delta flags below replace it.
        "update_weight_encoding",
        # Not defined by the pinned fork.
        "qkv_format",
        # Modal infra, not slime flags.
        "gpu_type",
        # Image/repo pins: build instructions, not slime flags.
        "slime_image_tag",
        "slime_repo_url",
        "slime_repo_ref",
        # Phase reporting hooks: colocated-slime wrappers that assume slime owns
        # the rollout engines.
        "custom_rollout_log_function_path",
        "custom_eval_rollout_log_function_path",
        "custom_megatron_before_log_prob_hook_path",
        "custom_megatron_before_train_step_hook_path",
    }
    | HOOK_CONFIG_FIELDS
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class SlimeStitchTrainRecipe(SlimeRecipe):
    """slime trainer for a disaggregated stitch run.

    Don't see the flag you need? Every :class:`SlimeRecipe` field applies here,
    and ``extra_config`` remains the escape hatch for the rest.
    """

    # ── SlimeRecipe's required fields, defaulted for a disagg run ───────────
    gpu_type: str = "H200"
    # Rollouts are served by the Flash pool, so the actor cluster allocates no
    # rollout GPUs and never shares its own with an engine.
    colocate: bool = False
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus_per_engine: int = 1
    # No rollout GPUs in this cluster — the Flash pool owns them.
    rollout_num_gpus: int | None = 0
    num_rollout: int = 3
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    save_interval: int = 20

    # ── Defaults that differ from colocated slime ───────────────────────────
    # No local Megatron rollout engine to put on the path.
    environment: dict = field(
        default_factory=lambda: {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    use_kl_loss: bool = True
    eval_max_response_len: int = 8192
    megatron_to_hf_mode: str = "bridge"
    # A disagg run's rollout traffic is HTTP against the pool, so a mid-run
    # restart would have to re-claim it; fail the run instead.
    use_fault_tolerance: bool = False
    rollout_function: Callable | str | None = (
        "slime.rollout.sglang_rollout.generate_rollout"
    )

    # ── The slime fork that speaks the bulletin protocol ────────────────────
    slime_image_tag: str = SLIME_IMAGE_TAG
    slime_repo_url: str = SLIME_REPO_URL
    slime_repo_ref: str = SLIME_REPO_REF

    # ── Weight sync: publish sparse deltas to the bulletin board ───────────
    update_weight_mode: str = "delta"
    update_weight_transport: str = "disk"
    update_weight_delta_encoding: str = "xor"
    update_weight_delta_checksum: str = "xxh3-128"
    # rank-0 publish hook: advance the pointer, commit the Volume, wake the pool.
    custom_delta_pre_push_path: str = (
        "modal_training_gym.frameworks.stitch.bulletin_hooks.commit_and_wake"
    )

    # ── Rollout request gating (stitch hooks) ───────────────────────────────
    # Pins each rollout request to a served weight version; a lagging replica
    # returns a retryable 409 so requests flow across a weight update.
    custom_rollout_request_hook_path: str = (
        "modal_training_gym.frameworks.stitch.bulletin_hooks.gated_rollout_request_hook"
    )
    rollout_request_weight_version_mode: str = "exact"
    rollout_request_weight_version_lag: int = 0
    rollout_request_retry_attempts: int = 240
    rollout_request_retry_sleep: float = 1.0
    # The trainer hits the Flash gateway directly, which routes session affinity
    # on Modal-Session-ID; emit that so GRPO siblings co-locate.
    rollout_session_affinity_header: str = "Modal-Session-ID"

    @model_validator(mode="after")
    def _validate_gpu_allocation(self) -> SlimeStitchTrainRecipe:
        """Actor-cluster checks only, replacing SlimeRecipe's.

        Its allocator reads ``colocate=False`` as "slime also owns rollout GPUs",
        and so requires a positive ``rollout_num_gpus``; here the rollout engines
        are a separate Modal Flash pool, sized by the serving half.
        """
        validate_megatron_actor_parallelism(self)
        return self

    @property
    def gpu_allocation(self) -> GpuAllocation:
        """The actor cluster's GPUs. Rollout GPUs live in the Flash pool and are
        not part of this allocation."""
        actor_gpus = self.actor_num_nodes * self.actor_num_gpus_per_node
        return GpuAllocation(
            actor_gpus=actor_gpus,
            critic_gpus=0,
            rollout_gpus=0,
            total_gpus=actor_gpus,
            total_nodes=self.actor_num_nodes,
            gpus_per_node=self.actor_num_gpus_per_node,
            rollout_num_gpus_per_engine=self.rollout_num_gpus_per_engine,
            rollout_engines=0,
            colocate=False,
        )

    def _fields(
        self,
        dataset: DatasetConfig | None = None,
        model: ModelConfig | None = None,
    ) -> dict[str, Any]:
        fields = super()._fields(dataset=dataset, model=model)
        for name in _TRAINER_DROP:
            fields.pop(name, None)
        # bridge mode loads HF weights directly as the reference; default
        # ref_load to the base checkpoint when the recipe didn't set one.
        if self.megatron_to_hf_mode == "bridge" and not fields.get("ref_load"):
            hf_checkpoint = fields.get("hf_checkpoint")
            if isinstance(hf_checkpoint, str) and hf_checkpoint:
                fields["ref_load"] = hf_checkpoint
        return fields
