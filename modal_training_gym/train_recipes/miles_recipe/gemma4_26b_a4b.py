"""Gemma-4-26B-A4B GRPO recipe on miles (1x8xH200), text-only or vision-language."""

from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfig

_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)

# Build-time shims for upstream gaps; see each patch's docstring.
_PATCHES = (
    "patch_router_startup_timeout",
    "patch_gemma4_vl_rollout_text",
)


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in _PATCHES
    ]


def _has_images(dataset: "DatasetConfig | None") -> bool:
    return "image" in (getattr(dataset, "multimodal_keys", None) or {})


_EPHEMERAL_DISK_MIB = 1_048_576


# Applied over the text defaults on an image dataset, for fields the caller left
# alone: smaller rollouts, since images are expensive.
_VISION_MODE: dict[str, Any] = {
    "num_rollout": 15,
    "rollout_batch_size": 8,
    "n_samples_per_prompt": 8,
    "global_batch_size": 64,
    "rollout_max_response_len": 256,
    "rollout_temperature": 1.0,
    "rollout_top_p": 0.95,
    "rollout_top_k": 64,
    "rm_type": None,
    "sglang_max_running_requests": 8,
    "save_interval": 10,
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26B_A4B_Recipe(MilesRecipe):
    """Gemma-4-26B-A4B MoE GRPO on 1×8×H200 with TP4/PP1/EP8, colocated.

    One checkpoint, two modes: these fields train on text, and an image dataset
    (``MultimodalDataset(modality="image")``) makes ``_for_dataset`` swap in
    ``_VISION_MODE`` for every field the caller left alone. Vision datasets need
    ``apply_chat_template=True`` so the prompt reaches the processor as a string,
    and a ``custom_rm_function``: the text path's ``gemma_math`` reward is
    cleared in vision mode, since no built-in scores an image task.

    Follows upstream ``scripts/run_gemma_4_26b_a4b.py``; the comments below mark
    where it does not, since three of its flags fail on any image with Gemma-4.
    """

    gpu_type: str = "H200"
    colocate: bool = True
    image_run_commands: list[str] = field(default_factory=_image_patches)

    hf_checkpoint: str = "google/gemma-4-26B-A4B-it"
    ref_load: str = "google/gemma-4-26B-A4B-it"
    megatron_to_hf_mode: str = "bridge"
    miles_model_script: str = "scripts/models/gemma-4-26b-a4b-it.sh"
    # Model overflows container disk, so reserve 1 TiB.
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _EPHEMERAL_DISK_MIB}
    )

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    # ── Parallelism ──────────────────────────────────────────────────────────
    train_backend: str = "megatron"
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1

    # Off, unlike upstream: Gemma-4's decoder layer returns a tuple and Megatron's
    # checkpointed forward fails on it ("save_for_backward can only save
    # variables"). Revisit first if activation memory bites.
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    recompute_num_layers: int | None = None
    # bshd (below) rules out dynamic batching; miles asserts on the pair, so use an
    # explicit micro batch. Upstream passes both and trips that assertion.
    # max_tokens_per_gpu is inert while dynamic batching is off.
    use_dynamic_batch_size: bool = False
    micro_batch_size: int = 1
    max_tokens_per_gpu: int = 1024

    # ── Rollout ──────────────────────────────────────────────────────────────
    rm_type: str | None = "gemma_math"
    rollout_shuffle: bool = True
    balance_data: bool = True
    num_rollout: int = 3
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 256
    rollout_temperature: float = 1.0
    # Widened to None so text mode omits the flag and Miles keeps its default;
    # vision mode sets both.
    rollout_top_p: float | None = None
    # generation_config.json's eos_token_id: <eos>, <turn|>, <|tool_response>.
    rollout_stop_token_ids: list[int] | None = field(
        default_factory=lambda: [1, 106, 50]
    )
    global_batch_size: int = 256
    save_interval: int = 20

    rollout_num_gpus_per_engine: int = 4
    # 0.25, not upstream's 0.55: both engines stay resident, and at 0.55 the
    # optimizer step OOMs on a 139.8 GiB H200.
    sglang_mem_fraction_static: float = 0.25
    # Gemma-4's global head_dim=512 exceeds FlashAttention's 256 cap.
    sglang_attention_backend: str = "triton"
    sglang_moe_runner_backend: str = "triton"
    sglang_disable_custom_all_reduce: bool = True
    sglang_disable_cuda_graph: bool = True
    sglang_disable_overlap_schedule: bool = True
    sglang_disable_radix_cache: bool = True
    sglang_max_running_requests: int | None = None
    # Resident, as upstream has it: offloading instead hits an illegal memory
    # access in SGLang's memory-saver path during the training step.
    no_offload_train: bool = True
    no_offload_rollout: bool = True
    # Off, unlike upstream: it enables sglang's routed-experts capturer, which
    # reads num_experts_per_tok — Gemma-4 calls it top_k_experts, so every
    # scheduler dies with AttributeError. Costs MoE routing replay.
    use_rollout_routing_replay: bool = False

    # ── Objective / optimizer ────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98

    # ── Numerics ─────────────────────────────────────────────────────────────
    attention_backend: str = "unfused"
    qkv_format: str = "bshd"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    no_gradient_accumulation_fusion: bool = True
    no_check_for_nan_in_loss_and_grad: bool = True

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Gemma4_26B_A4B_Recipe":
        """Keep the build-time patches at the head of ``image_run_commands``.

        Prepending rather than defaulting makes the field additive: a caller
        adding their own command would otherwise drop the patches, and losing the
        VL one shows up as a blind model rather than an error.
        """
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "Gemma4_26B_A4B_Recipe":
        """Keep the disk reservation when a caller supplies their own kwargs.

        Same additive reasoning as ``_keep_image_patches``: the dict is replaced
        wholesale, so passing ``{"secrets": [...]}`` would drop the reservation
        and the run would die part-way through the checkpoint download. A caller
        who names ``ephemeral_disk`` still wins.
        """
        kwargs = self.train_function_kwargs or {}
        if "ephemeral_disk" not in kwargs:
            object.__setattr__(
                self,
                "train_function_kwargs",
                {"ephemeral_disk": _EPHEMERAL_DISK_MIB, **kwargs},
            )
        return self

    def _brings_own_reward(self) -> bool:
        if self.custom_rm_function is not None or self.rollout_function is not None:
            return True
        if "rm_type" in self.explicit_fields and self.rm_type:
            return True
        extra = self.extra_config
        if isinstance(extra, str) and extra:
            return True
        return isinstance(extra, dict) and bool(extra.get("custom_rm_path"))

    def _for_dataset(self, dataset: "DatasetConfig | None") -> MilesRecipe:
        if not _has_images(dataset):
            return self
        if not self._brings_own_reward():
            raise TrainingGymConfigError(
                f"{type(self).__name__} on an image dataset needs its own reward: "
                f"the text default rm_type={self.rm_type!r} scores maths, not "
                "images, so vision mode clears it. Pass custom_rm_function=..., "
                "or rm_type=... to choose a built-in deliberately."
            )
        # Keyed on what the caller passed, not on how the value compares: an
        # explicit value equal to the text default must still win.
        explicit = self.explicit_fields
        vision = {
            name: value for name, value in _VISION_MODE.items() if name not in explicit
        }
        resolved = replace(self, **vision)
        # `replace` marks every field explicit, so carry the real set forward and
        # resolving twice stays a no-op.
        object.__setattr__(resolved, "_explicit_fields", explicit | vision.keys())
        return resolved

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        super().validate_model_parallelism(model)
        if self.pipeline_model_parallel_size != 1:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs pipeline_model_parallel_size=1: the "
                "bridge keeps the vision tower and embedding on one pipeline stage, "
                "so a split only fails once Megatron builds the model. Got "
                f"{self.pipeline_model_parallel_size}."
            )
