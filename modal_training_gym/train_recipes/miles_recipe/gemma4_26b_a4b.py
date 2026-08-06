"""Gemma-4-26B-A4B GRPO recipe on miles (1x8xH200), text-only or vision-language."""

from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfig

# No ``docker_image`` override: this recipe rides MilesRecipe's shared default.
# Gemma-4 needs an image built after 2026-06-24 (PR #1232, which also changed
# model_provider.py for the bridge's ``(logits, loss_mask)`` return, arguments.py
# for Gemma-4's per-attention-type ``rope_theta`` nesting, and
# hf_weight_iterator_bridge.py for its ``layer_scalar``/``scale`` buffers), and the
# shared default now satisfies that. If it is ever rolled back before that date,
# this recipe needs its own pin again.

_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)

# Build-time shims for upstream gaps, applied only to this recipe's image. Each
# patch's docstring carries the failure it fixes and when it can be dropped:
#
# * router_startup_timeout -- loading 26B through the bridge starves the spawned
#   router past miles' hardcoded 30s bind timeout.
# * gemma4_vl_rollout_text -- miles sends the processor's pre-expanded input_ids
#   and SGLang rejects them against the single raw image. Gated on a Gemma-4
#   processor inside the patch, so no other model changes behaviour.
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


# Applied over the text-only defaults below on an image dataset, to the fields the
# caller left alone. Vision mode diverges far less here than it did on slime: the
# canonical miles config already runs bridge mode with ``qkv_format=bshd``,
# ``attention_backend=unfused`` and PP=1, which is exactly what the VL model needs.
# What is left is rollout economics -- images are expensive -- plus the sampling
# defaults SGLang's /generate path does not read out of generation_config.json.
_VISION_MODE: dict[str, Any] = {
    "num_rollout": 15,
    "rollout_batch_size": 8,
    "n_samples_per_prompt": 8,
    "global_batch_size": 64,
    "rollout_max_response_len": 256,
    "rollout_temperature": 1.0,
    # From generation_config.json, which miles' /generate path does not apply.
    "rollout_top_p": 0.95,
    "rollout_top_k": 64,
    # generation_config.json's eos_token_id: <eos>, <turn|>, <|tool_response>.
    "rollout_stop_token_ids": [1, 106, 50],
    "sglang_max_running_requests": 8,
    "save_interval": 10,
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26B_A4B_Recipe(MilesRecipe):
    """Gemma-4-26B-A4B (26B-A4B MoE) on 1x8xH200 with TP4/PP1/EP8, colocated GRPO.

    Mirrors upstream ``scripts/run_gemma_4_26b_a4b.py``. One checkpoint, two modes:
    the fields below train on text data, and an image dataset
    (``MultimodalDataset(modality="image")``) switches ``_for_dataset`` to the
    vision-mode values for every field the caller left at its default. Nothing has
    to be passed by hand, and explicit values still win.

    Both modes go through the HF<->Megatron bridge on the base VLM checkpoint --
    there is no text-only variant of the weights, and SGLang serves
    ``Gemma4ForConditionalGeneration`` either way.

    Why the unusual backend flags:

    * ``attention_backend="unfused"`` + ``qkv_format="bshd"`` -- Gemma-4's global
      attention uses ``head_dim=512``, past what the fused/flash kernels accept.
    * ``sglang_attention_backend="triton"`` + ``sglang_moe_runner_backend="triton"``
      -- same 512 head_dim exceeds FlashAttention's 256 cap on the rollout side.
    * ``no_offload_train`` / ``no_offload_rollout`` with ``mem_fraction_static=0.25``
      -- colocated 26B MoE. Both engines stay resident, so SGLang's share is cut
      well below upstream's 0.55 to leave the training step room.

    Unlike the slime recipe this replaces, the vision tower is **not** frozen: miles
    has no ``--freeze-params-name-list`` equivalent, so RL updates the whole VLM.

    **Status.** Both modes are validated end-to-end on 1x8xH200, two GRPO steps each,
    on ``dev-202608051303`` (and before it on ``dev-202608041247``).

    * Text (DAPO-Math-17k): ``train_rollout_logprob_abs_diff`` ~0.011,
      ``train_rollout_kl`` ~0.0008.
    * Vision (geo3k, ``MultimodalDataset(modality="image")``): ``ppo_kl`` ~-8e-9,
      ``loss`` ~-6e-9, ``ess_ratio`` 1.0.

    Both say the same thing -- Megatron and SGLang agree on the weights after sync,
    so Gemma-4's ``layer_scalar`` buffers survive the transfer despite SGLang
    logging them as default-initialised at load.

    Losses sit at ~0 because these smoke runs truncate every response at 256 tokens,
    so no sample reaches an answer, every reward is 0 and GRPO's advantages vanish.
    That exercises the full pipeline but says nothing about learning; a real run
    needs a longer ``rollout_max_response_len``.

    Vision mode needs ``apply_chat_template=True`` on the dataset: the processor and
    the patched rollout both require ``sample.prompt`` to be a templated string.
    """

    gpu_type: str = "H200"
    colocate: bool = True
    image_run_commands: list[str] = field(default_factory=_image_patches)

    hf_checkpoint: str = "google/gemma-4-26B-A4B-it"
    ref_load: str = "google/gemma-4-26B-A4B-it"
    megatron_to_hf_mode: str = "bridge"
    miles_model_script: str = "scripts/models/gemma-4-26b-a4b-it.sh"
    # Model overflows container disk, so reserve 1 TiB.
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
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

    # Off, despite upstream's script asking for full/uniform/1. Gemma-4's decoder
    # layer returns a tuple, and Megatron's checkpointed forward hands it straight
    # to autograd: "save_for_backward can only save variables, but argument 4 is of
    # type tuple". Affordable to skip here -- micro_batch_size=1 at 1024 tokens on
    # H200 -- but it is the first thing to revisit if activation memory bites.
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    recompute_num_layers: int | None = None
    # ``qkv_format="bshd"`` (below) rules out dynamic batching -- miles asserts
    # "Dynamic batch size is not supported for bshd format. Please specify
    # --micro-batch-size instead." Upstream's own scripts/run_gemma_4_26b_a4b.py
    # still passes both and fails this assertion on any image that has it, so
    # follow the assertion rather than the script. ``max_tokens_per_gpu`` is
    # inert while dynamic batching is off; kept so the flag is there if a future
    # image lets Gemma-4 run THD.
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
    rollout_top_p: float | None = None
    rollout_top_k: int | None = None
    rollout_stop_token_ids: list[int] | None = None
    global_batch_size: int = 256
    save_interval: int = 20

    rollout_num_gpus_per_engine: int = 4
    # miles' own router rather than sglang-router. The sglang path never binds its
    # port here ("Server at 127.0.0.1:4077 not ready after 30s", a hardcoded
    # timeout in miles' wait_for_server_ready), and this is the path the gym
    # already exercises -- Kimi sets it too, and patch_sglang_abort special-cases it.
    use_miles_router: bool = True
    # 0.25, not upstream's 0.55. Both engines stay resident here (see the offload
    # note below) and upstream's share does not leave room: a failed step measured
    # ~64 GiB of training state against SGLang's ~68 GiB on a 139.8 GiB H200, and
    # the optimizer step OOMed reaching for another 968 MiB. At 0.25 SGLang takes
    # ~35 GiB -- still well clear of the ~13 GiB of TP4-sharded weights -- which
    # leaves roughly 40 GiB of headroom for activations.
    sglang_mem_fraction_static: float = 0.25
    # Gemma-4's global head_dim=512 exceeds FlashAttention's 256 cap.
    sglang_attention_backend: str = "triton"
    sglang_moe_runner_backend: str = "triton"
    sglang_disable_custom_all_reduce: bool = True
    sglang_disable_cuda_graph: bool = True
    sglang_disable_overlap_schedule: bool = True
    sglang_disable_radix_cache: bool = True
    sglang_max_running_requests: int | None = None
    # Both stay resident, as upstream has it. Offloading instead is what you would
    # reach for first, but SGLang's memory-saver path dies with "CUDA error: an
    # illegal memory access was encountered" during the training step, so the
    # memory has to come from the reservation above rather than from offload.
    no_offload_train: bool = True
    no_offload_rollout: bool = True
    # Off, despite upstream's script setting it. It sets sglang's
    # ``enable_return_routed_experts``, and that capturer reads
    # ``hf_text_config.num_experts_per_tok`` -- an attribute Gemma-4 does not have
    # (its text config calls it ``top_k_experts``), so every SGLang scheduler dies
    # with AttributeError before serving a token. Costs MoE routing replay, i.e. a
    # little extra train/inference mismatch; re-enable once sglang's
    # RoutedExpertsCapturer reads the topk off Gemma-4's config.
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

    def _for_dataset(self, dataset: "DatasetConfig | None") -> MilesRecipe:
        if not _has_images(dataset):
            return self
        # Keyed on what the caller passed, not on how the value compares to the
        # text default: an explicit `n_samples_per_prompt=8` matches this recipe's
        # default, and reading that as unset would silently run the vision default.
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
