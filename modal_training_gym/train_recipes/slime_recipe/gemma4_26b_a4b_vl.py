"""Gemma-4-26B-A4B vision-language GRPO recipe (1x8xH100)."""

from dataclasses import field
from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.patches import encode_patch
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe

# Build-time shims for upstream gaps that block a bridge-mode VL model on slime: THD
# packing vs the VL attention mask, an unset pg_collection, the forward's tuple return,
# and a weight sync that withholds registered buffers. Each patch's docstring has the
# traceback it fixes; report upstream and drop once fixed.
_VL_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "slime"
    / "modal_helpers"
    / "patches"
    / "model_specific_patches"
    / "gemma4_vl"
)
_VL_PATCHES = (
    "patch_gemma4_vl_packed_seq",
    "patch_gemma4_vl_buffer_sync",
    "patch_gemma4_vl_pg_collection",
    "patch_gemma4_vl_forward_return",
)


def _vl_image_run_commands() -> list[str]:
    return [
        f"echo {encode_patch(name, _VL_PATCH_DIR)} | base64 -d | python3"
        for name in _VL_PATCHES
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26B_A4B_VL_Recipe(SlimeRecipe):
    """Gemma-4-26B-A4B vision-language GRPO on 1×8×H100, colocated.

    ``megatron_to_hf_mode="bridge"`` resolves the checkpoint to a ``Gemma4VLModel``
    (vision tower + projector + language model) and exports through
    ``bridge.save_hf_pretrained`` so the ViT survives the round trip.

    ``slime_model_script`` is deliberately left unset: slime checks
    ``--custom-model-provider-path`` before the bridge branch, so setting it would
    build the text-only ``GPTModel`` and train while ignoring every image.
    """

    gpu_type: str = "H100"
    colocate: bool = True

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    tensor_model_parallel_size: int = 2
    # PP=1 (text recipe uses 2) keeps the vision tower and embedding on one stage.
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 2
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = True
    # attention_backend stays unset (the text recipe pins "flash"): flash cannot
    # honour the mask Gemma4VLModel builds, so pinning it makes TE reject every
    # backend, while "auto" picks one that supports it.

    rollout_num_gpus_per_engine: int = 8
    # Model overflows container disk, so reserve 1 TiB (as the text recipe does).
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    # ── The VL wiring ───────────────────────────────────────────────────────
    megatron_to_hf_mode: str = "bridge"
    # Freeze the vision tower; RL only updates the language backbone. Patterns are
    # re.search-ed against the Megatron parameter names the bridge assigns.
    freeze_params_name_list: list[str] | None = field(
        default_factory=lambda: ["vision_tower", "embed_vision"]
    )
    # Padded single-sample micro-batches; see Gemma4_26B_A4B_VL.requires_bshd.
    use_dynamic_batch_size: bool = False
    extra_config: dict | None = field(
        default_factory=lambda: {"qkv_format": "bshd", "micro_batch_size": 1}
    )
    image_run_commands: list[str] = field(default_factory=_vl_image_run_commands)

    # ── Rollout ─────────────────────────────────────────────────────────────
    num_rollout: int = 15
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 256
    rollout_temperature: float = 1.0
    # From generation_config.json, which slime's /generate path (unlike SGLang's chat
    # endpoint) does not apply: without top_k every rollout rambled to the token cap.
    rollout_top_p: float = 0.95
    rollout_top_k: int = 64
    # Colocated 26B MoE + ViT: leave the actor room (text-only runs use ~0.78).
    sglang_mem_fraction_static: float = 0.20
    sglang_cuda_graph_max_bs: int = 8
    sglang_max_running_requests: int | None = 8

    # ── Training ────────────────────────────────────────────────────────────
    global_batch_size: int = 64
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    # Covers one screenshot (~264 image tokens) + instruction + a 256-token response;
    # cp=1 means this budget must fit the longest sequence.
    max_tokens_per_gpu: int = 2048
    num_steps_per_rollout: int = 1
    balance_data: bool = True
    calculate_per_token_loss: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    loss_mask_type: str = "gemma4"

    # ── RL algorithm (GRPO) ─────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    entropy_coef: float = 0.001

    # ── Optimizer ───────────────────────────────────────────────────────────
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Checkpointing ───────────────────────────────────────────────────────
    save_interval: int = 10
    eval_interval: int | None = None

    def __post_init__(self) -> None:
        if self.rollout_stop_token_ids is None:
            # generation_config.json's eos_token_id: <eos>, <turn|>, <|tool_response>.
            object.__setattr__(self, "rollout_stop_token_ids", [1, 106, 50])
