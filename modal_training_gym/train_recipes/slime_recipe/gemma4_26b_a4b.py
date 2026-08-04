"""Gemma-4-26B-A4B GRPO recipe (1x8xH100), text-only or vision-language."""

from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfig

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


def _has_images(dataset: "DatasetConfig | None") -> bool:
    return "image" in (getattr(dataset, "multimodal_keys", None) or {})


# Applied over the text-only defaults below on an image dataset, to the fields the
# caller left alone. Callables are resolved at apply time so the VL patches are only
# encoded when they're used.
_VISION_MODE: dict[str, Any] = {
    "slime_model_script": "",
    "megatron_to_hf_mode": "bridge",
    "pipeline_model_parallel_size": 1,
    "attention_backend": None,
    "use_dynamic_batch_size": False,
    "extra_config": {"qkv_format": "bshd", "micro_batch_size": 1},
    "freeze_params_name_list": ["vision_tower", "embed_vision"],
    "image_run_commands": _vl_image_run_commands,
    "num_rollout": 15,
    "rollout_batch_size": 8,
    "n_samples_per_prompt": 8,
    "rollout_max_response_len": 256,
    "rollout_temperature": 1.0,
    # From generation_config.json, which slime's /generate path does not apply.
    "rollout_top_p": 0.95,
    "rollout_top_k": 64,
    "global_batch_size": 64,
    "save_interval": 10,
    "sglang_cuda_graph_max_bs": 8,
    "sglang_max_running_requests": 8,
    # generation_config.json's eos_token_id: <eos>, <turn|>, <|tool_response>.
    "rollout_stop_token_ids": [1, 106, 50],
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26B_A4B_Recipe(SlimeRecipe):
    """Gemma-4-26B-A4B (26B-A4B MoE) on 1×8×H100 with TP2/CP1/EP2, colocated GRPO.

    One checkpoint, two modes. The fields below train the MoE text decoder through
    slime's model script; an image dataset (``MultimodalDataset(modality="image")``)
    trains the whole vision-language model through megatron-bridge instead —
    ``_for_dataset`` swaps in the vision-mode values for every field the caller left at
    its default. Nothing has to be passed by hand, and explicit values still win.

    What vision mode changes and why:

    * ``megatron_to_hf_mode="bridge"`` with ``slime_model_script`` cleared — slime
      checks ``--custom-model-provider-path`` before the bridge branch, so a model
      script would build the text-only ``GPTModel`` and train while ignoring every
      image. The bridge resolves the checkpoint to a ``Gemma4VLModel`` (vision tower
      + projector + language model) and exports through ``bridge.save_hf_pretrained``
      so the ViT survives the round trip.
    * ``use_dynamic_batch_size=False`` + ``qkv_format="bshd"`` + ``micro_batch_size=1``
      — ``Gemma4VLModel.forward`` builds a dense mask that is only valid on one
      sequence, and image-token counts vary with aspect ratio, so a ragged
      micro-batch cannot be reconciled.
    * no ``attention_backend`` (text mode pins ``"flash"``) — flash cannot honour that
      mask, so pinning it makes TE reject every backend.
    * ``pipeline_model_parallel_size=1`` — keeps the vision tower and embedding on one
      pipeline stage.
    * ``freeze_params_name_list`` — RL only updates the language backbone. Patterns are
      re.search-ed against the Megatron parameter names the bridge assigns.
    """

    gpu_type: str = "H100"
    colocate: bool = True
    hf_checkpoint: str = "google/gemma-4-26B-A4B-it"
    slime_model_script: str = "scripts/models/gemma4-26B-A4B.sh"
    # Model overflows container disk, so reserve 1 TiB.
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    rollout_num_gpus_per_engine: int = 8

    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 2
    expert_tensor_parallel_size: int = 1
    attention_backend: str | None = "flash"

    num_rollout: int = 2
    rollout_batch_size: int = 4
    rollout_max_response_len: int = 512
    rollout_temperature: float = 0.8
    rollout_top_k: int | None = None
    rollout_stop_token_ids: list[int] | None = field(default_factory=lambda: [1, 106])
    # Colocated 26B MoE (plus the ViT in vision mode): leave the actor room.
    sglang_mem_fraction_static: float = 0.20
    sglang_cuda_graph_max_bs: int = 1
    sglang_max_running_requests: int | None = 4

    n_samples_per_prompt: int = 4
    global_batch_size: int = 16
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 2048
    num_steps_per_rollout: int = 1
    balance_data: bool = True
    calculate_per_token_loss: bool = True
    entropy_coef: float = 0.001
    loss_mask_type: str = "gemma4"

    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    megatron_to_hf_mode: str = "raw"
    save_interval: int = 20

    def _for_dataset(self, dataset: "DatasetConfig | None") -> SlimeRecipe:
        if not _has_images(dataset):
            return self
        # Keyed on what the caller passed, not on how the value compares to the
        # text default: an explicit `num_rollout=2` matches this recipe's default,
        # and reading that as unset would silently run the vision default of 15.
        explicit = self.explicit_fields
        vision = {
            name: value() if callable(value) else value
            for name, value in _VISION_MODE.items()
            if name != "extra_config" and name not in explicit
        }
        # extra_config may already hold hook paths resolved at construction, so merge
        # into it (keys the caller set win) instead of replacing it wholesale.
        vision["extra_config"] = {
            **_VISION_MODE["extra_config"],
            **(self.extra_config or {}),
        }
        resolved = replace(self, **vision)
        # `replace` marks every field explicit, so carry the real set forward and
        # resolving twice stays a no-op.
        object.__setattr__(resolved, "_explicit_fields", explicit | vision.keys())
        return resolved

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        super().validate_model_parallelism(model)
        if (
            self.megatron_to_hf_mode == "bridge"
            and self.pipeline_model_parallel_size != 1
        ):
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs pipeline_model_parallel_size=1 in vision "
                "mode: the bridge keeps the vision tower and embedding on one pipeline "
                f"stage, so a split only fails once Megatron builds the model. Got "
                f"{self.pipeline_model_parallel_size}."
            )
