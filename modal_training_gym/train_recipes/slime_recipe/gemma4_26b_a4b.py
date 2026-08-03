"""Gemma-4-26B-A4B GRPO recipe (1x8xH100), text-only or vision-language."""

from dataclasses import MISSING, field
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

# Build-time shims for upstream gaps that block a bridge-mode VL model on slime. Each
# patch's docstring has the traceback it fixes; report upstream and drop once fixed.
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


def _declared_default(cls: type, name: str) -> Any:
    """The dataclass default for ``name``, i.e. the value that means "unset"."""
    f = cls.__dataclass_fields__[name]
    if f.default_factory is not MISSING:
        return f.default_factory()
    return f.default


# Per-mode values, applied by __post_init__ to the fields left at their default.
# Callables are resolved there so the VL patches are only encoded when needed.
_TEXT_MODE: dict[str, Any] = {
    "slime_model_script": "scripts/models/gemma4-26B-A4B.sh",
    "megatron_to_hf_mode": "raw",
    "pipeline_model_parallel_size": 2,
    "attention_backend": "flash",
    "use_dynamic_batch_size": True,
    "num_rollout": 2,
    "rollout_batch_size": 4,
    "n_samples_per_prompt": 4,
    "rollout_max_response_len": 512,
    "rollout_temperature": 0.8,
    "rollout_top_p": 1.0,
    "global_batch_size": 16,
    "save_interval": 20,
    "sglang_cuda_graph_max_bs": 1,
    "sglang_max_running_requests": 4,
    "rollout_stop_token_ids": [1, 106],
}

# No slime_model_script and no attention_backend; see the class docstring.
_VISION_MODE: dict[str, Any] = {
    "megatron_to_hf_mode": "bridge",
    "pipeline_model_parallel_size": 1,
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
    "rollout_stop_token_ids": [1, 106, 50],
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26B_A4B_Recipe(SlimeRecipe):
    """Gemma-4-26B-A4B (26B-A4B MoE) on 1×8×H100 with TP2/CP1/EP2, colocated GRPO.

    ``vision=True`` (matching ``Gemma4_26B_A4B(vision=True)``) trains the vision-language
    model instead of the text-only decoder. Fields the two modes disagree on default to
    ``None`` here and are filled from ``_TEXT_MODE`` / ``_VISION_MODE``, so anything set
    explicitly wins — including across ``_merge_recipe``, which reconstructs the recipe.

    What vision mode changes and why:

    * ``megatron_to_hf_mode="bridge"`` with ``slime_model_script`` left unset — slime
      checks ``--custom-model-provider-path`` before the bridge branch, so a model script
      would build the text-only ``GPTModel`` and train while ignoring every image.
    * ``use_dynamic_batch_size=False`` + ``qkv_format="bshd"`` + ``micro_batch_size=1`` —
      ``Gemma4VLModel.forward`` builds a dense mask that is only valid on one sequence,
      and image-token counts vary with aspect ratio. The launcher enforces this
      (``model.requires_bshd``).
    * no ``attention_backend`` (text mode pins ``"flash"``) — flash cannot honour that
      mask, so pinning it makes TE reject every backend.
    * ``pipeline_model_parallel_size=1`` — keeps the vision tower and embedding on one
      pipeline stage.
    * ``freeze_params_name_list`` — RL only updates the language backbone. Patterns are
      re.search-ed against the Megatron parameter names the bridge assigns.
    """

    # Must match the attached model's own flag; not a slime CLI flag.
    vision: bool = False

    gpu_type: str = "H100"
    colocate: bool = True
    hf_checkpoint: str = "google/gemma-4-26B-A4B-it"
    # Model overflows container disk, so reserve 1 TiB.
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    rollout_num_gpus_per_engine: int = 8

    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int | None = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 2
    expert_tensor_parallel_size: int = 1
    attention_backend: str | None = None

    num_rollout: int | None = None
    rollout_batch_size: int | None = None
    rollout_max_response_len: int | None = None
    rollout_temperature: float | None = None
    rollout_top_p: float | None = None
    rollout_top_k: int | None = None
    # Colocated 26B MoE (plus the ViT in vision mode): leave the actor room.
    sglang_mem_fraction_static: float = 0.20
    sglang_cuda_graph_max_bs: int | None = None

    n_samples_per_prompt: int | None = None
    global_batch_size: int | None = None
    use_dynamic_batch_size: bool | None = None
    max_tokens_per_gpu: int = 2048
    num_steps_per_rollout: int = 1
    balance_data: bool = True
    calculate_per_token_loss: bool = True
    entropy_coef: float = 0.001
    loss_mask_type: str = "gemma4"

    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    save_interval: int | None = None

    def __post_init__(self) -> None:
        for name, value in (_VISION_MODE if self.vision else _TEXT_MODE).items():
            if getattr(self, name) != _declared_default(type(self), name):
                continue
            object.__setattr__(self, name, value() if callable(value) else value)

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        super().validate_model_parallelism(model)
        model_vision = getattr(model, "vision", False)
        if model_vision != self.vision:
            raise TrainingGymConfigError(
                f"{type(self).__name__}(vision={self.vision}) is attached to "
                f"{type(model).__name__}(vision={model_vision}). The two must agree: "
                "vision mode trains the VL model through megatron-bridge, text mode "
                "hands the same checkpoint to slime's text-only model script. Pass the "
                "same vision= to both."
            )

    def _fields(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfig | None" = None,
    ) -> dict[str, Any]:
        fields = super()._fields(dataset=dataset, model=model)
        fields.pop("vision", None)
        return fields
