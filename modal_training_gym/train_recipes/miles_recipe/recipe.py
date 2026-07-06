from collections.abc import Callable
from dataclasses import field
from typing import Any

import modal
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.base import (
    BaseTrainRecipe,
    RecipeType,
    # Re-exported for backwards compatibility (e.g. frameworks/miles/launcher.py
    # imports the volume paths from this module).
    CHECKPOINTS_PATH as CHECKPOINTS_PATH,
    DATA_PATH as DATA_PATH,
    HF_CACHE_PATH as HF_CACHE_PATH,
    JSON_CONFIG_FIELDS as JSON_CONFIG_FIELDS,
)
from modal_training_gym.train_recipes.gpu_allocation import (
    resolve_gpu_allocation,
)

_MILES_SKIP = {
    "recipe_type",
    "environment",
    "async_mode",
    "miles_model_script",
    "source_hf_checkpoint",
    "megatron_conversion_hf_checkpoint",
    "docker_image",
    "gpu_type",
    "memory",
    "cloud",
    "region",
    "name",
    "app_tags",
    "image_overlay",
    "image_run_commands",
    "image_env",
    "local_miles",
    "patch_files",
    "wandb",
    "custom_rm_function",
    "custom_generate_function",
}

YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class MilesConfig(BaseTrainRecipe):
    """Training Gym config for Miles training on Modal.

    Non-launcher attributes become Miles CLI flags. This intentionally mirrors
    the standalone Miles guide while adding Training Gym model, dataset, wandb,
    and custom reward wiring.
    """

    recipe_type: RecipeType = RecipeType.MILES

    docker_image: str = "radixark/miles:dev-202606111336"
    gpu_type: str = "H100"
    memory: tuple[int, int] | None = None
    cloud: str | None = None
    region: str | None = None
    name: str = ""
    app_tags: dict = field(default_factory=dict)
    image_overlay: Callable[[modal.Image], modal.Image] | None = None
    image_run_commands: list[str] = field(default_factory=list)
    image_env: dict[str, str] = field(default_factory=dict)
    local_miles: str | None = None
    patch_files: list[str] = field(default_factory=list)

    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )
    async_mode: bool = False
    miles_model_script: str = ""
    source_hf_checkpoint: str | None = None
    megatron_conversion_hf_checkpoint: str | None = None
    wandb: WandbConfig | None = None

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int | None = None
    colocate: bool = True
    use_critic: bool = False
    critic_num_nodes: int | None = None
    critic_num_gpus_per_node: int | None = None

    hf_checkpoint: str = ""
    save: str = str(CHECKPOINTS_PATH)
    load: str = ""
    ref_load: str = ""
    megatron_to_hf_mode: str = "bridge"
    save_interval: int = 10

    num_rollout: int = 1
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    rollout_shuffle: bool = True
    rollout_num_gpus_per_engine: int = 1

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    sequence_parallel: bool = False
    train_backend: str = "megatron"

    global_batch_size: int = 16
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer: str = "adam"

    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    target_modules: str | None = None
    experts_shared_outer_loras: bool = False
    lora_base_cpu_backup: bool = False
    no_gradient_accumulation_fusion: bool = False
    sglang_lora_backend: str | None = None
    sglang_lora_use_virtual_experts: bool = False
    use_tis: bool = False

    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.0
    entropy_coef: float = 0.0
    use_kl_loss: bool = False
    rm_type: str | None = None

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    accumulate_allreduce_grads_in_fp32: bool = True
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 9216

    eval_interval: int | None = None
    n_samples_per_eval_prompt: int = 4
    eval_max_response_len: int = 16384
    eval_top_p: float = 1.0
    eval_config: dict | str | None = None
    skip_eval_before_train: bool = False

    custom_config_path: dict | str | None = None
    sglang_config: dict | str | None = None
    sglang_mem_fraction_static: float = 0.75
    apply_chat_template_kwargs: str | dict = ""

    custom_rm_function: Callable | None = None
    custom_generate_function: Callable | None = None

    # ── Validators ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_gpu_allocation(self) -> "MilesConfig":
        resolve_gpu_allocation(self)
        return self

    # ── Container → miles flag converters ────────────────────────────────────

    @staticmethod
    def _model_to_fields(m: ModelConfig) -> dict[str, Any]:
        fields: dict[str, Any] = {"hf_checkpoint": m.model_path or m.model_name}
        arch = getattr(m, "architecture", None)
        if arch is None:
            return fields
        fields.update(
            {
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "ffn_hidden_size": arch.ffn_hidden_size,
                "num_attention_heads": arch.num_attention_heads,
                "group_query_attention": arch.group_query_attention,
                "num_query_groups": arch.num_query_groups,
                "kv_channels": arch.kv_channels,
                "vocab_size": arch.vocab_size,
                "normalization": arch.normalization,
                "norm_epsilon": arch.norm_epsilon,
                "swiglu": arch.swiglu,
                "disable_bias_linear": arch.disable_bias_linear,
                "qk_layernorm": arch.qk_layernorm,
                "untie_embeddings_and_output_weights": arch.untie_embeddings_and_output_weights,
                "use_rotary_position_embeddings": arch.use_rotary_position_embeddings,
                "rotary_base": arch.rotary_base,
            }
        )
        optional = {
            "num_experts": arch.num_experts,
            "moe_ffn_hidden_size": arch.moe_ffn_hidden_size,
            "moe_shared_expert_intermediate_size": arch.moe_shared_expert_intermediate_size,
            "moe_router_topk": arch.moe_router_topk,
            "moe_router_score_function": arch.moe_router_score_function,
            "moe_token_drop_policy": arch.moe_token_drop_policy,
            "moe_router_dtype": arch.moe_router_dtype,
            "moe_aux_loss_coeff": arch.moe_aux_loss_coeff,
            "spec": arch.megatron_spec,
            "rotary_percent": arch.rotary_percent
            if arch.rotary_percent != 1.0
            else None,
        }
        fields.update({k: v for k, v in optional.items() if v not in (None, "", 0)})
        for key in (
            "moe_grouped_gemm",
            "moe_shared_expert_gate",
            "moe_permute_fusion",
            "apply_layernorm_1p",
            "use_gated_attention",
            "attention_output_gate",
        ):
            if getattr(arch, key):
                fields[key] = True
        return fields

    def _fields(
        self,
        dataset: DatasetConfig | None = None,
        model: ModelConfig | None = None,
    ) -> dict[str, Any]:
        import dataclasses as _dc

        fields: dict[str, Any] = {}
        for f in _dc.fields(self):
            fields[f.name] = getattr(self, f.name)
        if model is not None:
            for k, v in self._model_to_fields(model).items():
                if k == "hf_checkpoint" and fields.get(k):
                    continue
                fields[k] = v
        if dataset is not None:
            fields.update(self._dataset_to_fields(dataset))
        if self.wandb is not None:
            fields.update(self._wandb_to_fields(self.wandb))
        return {k: v for k, v in fields.items() if k not in _MILES_SKIP}

    def download_model(self) -> None:
        from modal_training_gym.frameworks.miles.modal_helpers.utils import (
            resolve_checkpoint_ref,
        )

        ref = self.source_hf_checkpoint or self.hf_checkpoint
        if ref:
            resolve_checkpoint_ref(ref, local_files_only=False)

    def post_process_model(self) -> None:
        pass

    def post_process_data(self) -> None:
        pass
