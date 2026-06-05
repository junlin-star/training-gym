from __future__ import annotations

from modal_training_gym.train_recipes.miles_recipe.recipe import MilesConfig


class Qwen3_6_35B_A3B_LoRA_Recipe(MilesConfig):
    """Miles LoRA recipe for Qwen3.6-35B-A3B (35B total, ~3B active MoE).

    256-expert MoE model with 8 active per token. Uses the pre-built
    model script from the miles container for architecture args.
    """

    miles_model_script: str = "scripts/models/qwen3.6-35B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"
    skip_eval_before_train: bool = True

    lora_rank: int | None = 32
    lora_alpha: int | None = 32
    lora_dropout: float | None = 0.0
    target_modules: str | None = (
        "q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj"
    )
    experts_shared_outer_loras: bool = True
    lora_base_cpu_backup: bool = True
    no_gradient_accumulation_fusion: bool = True
    sglang_lora_backend: str | None = "triton"
    sglang_lora_use_virtual_experts: bool = True
    use_tis: bool = True
