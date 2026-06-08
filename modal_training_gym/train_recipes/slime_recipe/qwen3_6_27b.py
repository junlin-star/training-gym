from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_27b_Recipe(SlimeRecipe):
    """Qwen3.6-27B dense hybrid model on 1×8×H100 with TP4×PP2, colocated GRPO."""

    gpu_type: str = "H100"
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )
    colocate: bool = True

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    # ── Parallelism ───────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    # ── Rollout ───────────────────────────────────────────────────────────
    rollout_num_gpus_per_engine: int = 4
    num_rollout: int = 1
    rollout_batch_size: int = 32
    rollout_max_response_len: int = 8192
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.75
    sglang_disable_custom_all_reduce: bool = True

    # ── Training ──────────────────────────────────────────────────────────
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    lr: float = 1e-6
    max_tokens_per_gpu: int = 8192
    calculate_per_token_loss: bool = True
    balance_data: bool = True
    accumulate_allreduce_grads_in_fp32: bool = False
    use_distributed_optimizer: bool = True

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Attention ─────────────────────────────────────────────────────────
    attention_backend: str = "flash"

    # ── Checkpointing / eval ──────────────────────────────────────────────
    save_interval: int = 20
    eval_interval: int | None = 20
    n_samples_per_eval_prompt: int = 4
    eval_max_response_len: int = 4096
