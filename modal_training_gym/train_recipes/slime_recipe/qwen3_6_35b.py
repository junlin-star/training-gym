from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_35b_Recipe(SlimeRecipe):
    """Qwen3.6-35B-A3B (MoE) on 1×8×H100 with TP2/PP2/CP2/EP4."""

    gpu_type: str = "H100"
    slime_model_script: str = "scripts/models/qwen3.5-35B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    colocate: bool = True
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    # ── Parallelism ───────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 2
    expert_model_parallel_size: int = 4
    expert_tensor_parallel_size: int = 1

    # ── Rollout ───────────────────────────────────────────────────────────
    rollout_num_gpus_per_engine: int = 4
    num_rollout: int = 1
    rollout_batch_size: int = 8
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.75
    sglang_enable_dp_attention: bool = True
    sglang_dp_size: int | None = 4
    sglang_ep_size: int | None = 4
    sglang_enable_dp_lm_head: bool = True
    sglang_cuda_graph_bs: list[int] | None = None
    sglang_max_running_requests: int | None = 512

    # ── Training ──────────────────────────────────────────────────────────
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    lr: float = 1e-6
    max_tokens_per_gpu: int = 8192
    calculate_per_token_loss: bool = True
    balance_data: bool = True

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Attention ─────────────────────────────────────────────────────────
    attention_backend: str = "flash"

    # ── Checkpointing / eval ──────────────────────────────────────────────
    megatron_to_hf_mode: str = ""
    ref_load: str = "/checkpoints/Qwen3.6-35B-A3B_torch_dist_tp2pp2"
    save_interval: int = 20
    eval_interval: int | None = None
    eval_max_response_len: int = 4096
