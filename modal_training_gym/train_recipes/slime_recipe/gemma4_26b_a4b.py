from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26B_A4B_Recipe(SlimeRecipe):
    """Gemma-4-26B-A4B (26B-A4B MoE) on 1×8×H100 with TP2/PP2/CP1/EP2, colocated GRPO."""

    gpu_type: str = "H100"
    slime_model_script: str = "scripts/models/gemma4-26B-A4B.sh"
    hf_checkpoint: str = "google/gemma-4-26B-A4B-it"
    colocate: bool = True
    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    rollout_num_gpus_per_engine: int = 8

    num_rollout: int = 3000
    rollout_batch_size: int = 4
    rollout_max_response_len: int = 512
    rollout_temperature: float = 0.8
    sglang_mem_fraction_static: float = 0.20

    save_interval: int = 20

    actor_num_nodes: int = 1
    n_samples_per_prompt: int = 4
    global_batch_size: int = 16
    lr: float = 1e-6
    max_tokens_per_gpu: int = 2048

    # MoE parallelism
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 2
    expert_tensor_parallel_size: int = 1
    attention_backend: str = "flash"

    # RL algorithm (GRPO)
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    entropy_coef: float = 0.001

    # Optimizer
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # Rollout sglang
    sglang_cuda_graph_max_bs: int = 1
    sglang_max_running_requests: int | None = 4

    # Training
    num_steps_per_rollout: int = 1
    balance_data: bool = True
    use_dynamic_batch_size: bool = True
    calculate_per_token_loss: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    loss_mask_type: str = "gemma4"

    # Checkpointing / weight sync
    megatron_to_hf_mode: str = "raw"
    eval_interval: int | None = None
