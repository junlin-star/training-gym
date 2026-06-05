from __future__ import annotations

from modal_training_gym.train_recipes.miles_recipe.recipe import MilesConfig


class Qwen3_6_35B_A3B_Recipe(MilesConfig):
    """Miles full-parameter recipe for Qwen3.6-35B-A3B (35B total, ~3B active MoE).

    256-expert MoE model with 8 active per token. Uses the pre-built
    model script from the miles container for architecture args.
    """

    miles_model_script: str = "scripts/models/qwen3.6-35B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"
    memory: tuple[int, int] = (1024, 2 * 1024 * 1024)
    ephemeral_disk: int = 2048000
    skip_eval_before_train: bool = True
    no_gradient_accumulation_fusion: bool = True
    use_tis: bool = True

    # Pre-convert HF → torch_dist to bypass the Megatron Bridge crash
    # with EP>1 ('NoneType' has no attribute 'megatron_module').
    megatron_to_hf_mode: str = ""
    ref_load: str = "/checkpoints/Qwen3.6-35B-A3B-torch-dist-ep8"
    mtp_num_layers: int = 0

    # Cluster topology — disaggregated: 1 train node + 1 rollout node
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    colocate: bool = False
    rollout_num_gpus: int = 8

    # Training parallelism — EP=8 on 8 GPUs (TP*EP must <= world_size)
    train_backend: str = "megatron"
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    use_distributed_optimizer: bool = True
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    attention_backend: str = "flash"
    no_check_for_nan_in_loss_and_grad: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    max_tokens_per_gpu: int = 2048

    # Reward model — rule-based math verifier
    rm_type: str = "deepscaler"

    # Rollout (SGLang)
    rollout_num_gpus_per_engine: int = 8
    sglang_ep_size: int = 8
    sglang_mem_fraction_static: float = 0.7
