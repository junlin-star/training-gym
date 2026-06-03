from __future__ import annotations

from modal_training_gym.train_recipes.miles_recipe.recipe import MilesConfig


class _KimiK2Recipe(MilesConfig):
    gpu_type: str = "H200"
    memory: tuple[int, int] = (1024, int(2 * 1024 * 1024))
    image_run_commands: list[str] = [
        "rm -rf /root/.cache/huggingface 2>/dev/null || true",
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
    ]
    image_env: dict[str, str] = {
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
    }
    miles_model_script: str = "scripts/models/kimi-k2-thinking.sh"
    environment: dict[str, str] = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NCCL_TIMEOUT": "3600",
        "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
        "OPEN_TRAINING_INT4_GROUP_SIZE": "32",
    }

    actor_num_nodes: int = 16
    actor_num_gpus_per_node: int = 8
    colocate: bool = True
    use_miles_router: bool = True
    skip_eval_before_train: bool = True
    update_weight_buffer_size: int = 4 * 512 * 1024 * 1024

    prompt_data: str = "/data/dapo-math-17k/dapo-math-17k.jsonl"
    input_key: str = "prompt"
    label_key: str = "label"
    apply_chat_template: bool = True
    rollout_shuffle: bool = True
    balance_data: bool = True
    rm_type: str = "deepscaler"

    num_rollout: int = 20
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 16384
    rollout_temperature: float = 1.0
    sglang_cuda_graph_bs: list[int] = [1, 2, 4, 8] + list(range(16, 129, 8))
    global_batch_size: int = 256

    advantage_estimator: str = "grpo"
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    optimizer: str = "adam"
    lr: float = 1e-5
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    use_distributed_optimizer: bool = True

    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = (
        "q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj"
    )
    experts_shared_outer_loras: bool = True
    lora_base_cpu_backup: bool = True
    no_gradient_accumulation_fusion: bool = True
    sglang_lora_backend: str = "triton"
    sglang_lora_use_virtual_experts: bool = True
    use_tis: bool = True

    train_backend: str = "megatron"
    tensor_model_parallel_size: int = 8
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 8
    expert_model_parallel_size: int = 64
    expert_tensor_parallel_size: int = 1
    decoder_last_pipeline_num_layers: int = 30

    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    attention_backend: str = "flash"
    no_check_for_nan_in_loss_and_grad: bool = True

    rollout_num_gpus_per_engine: int = 8
    sglang_mem_fraction_static: float = 0.7
    sglang_ep_size: int = 8
    sglang_server_concurrency: int = 1024
    use_rollout_routing_replay: bool = True


class _KimiK2FullParamRecipe(_KimiK2Recipe):
    actor_num_nodes: int = 32
    lr: float = 1e-6

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

    pipeline_model_parallel_size: int = 8
    context_parallel_size: int = 4
    expert_model_parallel_size: int = 32
    decoder_last_pipeline_num_layers: int = 5


class Kimi_K2_5_Recipe(_KimiK2Recipe):
    """Kimi-K2.5 LoRA on 16x8xH200 with Miles INT4 rollout and BF16 reference load."""

    source_hf_checkpoint: str = "moonshotai/Kimi-K2.5"
    hf_checkpoint: str = "/checkpoints/Kimi-K2.5-int4"
    ref_load: str = "/checkpoints/Kimi-K2.5-bf16"


class Kimi_K2_6_Recipe(_KimiK2Recipe):
    """Kimi-K2.6 LoRA on 16x8xH200 with Miles INT4 rollout and BF16 reference load."""

    source_hf_checkpoint: str = "moonshotai/Kimi-K2.6"
    hf_checkpoint: str = "/checkpoints/Kimi-K2.6-int4"
    ref_load: str = "/checkpoints/Kimi-K2.6-bf16"


class Kimi_K2_5_FullParam_Recipe(_KimiK2FullParamRecipe):
    """Kimi-K2.5 full-param on 32x8xH200 with Miles INT4 rollout and BF16 reference load."""

    source_hf_checkpoint: str = "moonshotai/Kimi-K2.5"
    hf_checkpoint: str = "/checkpoints/Kimi-K2.5-int4"
    ref_load: str = "/checkpoints/Kimi-K2.5-bf16"


class Kimi_K2_6_FullParam_Recipe(_KimiK2FullParamRecipe):
    """Kimi-K2.6 full-param on 32x8xH200 with Miles INT4 rollout and BF16 reference load."""

    source_hf_checkpoint: str = "moonshotai/Kimi-K2.6"
    hf_checkpoint: str = "/checkpoints/Kimi-K2.6-int4"
    ref_load: str = "/checkpoints/Kimi-K2.6-bf16"
