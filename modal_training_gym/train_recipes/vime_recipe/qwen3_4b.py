from modal_training_gym.train_recipes.vime_recipe.recipe import VimeConfig


class Qwen3_4b_Vime_Recipe(VimeConfig):
    """Qwen3-4B GRPO on a single 8×H100 node, colocated, vLLM rollout.

    Mirrors the slime ``Qwen3_4b_Recipe`` defaults but routes the rollout
    through vime's vLLM backend. Vime upstream ships ``scripts/run-qwen3-4B.sh``;
    these defaults track it. Bring your own ``custom_rm_function`` (and optionally
    ``wandb=WandbConfig(...)``).
    """

    gpu_type: str = "H100"
    colocate: bool = True
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus_per_engine: int = 1

    num_rollout: int = 1
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0

    save_interval: int = 10
    megatron_to_hf_mode: str = "bridge"

    lr: float = 5e-7
    max_tokens_per_gpu: int = 8192
    eval_interval: int | None = 10
    eval_max_response_len: int = 4096

    vllm_gpu_memory_utilization: float = 0.7
