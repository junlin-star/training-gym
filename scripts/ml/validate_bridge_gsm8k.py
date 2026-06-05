"""Validate MoE bridge weight conversion fix with GSM8K.

Runs Qwen3.6-35B-A3B on GSM8K with bridge mode to confirm that the
gate_up_proj transpose fix produces coherent rollout outputs (not
gibberish/multilingual garbage) and non-zero rewards.

Usage:
    uv run modal run --env joy-agent-dev -d scripts/ml/validate_bridge_gsm8k.py
"""

from modal_training_gym import (
    HuggingFaceDataset,
    Qwen3_6_35B,
    TrainConfig,
)
from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_35b_Recipe


class GSM8KDataset(HuggingFaceDataset):
    hf_repo = "openai/gsm8k"
    hf_config = "main"
    hf_split = "train"
    input_column = "question"
    output_column = "answer"
    output_format = "parquet"
    apply_chat_template = True
    n_rows = 200


dataset = GSM8KDataset()

training_run = TrainConfig(
    model=Qwen3_6_35B(),
    dataset=dataset,
    recipe=Qwen3_6_35b_Recipe(
        rm_type="deepscaler",
        n_samples_per_prompt=4,
        sglang_mem_fraction_static=0.75,
        sglang_max_running_requests=512,
        eval_max_response_len=4096,
        n_samples_per_eval_prompt=4,
        num_rollout=50,
        rollout_batch_size=8,
        # 2 nodes (16 GPUs) — 1 node OOMs during backward pass
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        # TP2 x EP8 = 16 GPUs
        tensor_model_parallel_size=2,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        # SGLang: 4 engines x 4 GPUs each = 16 GPUs (colocated)
        rollout_num_gpus_per_engine=4,
        sglang_dp_size=4,
        sglang_ep_size=4,
        # Stop at Qwen3 EOS tokens so responses terminate cleanly
        # <|im_end|>=151645, <|endoftext|>=151643
        rollout_stop_token_ids=[151645, 151643],
        environment={
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        },
    ),
)

print("Starting GSM8K validation run...")
train_result = training_run.train()
print(f"Training run id: {train_result.training_run_id}")
