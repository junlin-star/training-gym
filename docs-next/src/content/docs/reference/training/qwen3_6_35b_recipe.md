---
title: Qwen3_6_35b_Recipe
description: API reference for Qwen3_6_35b_Recipe
---

```python
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b import Qwen3_6_35b_Recipe
```

Qwen3.6-35B-A3B (MoE) on 1×8×H100 with TP2/PP1/CP2/EP8, colocated GRPO.

**Inherits from:** `SlimeRecipe`, `BaseTrainRecipe`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_type` | `str` | `"H100"` |  |
| `colocate` | `bool` | `True` |  |
| `tensor_model_parallel_size` | `int` | `2` |  |
| `sequence_parallel` | `bool` | `True` |  |
| `rollout_num_gpus_per_engine` | `int` | `4` |  |
| `num_rollout` | `int` | `1` |  |
| `rollout_batch_size` | `int` | `8` |  |
| `rollout_max_response_len` | `int` | `4096` |  |
| `rollout_temperature` | `float` | `1.0` |  |
| `save_interval` | `int` | `20` |  |
| `recipe_type` | `RecipeType` | `slime` |  |
| `name` | `str` | `""` |  |
| `app_tags` | `dict` | `{}` |  |
| `environment` | `dict` | `{'PYTHONPATH': '/root/Megatron-LM/', 'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '1'}` |  |
| `async_mode` | `bool` | `False` |  |
| `wandb` | `WandbConfig \| None` | `None` |  |
| `image_overlay` | `collections.abc.Callable[[modal.image.Image], modal.image.Image] \| None` | `None` |  |
| `local_slime` | `str \| None` | `None` |  |
| `actor_num_nodes` | `int` | `1` |  |
| `actor_num_gpus_per_node` | `int` | `8` |  |
| `rollout_num_gpus` | `int \| None` | `None` |  |
| `use_critic` | `bool` | `False` |  |
| `critic_num_nodes` | `int \| None` | `None` |  |
| `critic_num_gpus_per_node` | `int \| None` | `None` |  |
| `advantage_estimator` | `str` | `"grpo"` |  |
| `n_samples_per_prompt` | `int` | `8` |  |
| `eps_clip` | `float` | `0.2` |  |
| `eps_clip_high` | `float` | `0.28` |  |
| `use_kl_loss` | `bool` | `False` |  |
| `kl_loss_type` | `str` | `"low_var_kl"` |  |
| `kl_loss_coef` | `float` | `0.0` |  |
| `entropy_coef` | `float` | `0.0` |  |
| `ref_load` | `str` | `""` |  |
| `rollout_shuffle` | `bool` | `True` |  |
| `sglang_mem_fraction_static` | `float` | `0.75` |  |
| `global_batch_size` | `int` | `32` |  |
| `lr` | `float` | `1e-06` |  |
| `lr_decay_style` | `str` | `"constant"` |  |
| `weight_decay` | `float` | `0.1` |  |
| `adam_beta1` | `float` | `0.9` |  |
| `adam_beta2` | `float` | `0.98` |  |
| `optimizer` | `str` | `"adam"` |  |
| `attention_dropout` | `float` | `0.0` |  |
| `hidden_dropout` | `float` | `0.0` |  |
| `attention_softmax_in_fp32` | `bool` | `True` |  |
| `accumulate_allreduce_grads_in_fp32` | `bool` | `True` |  |
| `recompute_granularity` | `str` | `"full"` |  |
| `recompute_method` | `str` | `"uniform"` |  |
| `recompute_num_layers` | `int` | `1` |  |
| `use_dynamic_batch_size` | `bool` | `True` |  |
| `max_tokens_per_gpu` | `int` | `8192` |  |
| `eval_interval` | `int \| None` | `20` |  |
| `n_samples_per_eval_prompt` | `int` | `4` |  |
| `eval_max_response_len` | `int` | `4096` |  |
| `eval_top_p` | `float` | `1.0` |  |
| `eval_config` | `dict \| None` | `None` |  |
| `save` | `str` | `"/checkpoints"` |  |
| `load` | `str` | `""` |  |
| `megatron_to_hf_mode` | `str` | `"bridge"` |  |
| `use_fault_tolerance` | `bool` | `True` |  |
| `rm_type` | `str \| None` | `None` |  |
| `custom_rm_function` | `collections.abc.Callable \| None` | `None` |  |
| `custom_generate_function` | `collections.abc.Callable \| None` | `None` |  |
| `rollout_function` | `collections.abc.Callable \| str \| None` | `None` |  |
| `custom_megatron_before_train_step_hook` | `collections.abc.Callable \| str \| None` | `None` |  |
| `sglang_enable_dp_attention` | `bool` | `True` |  |
| `sglang_dp_size` | `int \| None` | `4` |  |
| `sglang_ep_size` | `int \| None` | `4` |  |
| `sglang_enable_dp_lm_head` | `bool` | `True` |  |
| `sglang_disable_custom_all_reduce` | `bool` | `False` |  |
| `sglang_cuda_graph_bs` | `list[int] \| None` | `None` |  |
| `sglang_max_running_requests` | `int \| None` | `512` |  |
| `extra_config` | `dict \| None` | `None` |  |
| `sglang_config` | `dict \| None` | `None` |  |
| `apply_chat_template_kwargs` | `str` | `""` |  |
| `pipeline_model_parallel_size` | `int` | `1` |  |
| `context_parallel_size` | `int` | `1` |  |
| `expert_model_parallel_size` | `int` | `8` |  |
| `expert_tensor_parallel_size` | `int` | `1` |  |
| `calculate_per_token_loss` | `bool` | `True` |  |
| `balance_data` | `bool` | `True` |  |
| `optimizer_cpu_offload` | `bool` | `True` |  |
| `overlap_cpu_optimizer_d2h_h2d` | `bool` | `True` |  |
| `use_precision_aware_optimizer` | `bool` | `True` |  |
| `attention_backend` | `str` | `"flash"` |  |

## Methods

### `cli_args(self, dataset: 'DatasetConfig | None' = None, model: 'ModelConfig | None' = None) -> list[str]`

### `get_base_recipe(model_config: modal_training_gym.common.models.base.ModelConfig) -> 'SlimeRecipe | None'`

**Source:** [`modal_training_gym/train_recipes/slime_recipe/qwen3_6_35b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/train_recipes/slime_recipe/qwen3_6_35b.py)
