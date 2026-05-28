---
title: Qwen3-4B
description: API reference for Qwen3_4B
---

```python
from modal_training_gym.common.models.qwen3_4b import Qwen3_4B
```

Qwen3-4B (4 billion parameters) from Alibaba.

**Inherits from:** `HFModelConfiguration`, `ModelConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen3-4B"` |  |
| `model_path` | `str \| None` | `None` |  |
| `architecture` | `ModelArchitecture \| None` | `ModelArchitecture(num_layers=36, hidden_size=2560, ffn_hidden_size=9728, num_attention_heads=32, group_query_attention=True, num_query_groups=8, kv_channels=128, vocab_size=151936, normalization='RMSNorm', norm_epsilon=1e-06, swiglu=True, disable_bias_linear=True, qk_layernorm=True, untie_embeddings_and_output_weights=False, num_experts=0, moe_ffn_hidden_size=0, moe_shared_expert_intermediate_size=0, moe_grouped_gemm=False, moe_shared_expert_gate=False, moe_router_topk=0, moe_router_score_function='', moe_token_drop_policy='', moe_router_dtype='', moe_permute_fusion=False, moe_aux_loss_coeff=None, megatron_spec=None, megatron_model_type='', apply_layernorm_1p=False, use_gated_attention=False, attention_output_gate=False, use_rotary_position_embeddings=True, rotary_base=1000000, rotary_percent=1.0)` |  |
| `response_parser` | `Optional[Callable[[str], ParsedResponse]]` | `<function parse_qwen3_response at 0x7f5a198f8ea0>` |  |

## Methods

### `download(self) -> 'None'`

Download or materialize weights into the model volume.

### `parse_response(self, text: 'str') -> 'ParsedResponse'`

Parse raw model output into structured content.

### `response_parser(text: 'str') -> 'ParsedResponse'`

Parse Qwen3-family model output into structured content.

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Code RL with Harbor hello-world and sandboxed verification](/tutorials/rl/001_sandboxes/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn/)
- [On-policy distillation on math — Qwen3-8B teacher, Qwen3-4B student](/tutorials/rl/003_on_policy_distillation/)

**Source:** [`modal_training_gym/common/models/qwen3_4b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/qwen3_4b.py)
