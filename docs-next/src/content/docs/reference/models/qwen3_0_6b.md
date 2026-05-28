---
title: Qwen3-0.6B
description: API reference for Qwen3_0_6B
---

```python
from modal_training_gym.common.models.qwen3_0_6b import Qwen3_0_6B
```

Qwen3-0.6B (0.6 billion parameters) from Alibaba.

**Inherits from:** `HFModelConfiguration`, `ModelConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen3-0.6B"` |  |
| `model_path` | `str \| None` | `None` |  |
| `architecture` | `ModelArchitecture \| None` | `ModelArchitecture(num_layers=28, hidden_size=1024, ffn_hidden_size=3072, num_attention_heads=16, group_query_attention=True, num_query_groups=8, kv_channels=128, vocab_size=151936, normalization='RMSNorm', norm_epsilon=1e-06, swiglu=True, disable_bias_linear=True, qk_layernorm=True, untie_embeddings_and_output_weights=False, num_experts=0, moe_ffn_hidden_size=0, moe_shared_expert_intermediate_size=0, moe_grouped_gemm=False, moe_shared_expert_gate=False, moe_router_topk=0, moe_router_score_function='', moe_token_drop_policy='', moe_router_dtype='', moe_permute_fusion=False, moe_aux_loss_coeff=None, megatron_spec=None, megatron_model_type='', apply_layernorm_1p=False, use_gated_attention=False, attention_output_gate=False, use_rotary_position_embeddings=True, rotary_base=1000000, rotary_percent=1.0)` |  |
| `response_parser` | `Optional[Callable[[str], ParsedResponse]]` | `<function parse_qwen3_response at 0x7f8490d50360>` |  |

## Methods

### `download(self) -> 'None'`

Download or materialize weights into the model volume.

### `parse_response(self, text: 'str') -> 'ParsedResponse'`

Parse raw model output into structured content.

### `response_parser(text: 'str') -> 'ParsedResponse'`

Parse Qwen3-family model output into structured content.

**Source:** [`modal_training_gym/common/models/qwen3_0_6b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/qwen3_0_6b.py)
