---
title: ModelArchitecture
description: API reference for ModelArchitecture
---

```python
from modal_training_gym.common.models.base import ModelArchitecture
```

Transformer architecture parameters for a specific model.

## Model Dimensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_layers` | `int` | `0` | Number of transformer layers. Default `0`. |
| `hidden_size` | `int` | `0` | Hidden dimension size. Default `0`. |
| `ffn_hidden_size` | `int` | `0` | Feed-forward network intermediate size. Default `0`. |
| `vocab_size` | `int` | `0` | Vocabulary size. Default `0`. |

## Attention

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_attention_heads` | `int` | `0` | Number of attention heads. Default `0`. |
| `group_query_attention` | `bool` | `True` | Enable grouped-query attention (GQA). Default `True`. |
| `num_query_groups` | `int` | `0` | Number of KV head groups for GQA. Default `0`. |
| `kv_channels` | `int` | `0` | Per-head key/value channel dimension. Default `0`. |

## Normalization and Activation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `normalization` | `str` | `"RMSNorm"` | Layer normalization type. Default `"RMSNorm"`. |
| `norm_epsilon` | `float` | `1e-06` | Normalization epsilon. Default `1e-6`. |
| `swiglu` | `bool` | `True` | Use SwiGLU activation in FFN. Default `True`. |
| `disable_bias_linear` | `bool` | `True` | Disable bias in linear layers. Default `True`. |
| `qk_layernorm` | `bool` | `True` | Apply layer norm to query and key projections. Default `True`. |
| `untie_embeddings_and_output_weights` | `bool` | `False` | Use separate output projection weights instead of tying to token embeddings. Default `False`. |

## Mixture of Experts

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_experts` | `int` | `0` | Total number of MoE experts. Default `0` (dense model). |
| `moe_ffn_hidden_size` | `int` | `0` | Per-expert FFN intermediate size. Default `0`. |
| `moe_shared_expert_intermediate_size` | `int` | `0` | Shared expert FFN intermediate size. Default `0`. |

## MoE Routing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `moe_router_score_function` | `str` | `""` | Router scoring function (e.g. `"softmax"`). Default `""`. |
| `moe_token_drop_policy` | `str` | `""` | Token drop policy for MoE routing. Default `""`. |
| `moe_router_dtype` | `str` | `""` | Data type for router computation (e.g. `"fp32"`). Default `""`. |
| `moe_permute_fusion` | `bool` | `False` | Enable permute fusion optimization for MoE. Default `False`. |
| `moe_aux_loss_coeff` | `float \| None` | `None` | Auxiliary load-balancing loss coefficient. Default `None`. |

## Checkpoint Conversion

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `megatron_model_type` | `str` | `""` | Slime/Megatron model type string for pre-conversion (e.g. `"qwen3.5-35B-A3B"`). When set, the launcher pre-converts the HF checkpoint to torch_dist format before training instead of relying on bridge-mode auto-detection. Default `""`. |

## Normalization Extras

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `apply_layernorm_1p` | `bool` | `False` | Use zero-centered LayerNorm (add 1 to gamma). Default `False`. |

## Attention Extras

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_gated_attention` | `bool` | `False` | Enable gated attention mechanism. Default `False`. |
| `attention_output_gate` | `bool` | `False` | Enable output gating on attention layers (required by some hybrid architectures such as Qwen 3.6). Default `False`. |

## Position Encoding

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_rotary_position_embeddings` | `bool` | `True` | Use RoPE positional encoding. Default `True`. |
| `rotary_base` | `int` | `10000` | Base frequency for RoPE. Default `10000`. |
| `rotary_percent` | `float` | `1.0` | Fraction of hidden dims to apply RoPE to. Default `1.0`. |

## Other Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `moe_grouped_gemm` | `bool` | `False` |  |
| `moe_shared_expert_gate` | `bool` | `False` |  |
| `moe_router_topk` | `int` | `0` |  |
| `megatron_spec` | `list[str] \| None` | `None` |  |

## Methods

### `to_megatron_args(self) -> 'list[str]'`

Generate Megatron-LM CLI flags from this architecture spec.

**Source:** [`modal_training_gym/common/models/base.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/base.py)
