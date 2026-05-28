---
title: EvalRowResult
description: API reference for EvalRowResult
---

```python
from modal_training_gym.common.eval import EvalRowResult
```

One evaluated row: score, response text, and optional metadata.

## Constructor

```python
EvalRowResult(**data)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `score` | `float` |  |  |
| `response` | `str` |  |  |
| `prompt` | `str` |  |  |
| `parsed_response` | `ParsedResponse \| None` |  |  |
| `metadata` | `dict[str, Any]` |  |  |

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn/)
- [On-policy distillation on math — Qwen3-8B teacher, Qwen3-4B student](/tutorials/rl/003_on_policy_distillation/)
- [Hill-climb Qwen3.6-35B-A3B on GSM8K math with GRPO](/tutorials/rl/004_qwen35b/)

**Source:** [`modal_training_gym/common/eval.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/eval.py)
