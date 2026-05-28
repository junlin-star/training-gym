---
title: HarborEval
description: API reference for HarborEval
---

```python
from modal_training_gym.common.eval import HarborEval
```

Evaluate a deployed model on a Harbor dataset using sandbox execution.

**Inherits from:** `EvalConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `'DatasetConfig'` |  |  |
| `eval_fn` | `EvalFn \| None` | `None` |  |
| `eval_response_fn` | `EvalResponseFn \| None` | `None` |  |
| `prompt_column` | `str \| None` | `None` |  |
| `eval_config_id` | `str \| None` | `None` |  |
| `generate_kwargs` | `dict[str, Any]` | `{}` |  |
| `model` | `'ModelConfig \| None'` | `None` |  |
| `test_cases` | `list[dict[str, str]] \| None` | `None` |  |
| `sandbox_timeout` | `int` | `60` |  |
| `sandbox_cpu` | `float` | `1.0` |  |
| `sandbox_memory` | `int` | `1024` |  |
| `sandbox_python_version` | `str` | `"3.11"` |  |
| `extract_code_fn` | `Callable[[str], str] \| None` | `None` |  |

## Methods

### `build_prompt(self, row: 'DatasetRow') -> 'str'`

### `evaluate(self, deployment: "'ModelDeployment'", debug: 'bool' = False, max_concurrency: 'int' = 1) -> 'EvalResult'`

### `save(self) -> 'EvalConfigDurable'`

### `to_durable(self) -> 'EvalConfigDurable'`

## Related Tutorials

- [Code RL with Harbor hello-world and sandboxed verification](/tutorials/rl/001_sandboxes/)

**Source:** [`modal_training_gym/common/eval.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/eval.py)
