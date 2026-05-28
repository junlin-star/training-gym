---
title: ToolCall
description: API reference for ToolCall
---

```python
from modal_training_gym.common.models.base import ToolCall
```

A parsed tool invocation from model output.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` |  |  |
| `arguments` | `dict[str, Any]` | `{}` |  |

**Source:** [`modal_training_gym/common/models/base.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/base.py)
