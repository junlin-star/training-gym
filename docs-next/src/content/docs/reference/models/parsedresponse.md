---
title: ParsedResponse
description: API reference for ParsedResponse
---

```python
from modal_training_gym.common.models.base import ParsedResponse
```

Structured result of parsing raw model output.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | `str` | `""` |  |
| `tool_calls` | `list[ToolCall]` | `[]` |  |
| `thinking` | `str \| None` | `None` |  |

**Source:** [`modal_training_gym/common/models/base.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/base.py)
