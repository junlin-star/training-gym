---
title: "parse_qwen3_response"
---

# `parse_qwen3_response`

`parse_qwen3_response(text: 'str') -> 'ParsedResponse'`

Parse Qwen3-family model output into structured content.

Handles ``<think>``/``</think>`` reasoning blocks,
``<|im_start|>``/``<|im_end|>`` chat-template delimiters,
and ``<tool_call>``/``</tool_call>`` tool invocations.

