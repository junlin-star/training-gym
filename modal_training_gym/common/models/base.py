from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ModelArchitecture:
    """Transformer architecture parameters for a specific model.

    These fields map directly to Megatron-LM model-parallel configuration
    flags. Framework launchers read them to generate the correct CLI
    arguments for distributed training.

    ## Model Dimensions

    num_layers : int
        Number of transformer layers. Default ``0``.
    hidden_size : int
        Hidden dimension size. Default ``0``.
    ffn_hidden_size : int
        Feed-forward network intermediate size. Default ``0``.
    vocab_size : int
        Vocabulary size. Default ``0``.

    ## Attention

    num_attention_heads : int
        Number of attention heads. Default ``0``.
    group_query_attention : bool
        Enable grouped-query attention (GQA). Default ``True``.
    num_query_groups : int
        Number of KV head groups for GQA. Default ``0``.
    kv_channels : int
        Per-head key/value channel dimension. Default ``0``.

    ## Normalization and Activation

    normalization : str
        Layer normalization type. Default ``"RMSNorm"``.
    norm_epsilon : float
        Normalization epsilon. Default ``1e-6``.
    swiglu : bool
        Use SwiGLU activation in FFN. Default ``True``.
    disable_bias_linear : bool
        Disable bias in linear layers. Default ``True``.
    qk_layernorm : bool
        Apply layer norm to query and key projections. Default ``True``.
    untie_embeddings_and_output_weights : bool
        Use separate output projection weights instead of tying to token
        embeddings. Default ``False``.

    ## Mixture of Experts

    num_experts : int
        Total number of MoE experts. Default ``0`` (dense model).
    moe_ffn_hidden_size : int
        Per-expert FFN intermediate size. Default ``0``.
    moe_shared_expert_intermediate_size : int
        Shared expert FFN intermediate size. Default ``0``.

    ## Checkpoint Conversion

    megatron_model_type : str
        Slime/Megatron model type string for pre-conversion (e.g.
        ``"qwen3.5-35B-A3B"``). When set, the launcher pre-converts
        the HF checkpoint to torch_dist format before training instead
        of relying on bridge-mode auto-detection. Default ``""``.

    ## Position Encoding

    use_rotary_position_embeddings : bool
        Use RoPE positional encoding. Default ``True``.
    rotary_base : int
        Base frequency for RoPE. Default ``10000``.
    """

    num_layers: int = 0
    hidden_size: int = 0
    ffn_hidden_size: int = 0
    num_attention_heads: int = 0
    group_query_attention: bool = True
    num_query_groups: int = 0
    kv_channels: int = 0
    vocab_size: int = 0
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-6
    swiglu: bool = True
    disable_bias_linear: bool = True
    qk_layernorm: bool = True
    untie_embeddings_and_output_weights: bool = False
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    megatron_model_type: str = ""
    use_rotary_position_embeddings: bool = True
    rotary_base: int = 10000

    @property
    def needs_pre_conversion(self) -> bool:
        return bool(self.megatron_model_type)

    def to_megatron_args(self) -> list[str]:
        """Generate Megatron-LM CLI flags from this architecture spec."""
        args: list[str] = []
        if self.num_layers:
            args += ["--num-layers", str(self.num_layers)]
        if self.hidden_size:
            args += ["--hidden-size", str(self.hidden_size)]
        if self.ffn_hidden_size:
            args += ["--ffn-hidden-size", str(self.ffn_hidden_size)]
        if self.num_attention_heads:
            args += ["--num-attention-heads", str(self.num_attention_heads)]
        if self.group_query_attention:
            args.append("--group-query-attention")
        if self.num_query_groups:
            args += ["--num-query-groups", str(self.num_query_groups)]
        if self.kv_channels:
            args += ["--kv-channels", str(self.kv_channels)]
        if self.vocab_size:
            args += ["--vocab-size", str(self.vocab_size)]
            args += ["--make-vocab-size-divisible-by", "1"]
        if self.normalization:
            args += ["--normalization", self.normalization]
        if self.norm_epsilon:
            args += ["--norm-epsilon", str(self.norm_epsilon)]
        if self.swiglu:
            args.append("--swiglu")
        if self.disable_bias_linear:
            args.append("--disable-bias-linear")
        if self.qk_layernorm:
            args.append("--qk-layernorm")
        if self.untie_embeddings_and_output_weights:
            args.append("--untie-embeddings-and-output-weights")
        if self.num_experts:
            args += ["--num-experts", str(self.num_experts)]
        if self.moe_ffn_hidden_size:
            args += ["--moe-ffn-hidden-size", str(self.moe_ffn_hidden_size)]
        if self.moe_shared_expert_intermediate_size:
            args += ["--moe-shared-expert-intermediate-size", str(self.moe_shared_expert_intermediate_size)]
        if self.use_rotary_position_embeddings:
            args += ["--position-embedding-type", "rope"]
            if self.rotary_base != 10000:
                args += ["--rotary-base", str(self.rotary_base)]
        return args


@dataclass
class ToolCall:
    """A parsed tool invocation from model output."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedResponse:
    """Structured result of parsing raw model output."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None


ResponseParser = Callable[[str], ParsedResponse]


class ModelConfig:
    """Base class for model identity and weight-download logic.

    Subclass and set ``model_name`` (and optionally ``model_path`` and
    ``architecture``) as class attributes, then override ``download()``
    to materialize weights into the shared model volume.

    Set ``response_parser`` to a function that converts raw model output
    into a :class:`ParsedResponse`.  For example, Qwen3 models set
    ``response_parser = parse_qwen3_response``.
    """

    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None
    response_parser: ResponseParser | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def download(self) -> None:
        """Download or materialize weights into the model volume."""
        raise NotImplementedError(f"{type(self).__name__} has no download()")

    def parse_response(self, text: str) -> ParsedResponse:
        """Parse raw model output into structured content.

        Delegates to ``self.response_parser`` when set, otherwise
        returns the text as-is.
        """
        if self.response_parser is not None:
            return self.response_parser(text)
        return ParsedResponse(content=text)


class HFModelConfiguration(ModelConfig):
    """ModelConfig for models hosted on HuggingFace.

    Implements ``download()`` via ``huggingface_hub.snapshot_download``
    using ``self.model_name`` as the repo ID.
    """

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        kwargs: dict = {"repo_id": self.model_name}
        if self.model_path:
            kwargs["local_dir"] = str(self.model_path)
        snapshot_download(**kwargs)


# ── Qwen3 family ───────────────────────────────────────────────────────

_QWEN3_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_qwen3_response(text: str) -> ParsedResponse:
    """Parse Qwen3-family model output into structured content.

    Handles ``<think>``/``</think>`` reasoning blocks,
    ``<|im_start|>``/``<|im_end|>`` chat-template delimiters,
    and ``<tool_call>``/``</tool_call>`` tool invocations.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "<|im_start|>assistant" in text:
        text = text.rsplit("<|im_start|>assistant", 1)[-1]
    text = text.replace("<|im_end|>", "")

    thinking: str | None = None
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip() or None
        text = parts[1]
    text = text.replace("<think>", "")

    tool_calls: list[ToolCall] = []
    for match in _QWEN3_TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            tool_calls.append(
                ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    content = _QWEN3_TOOL_CALL_RE.sub("", text).strip()

    return ParsedResponse(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
    )
