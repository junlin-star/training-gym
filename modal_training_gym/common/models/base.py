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

    ## MoE Routing

    moe_router_score_function : str
        Router scoring function (e.g. ``"softmax"``). Default ``""``.
    moe_token_drop_policy : str
        Token drop policy for MoE routing. Default ``""``.
    moe_router_dtype : str
        Data type for router computation (e.g. ``"fp32"``). Default ``""``.
    moe_permute_fusion : bool
        Enable permute fusion optimization for MoE. Default ``False``.
    moe_aux_loss_coeff : float | None
        Auxiliary load-balancing loss coefficient. Default ``None``.

    ## Checkpoint Conversion

    megatron_model_type : str
        Slime/Megatron model type string for checkpoint conversion (e.g.
        ``"qwen3.5-35B-A3B"``). Used when the training recipe selects
        a non-bridge conversion mode. Default ``""``.

    ## Normalization Extras

    apply_layernorm_1p : bool
        Use zero-centered LayerNorm (add 1 to gamma). Default ``False``.

    ## Attention Extras

    use_gated_attention : bool
        Enable gated attention mechanism. Default ``False``.
    attention_output_gate : bool
        Enable output gating on attention layers (required by some
        hybrid architectures such as Qwen 3.6). Default ``False``.

    ## Position Encoding

    use_rotary_position_embeddings : bool
        Use RoPE positional encoding. Default ``True``.
    rotary_base : int
        Base frequency for RoPE. Default ``10000``.
    rotary_percent : float
        Fraction of hidden dims to apply RoPE to. Default ``1.0``.
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
    moe_grouped_gemm: bool = False
    moe_shared_expert_gate: bool = False
    moe_router_topk: int = 0
    moe_router_score_function: str = ""
    moe_token_drop_policy: str = ""
    moe_router_dtype: str = ""
    moe_permute_fusion: bool = False
    moe_aux_loss_coeff: float | None = None
    megatron_spec: list[str] | None = None
    megatron_model_type: str = ""
    # Names of compat patch files (in frameworks/slime/modal_helpers/patches/) to
    # apply at image build for this model — for upstream gaps a model needs until
    # they're fixed upstream (e.g. Qwen3-ASR's bridge/processor/export shims). The
    # launcher applies these only when this model is used.
    compat_patches: list[str] | None = None
    apply_layernorm_1p: bool = False
    use_gated_attention: bool = False
    attention_output_gate: bool = False
    use_rotary_position_embeddings: bool = True
    rotary_base: int = 10000
    rotary_percent: float = 1.0

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
            args += [
                "--moe-shared-expert-intermediate-size",
                str(self.moe_shared_expert_intermediate_size),
            ]
        if self.moe_grouped_gemm:
            args.append("--moe-grouped-gemm")
        if self.moe_shared_expert_gate:
            args.append("--moe-shared-expert-gate")
        if self.moe_router_topk:
            args += ["--moe-router-topk", str(self.moe_router_topk)]
        if self.moe_router_score_function:
            args += ["--moe-router-score-function", self.moe_router_score_function]
        if self.moe_token_drop_policy:
            args += ["--moe-token-drop-policy", self.moe_token_drop_policy]
        if self.moe_router_dtype:
            args += ["--moe-router-dtype", self.moe_router_dtype]
        if self.moe_permute_fusion:
            args.append("--moe-permute-fusion")
        if self.moe_aux_loss_coeff is not None:
            args += ["--moe-aux-loss-coeff", str(self.moe_aux_loss_coeff)]
        if self.megatron_spec:
            args += ["--spec"] + list(self.megatron_spec)
        if self.apply_layernorm_1p:
            args.append("--apply-layernorm-1p")
        if self.use_gated_attention:
            args.append("--use-gated-attention")
        if self.attention_output_gate:
            args.append("--attention-output-gate")
        if self.use_rotary_position_embeddings:
            args += ["--position-embedding-type", "rope"]
            if self.rotary_base != 10000:
                args += ["--rotary-base", str(self.rotary_base)]
            if self.rotary_percent != 1.0:
                args += ["--rotary-percent", str(self.rotary_percent)]
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
