from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from modal_training_gym.common.models.base import ParsedResponse


class TraceSpan(BaseModel):
    """One span or instant event in a sample's execution trace.

    A duration span has both ``start`` and ``end`` (seconds, rebased so the
    sample's first span starts at 0); an instant event has ``end is None``.
    ``attributes`` carries timings/counts only — never response or tool
    payloads, which already live on the Sample — so traces stay small (see the
    recorder's normalizer). ``parent`` is the enclosing span's name, if any.
    """

    name: str = ""
    start: float = 0.0
    end: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    parent: str | None = None


class Sample(BaseModel):
    """One model interaction: the prompt, the raw response, its parsed
    structure (thinking / answer / tool calls), a score, and free-form
    metadata.

    Shared by custom post-train check rows and training rollout samples
    (``TrainingRolloutResult.samples``).
    """

    score: float = 0.0
    prompt: str = ""
    response: str = ""
    parsed_response: ParsedResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # captured only when trace recording is enabled and only for a sampled
    # subset of each rollout's samples. ``None`` for untraced samples.
    trace: list[TraceSpan] | None = None
