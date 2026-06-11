from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from modal_training_gym.common.models.base import ParsedResponse


class Sample(BaseModel):
    """One model interaction: the prompt, the raw response, its parsed
    structure (thinking / answer / tool calls), a score, and free-form
    metadata.

    Shared by eval rows (``EvalResult.rows``) and training rollout samples
    (``TrainingRolloutResult.samples``) — they were the same shape, so this is
    the single canonical type for both.
    """

    score: float = 0.0
    prompt: str = ""
    response: str = ""
    parsed_response: ParsedResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
