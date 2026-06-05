"""Dummy reward for the multimodal RL smoke demos.

slime cloudpickles this and calls it per rollout sample with a ``Sample``
exposing ``.response`` (decoded text) and ``.label`` (the dataset's
``label_key`` value). This placeholder scores token overlap against the label
so the no-GPU smoke has something to exercise across modalities.

Replace per task in a real run, e.g. audio ASR -> ``-jiwer.wer(label, response)``.
"""

from __future__ import annotations

from typing import Any


def _score(response: str, reference: str) -> float:
    """Token-overlap F1 in [0, 1]. Placeholder for a real task metric."""
    hyp, ref = response.lower().split(), reference.lower().split()
    if not hyp or not ref:
        return 0.0
    overlap = sum(min(hyp.count(w), ref.count(w)) for w in set(ref))
    p, r = overlap / len(hyp), overlap / len(ref)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _get(sample: Any, attr: str) -> Any:
    """slime passes a Sample object; tests pass a dict — support both."""
    return sample.get(attr, "") if isinstance(sample, dict) else getattr(sample, attr, "")


async def dummy_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    return float(_score(_get(sample, "response") or "", _get(sample, "label") or ""))


async def wer_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    """ASR reward: negative word error rate of the transcript vs. the reference."""
    import jiwer

    response = (_get(sample, "response") or "").lower().strip()
    reference = (_get(sample, "label") or "").lower().strip()
    if not reference:
        return 0.0
    return -float(jiwer.wer(reference, response))
