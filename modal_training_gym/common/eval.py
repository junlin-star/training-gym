from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import datetime
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from modal_training_gym.common.dataset import DatasetRow
from modal_training_gym.common.ids import create_hash
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_list, vol_put

from modal_training_gym.common.sample import Sample

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.deployment import ModelDeployment
    from modal_training_gym.common.models.base import ModelConfig

EVAL_SUMMARY_STORE = MetadataStore.EVALS
EVAL_SUMMARY_KEY = "summary"
EVAL_SUMMARY_PAYLOAD_KEY = "summaries"

#: How often (in completed rows) a running eval flushes partial results to the
#: metadata volume. Smaller = fresher dashboard, more volume writes.
_INTERMEDIATE_SAVE_EVERY = 5


def _callable_name(fn: Callable[..., Any]) -> str:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if name:
        return name
    return type(fn).__name__


class EvalConfigDurable(BaseModel):
    """JSON-serializable audit record for an :class:`EvalConfig`."""

    eval_config_id: str
    dataset_name: str
    eval_fn_name: str
    prompt_column: str | None = None
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)

    def save(self) -> None:
        vol_put(
            MetadataStore.EVAL_CONFIGS,
            self.eval_config_id,
            self.model_dump(mode="json"),
        )

    @classmethod
    def from_id(cls, eval_config_id: str) -> "EvalConfigDurable":
        return cls.model_validate(vol_get(MetadataStore.EVAL_CONFIGS, eval_config_id))

    @classmethod
    def list_configs(cls) -> list["EvalConfigDurable"]:
        return [cls.model_validate(v) for v in vol_list(MetadataStore.EVAL_CONFIGS)]


# An eval row is just a Sample. Kept as an alias for the public API / existing
# imports; new code should use Sample directly.
EvalRowResult = Sample


class AudioEvalRowResult(Sample):
    """``Sample`` for an audio eval, with the audio fields lifted to
    constructor arguments.

    ``audio`` (a browser-playable data-URI), ``reference`` (the ground truth), and
    ``metrics`` (a ``{name: value}`` dict — the eval picks its own metrics, e.g.
    ``{"wer": 0.1}`` or ``{"mos": 4.2}``) are folded into ``metadata`` under
    ``_metadata_type="audio"`` so the evals dashboard auto-detects and renders an
    audio cell. The model output stays on ``response`` (there is no separate
    ``hypothesis``); ``score`` remains the canonical headline number. Extra
    ``metadata`` is kept.

    Usage:
        AudioEvalRowResult(
            score=1.0 - wer, response=hypothesis, prompt=prompt,
            audio=audio_uri, reference=reference, metrics={"wer": wer},
        )
    """

    @model_validator(mode="before")
    @classmethod
    def _fold_audio_into_metadata(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        metadata = dict(data.pop("metadata", None) or {})
        metadata["_metadata_type"] = "audio"
        for key in ("audio", "reference", "metrics"):
            value = data.pop(key, None)
            if value is not None:
                metadata[key] = value
        data["metadata"] = metadata
        return data


#: Lifecycle of an eval run. ``running`` rows are streamed in as examples
#: complete so the dashboard shows intermediate output; ``completed`` is the
#: terminal success state and ``failed`` marks a run that raised partway
#: through. The default is ``completed`` so records written before this field
#: existed validate as finished runs.
EvalStatus = Literal["running", "completed", "failed"]


class EvalSummary(BaseModel):
    eval_id: str
    eval_config_id: str
    created_at: datetime.datetime
    total: int
    mean: float
    status: EvalStatus = "completed"

    @classmethod
    def list_summaries(cls) -> list["EvalSummary"]:
        try:
            payload = vol_get(EVAL_SUMMARY_STORE, EVAL_SUMMARY_KEY)
        except KeyError:
            return []
        summaries = (
            payload.get(EVAL_SUMMARY_PAYLOAD_KEY, [])
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(summaries, list):
            return []
        return [cls.model_validate(summary) for summary in summaries]

    @classmethod
    def save_summaries(cls, summaries: list["EvalSummary"]) -> None:
        vol_put(
            EVAL_SUMMARY_STORE,
            EVAL_SUMMARY_KEY,
            {
                EVAL_SUMMARY_PAYLOAD_KEY: [
                    summary.model_dump(mode="json") for summary in summaries
                ]
            },
        )


class EvalResult(BaseModel):
    """Saved results for one evaluation run across a dataset."""

    eval_id: str
    eval_config_id: str
    deployment_id: str
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    status: EvalStatus = "completed"
    rows: list[EvalRowResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def mean(self) -> float:
        return sum(r.score for r in self.rows) / self.total if self.total else 0.0

    def to_summary(self) -> EvalSummary:
        return EvalSummary(
            eval_id=self.eval_id,
            eval_config_id=self.eval_config_id,
            created_at=self.created_at,
            total=self.total,
            mean=self.mean,
            status=self.status,
        )

    def save(self) -> None:
        vol_put(MetadataStore.EVAL_RESULTS, self.eval_id, self.model_dump(mode="json"))
        summaries = EvalSummary.list_summaries()
        summaries_by_id = {summary.eval_id: summary for summary in summaries}
        summaries_by_id[self.eval_id] = self.to_summary()
        EvalSummary.save_summaries(
            sorted(
                summaries_by_id.values(),
                key=lambda summary: summary.created_at,
                reverse=True,
            )
        )

    @classmethod
    def from_id(cls, eval_id: str) -> "EvalResult":
        return cls.model_validate(vol_get(MetadataStore.EVAL_RESULTS, eval_id))

    @classmethod
    def list_results(cls) -> list["EvalResult"]:
        return [cls.model_validate(v) for v in vol_list(MetadataStore.EVAL_RESULTS)]


Response = str
EvalResponseFn = Callable[[DatasetRow, Response], EvalRowResult]  # TOOD: bad name
EvalFn = Callable[["ModelDeployment", DatasetRow], EvalRowResult]


@dataclass
class EvalConfig:
    """Evaluate a deployed model on a dataset config.

    The dataset must expose ``load()`` and return iterable dict examples.
    """

    dataset: "DatasetConfig"
    eval_fn: EvalFn | None = None
    eval_response_fn: EvalResponseFn | None = None
    prompt_column: str | None = None
    eval_config_id: str | None = None
    generate_kwargs: dict[str, Any] = field(default_factory=dict)

    def _build_eval_fn(self, eval_response_fn: EvalResponseFn) -> EvalFn:
        def eval_fn(
            deployment: ModelDeployment,
            example: DatasetRow,
        ) -> EvalRowResult:
            prompt = self.build_prompt(example)
            text = deployment.generate(
                prompt,
                ensure_ready=False,
                **self.generate_kwargs,
            )
            result = eval_response_fn(example, text)
            return EvalRowResult(
                score=result.score,
                response=text,
                prompt=prompt,
                metadata=result.metadata,
            )

        return eval_fn

    def __post_init__(self):
        if self.eval_config_id is None:
            class_name = type(self).__name__
            dataset_name = type(self.dataset).__name__
            eval_fn_name = _callable_name(self.eval_fn or self.eval_response_fn)
            self.eval_config_id = create_hash(
                "eval-config",
                class_name,
                dataset_name,
                eval_fn_name,
                self.prompt_column or "",
            )
        if self.eval_fn is None:
            assert self.eval_response_fn is not None, (
                "eval_fn or eval_response_fn must be set"
            )
            self.eval_fn = self._build_eval_fn(self.eval_response_fn)

    def to_durable(self) -> EvalConfigDurable:
        eval_callable = (
            self.eval_response_fn if self.eval_response_fn is not None else self.eval_fn
        )
        return EvalConfigDurable(
            eval_config_id=self.eval_config_id,
            dataset_name=type(self.dataset).__name__,
            eval_fn_name=_callable_name(eval_callable),
            prompt_column=self.prompt_column,
            generate_kwargs=self.generate_kwargs,
        )

    def save(self) -> EvalConfigDurable:
        durable = self.to_durable()
        durable.save()
        return durable

    def build_prompt(self, row: DatasetRow) -> str:
        prompt_column = (self.prompt_column or "").strip()
        input_column = getattr(self.dataset, "input_column", "")
        dataset_column = input_column if isinstance(input_column, str) else ""

        preferred_columns: list[str] = []
        if prompt_column:
            preferred_columns.append(prompt_column)
        if dataset_column and dataset_column not in preferred_columns:
            preferred_columns.append(dataset_column)
        for fallback in ("prompt", "input", "instruction", "question"):
            if fallback not in preferred_columns:
                preferred_columns.append(fallback)

        for column in preferred_columns:
            if column not in row:
                continue
            raw = str(row[column])
            if column in {prompt_column, dataset_column}:
                template = (
                    getattr(self.dataset, "prompt_template", "{input}") or "{input}"
                )
                row_context = {
                    key: str(value)
                    for key, value in row.items()
                    if isinstance(key, str)
                }
                try:
                    return template.format(input=raw, **row_context)
                except (KeyError, ValueError):
                    return raw
            return raw

        raise ValueError(
            "EvalConfig.build_prompt() could not resolve a prompt column. "
            "Set EvalConfig.prompt_column or dataset.input_column, or include one of "
            "['prompt', 'input', 'instruction', 'question'] in dataset rows."
        )

    def evaluate(
        self,
        deployment: "ModelDeployment",
        debug: bool = False,
        max_concurrency: int = 1,
    ) -> EvalResult:
        from modal_training_gym.setup import ensure_dashboard_deployed

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        ensure_dashboard_deployed()

        self.save()
        deployment.wait_until_ready()

        def _evaluate_indexed(
            item: tuple[int, DatasetRow],
        ) -> tuple[int, EvalRowResult]:
            idx, example = item
            return idx, self.eval_fn(deployment, example)

        # Persist a ``running`` record up front (and stream rows into it as
        # they complete) so the dashboard surfaces an in-progress eval with
        # intermediate output instead of nothing until the whole run finishes.
        eval_id = create_hash(
            "eval",
            self.eval_config_id,
            deployment.deployment_id,
            type(self.dataset).__name__,
            _callable_name(self.eval_fn or self.eval_response_fn),
        )
        result = EvalResult(
            eval_id=eval_id,
            deployment_id=deployment.deployment_id,
            eval_config_id=self.eval_config_id,
            created_at=datetime.datetime.now(datetime.UTC),
            status="running",
            rows=[],
        )
        result.save()

        try:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                indexed_results = executor.map(
                    _evaluate_indexed,
                    enumerate(self.dataset.load(split="eval"), start=1),
                )
                for idx, row_result in indexed_results:
                    if debug:
                        print(
                            f"Finished example {idx}: "
                            f"response={row_result.response!r} "
                            f"score={row_result.score}",
                            flush=True,
                        )
                    result.rows.append(row_result)
                    # Flush partial progress periodically rather than per row:
                    # each save rewrites the shared summary list, so throttle to
                    # keep volume writes bounded on large datasets.
                    if result.total % _INTERMEDIATE_SAVE_EVERY == 0:
                        result.save()
        except Exception:
            result.status = "failed"
            result.save()
            raise

        result.status = "completed"
        result.save()
        return result