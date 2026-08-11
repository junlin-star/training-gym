from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from datasets import load_dataset

from modal_training_gym.common.dataset import (
    DatasetConfig,
    HuggingFaceDataset,
    MultimodalDataset,
)
from modal_training_gym.common.launcher_utils import serialize_recipe_params
from modal_training_gym.common.modal_lifecycle import stop_app
from modal_training_gym.common.models.qwen3_asr_1_7b import Qwen3_ASR_1_7B
from modal_training_gym.common.models.validation import VALIDATABLE_MODELS
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.model import ModelConfig
from modal_training_gym.train import TrainConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

VALIDATION_EPHEMERAL_DISK_MIB = 2_097_152


@dataclass
class TutorialResult:
    base_model_name: str
    step_count: int
    training_run_id: str
    training_run_status: TrainingRunStatus
    total_duration_s: float
    step_times: dict[str, dict[str, int | None]] | None = None
    substep_times: dict[str, dict[str, dict[str, float | None]]] | None = None
    modal_app_id: str = ""
    modal_app_url: str = ""

    @property
    def succeeded(self) -> bool:
        return self.training_run_status == TrainingRunStatus.COMPLETED

    def print_summary(self) -> None:
        print(f"Training run result for {self.training_run_id}")
        print("Parameters:")
        print(f"Base model name: {self.base_model_name}")
        print(f"Step count: {self.step_count}")
        print("Result:")
        print(f"Training run status: {self.training_run_status}")
        print(f"Total step time (s): {total_step_time_s(self)}")
        print(f"Total duration (s): {self.total_duration_s}")

        keys = _step_keys(self)
        if not keys:
            return

        print("Timings:")
        for key in keys:
            step = (self.step_times or {}).get(key, {})
            duration = step.get("duration_s")
            print(f"Step {key} ({fmt_secs(duration)})")

            for name, entry in _ordered_substeps(
                (self.substep_times or {}).get(key, {})
            ):
                print(
                    f"    {phase_label(key, name)}: {fmt_secs(entry.get('duration_s'))}"
                )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["training_run_status"] = self.training_run_status.value
        data["succeeded"] = self.succeeded
        return data

    @classmethod
    def from_dict(cls, data: dict) -> TutorialResult:
        return cls(
            base_model_name=data["base_model_name"],
            step_count=data["step_count"],
            training_run_id=data["training_run_id"],
            training_run_status=TrainingRunStatus(data["training_run_status"]),
            total_duration_s=data["total_duration_s"],
            step_times=data.get("step_times"),
            substep_times=data.get("substep_times"),
            modal_app_id=data.get("modal_app_id", ""),
            modal_app_url=data.get("modal_app_url", ""),
        )


def fmt_secs(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    n = float(seconds)
    if n >= 60:
        minutes = int(n // 60)
        rem = n - minutes * 60
        return f"{minutes}m {rem:.3f}s"
    return f"{n:.3f}s"


def phase_label(step_key: str, name: str) -> str:
    labels = {
        "evaluate_rollouts": "Eval (before)",
        "generate_rollouts": "Generate rollouts",
        "offload_rollout": "Offload rollout",
        "compute_log_probs": "Compute log probs",
        "optimizer_step": "Optimizer step",
        "checkpoint_save": "Checkpoint save",
        "offload_train": "Offload train",
        "weight_sync": "Weight sync",
        "evaluate_rollouts_end": "Eval (after)",
    }
    return f"{labels.get(name, name.replace('_', ' '))} (step {step_key})"


def _step_keys(result: TutorialResult) -> list[str]:
    keys = set(result.step_times or {}) | set(result.substep_times or {})
    return sorted(keys, key=lambda k: int(k) if k.isdigit() else k)


def _ordered_substeps(
    subs: dict[str, dict[str, float | None]],
) -> list[tuple[str, dict[str, float | None]]]:
    return sorted(
        subs.items(),
        key=lambda item: (
            item[1].get("start") is None,
            item[1].get("start") or 0,
        ),
    )


def total_step_time_s(result: TutorialResult) -> float:
    return float(
        sum(step.get("duration_s") or 0 for step in (result.step_times or {}).values())
    )


class Gsm8kDataset(HuggingFaceDataset):
    hf_repo = "openai/gsm8k"
    hf_config = "main"
    input_column = "question"
    output_column = "answer"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True

    def load(self, split: str = "all"):
        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds.map(lambda r: {"answer": r["answer"].split("####")[-1].strip()})


class LibriSpeechASRDataset(MultimodalDataset):
    modality = "audio"
    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"
    n_rows = 8
    always_prepare = True
    apply_chat_template = False

    _INSTRUCTION = (
        "<audio>\nTranscribe the speech to text. Respond with only the transcript."
    )

    def __init__(self, **kwargs):
        super().__init__(rows=[], **kwargs)

    def _build_rows(self) -> list[dict]:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "soundfile is required for audio model validation"
            ) from exc
        from datasets import Audio

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.select(range(min(self.n_rows, len(ds))))
        ds = ds.cast_column("audio", Audio(decode=False))
        rows = []
        for ex in ds:
            audio = ex["audio"]
            data = (
                audio["bytes"]
                if audio.get("bytes")
                else open(audio["path"], "rb").read()
            )
            arr, sr = sf.read(io.BytesIO(data))
            buf = io.BytesIO()
            sf.write(buf, arr, sr, format="WAV")
            data_uri = "data:audio/wav;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
            rows.append(
                {
                    self.input_key: self._INSTRUCTION,
                    self.media_column: [data_uri],
                    self.label_key: ex["text"].lower().strip(),
                }
            )
        return rows

    def load(self, split: str = "all") -> list[dict]:
        return self._build_rows()

    def prepare(self, path, eval_paths=None):
        rows = self._build_rows()
        self._write_jsonl(rows, path)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_jsonl(rows, eval_path)


def available_model_names() -> list[str]:
    return sorted(name for name, _cls in VALIDATABLE_MODELS)


def get_model_config_from_model_name(model_name: str) -> ModelConfig:
    registry: dict[str, type[ModelConfig]] = {}
    for name, model_config in VALIDATABLE_MODELS:
        registry[name.lower()] = model_config
        registry[model_config.model_name.lower()] = model_config
    config_cls = registry.get(model_name.lower())
    if config_cls is None:
        available = sorted({cls.model_name for cls in registry.values()})
        raise ValueError(
            f"unknown model {model_name!r}; available: {', '.join(available)}"
        )
    return config_cls()


@dataclass(frozen=True)
class ValidationPlan:
    model_name: str
    step_count: int
    model_config: ModelConfig
    dataset: DatasetConfig
    train_recipe: SlimeRecipe


def build_validation_plan(
    model_name: str,
    *,
    step_count: int = 1,
    eval_interval: int | None = None,
    save_interval: int | None = None,
    colocate: bool | None = None,
) -> ValidationPlan:
    model_config = get_model_config_from_model_name(model_name)
    if isinstance(model_config, Qwen3_ASR_1_7B):
        dataset: DatasetConfig = LibriSpeechASRDataset(n_rows=8)
    else:
        dataset = Gsm8kDataset(n_rows=10)
    train_recipe = SlimeRecipe.get_base_recipe(model_config)
    train_recipe.num_rollout = step_count
    if colocate is not None:
        train_recipe.colocate = colocate
    if colocate is False and train_recipe.rollout_num_gpus is None:
        train_recipe.rollout_num_gpus = (
            train_recipe.actor_num_nodes * train_recipe.actor_num_gpus_per_node
        )
    if eval_interval is not None:
        train_recipe.eval_interval = eval_interval
    if save_interval is not None:
        train_recipe.save_interval = save_interval
    train_recipe.rm_type = "deepscaler"
    train_recipe.train_function_kwargs = {
        **dict(train_recipe.train_function_kwargs or {}),
        "ephemeral_disk": VALIDATION_EPHEMERAL_DISK_MIB,
    }
    return ValidationPlan(
        model_name=model_name,
        step_count=step_count,
        model_config=model_config,
        dataset=dataset,
        train_recipe=train_recipe,
    )


def validation_config_digest(plan: ValidationPlan) -> str:
    payload = {
        "model": plan.model_config.model_name,
        "dataset": {
            "type": type(plan.dataset).__name__,
            "hf_repo": getattr(plan.dataset, "hf_repo", ""),
            "hf_config": getattr(plan.dataset, "hf_config", ""),
            "hf_split": getattr(plan.dataset, "hf_split", ""),
            "n_rows": getattr(plan.dataset, "n_rows", None),
        },
        "ephemeral_disk_mib": VALIDATION_EPHEMERAL_DISK_MIB,
        "gpu_type": plan.train_recipe.gpu_type,
        "recipe": serialize_recipe_params(
            plan.train_recipe,
            dataset=plan.dataset,
            model=plan.model_config,
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def run_validation_plan(
    plan: ValidationPlan,
    *,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_secret_name: str = "wandb-secret",
    group_id: str | None = None,
    group_overrides: dict[str, Any] | None = None,
    group_axes: list[str] | None = None,
    result_deadline: float | None = None,
) -> TutorialResult:
    train_recipe = plan.train_recipe
    ds_key = getattr(plan.dataset, "hf_repo", type(plan.dataset).__name__).rsplit(
        "/", 1
    )[-1]
    model_short_name = plan.model_config.model_name.rsplit("/", 1)[-1]
    if wandb_project is not None:
        train_recipe.wandb = WandbConfig(
            project=wandb_project or f"model-validation-{model_short_name}-{ds_key}",
            group=wandb_group or f"model-validator-{model_short_name}-{ds_key}",
            modal_wandb_secret_name=wandb_secret_name,
        )

    train_config = TrainConfig(
        model=plan.model_config,
        dataset=plan.dataset,
        recipe=train_recipe,
        group_id=group_id,
        group_overrides=group_overrides,
        group_axes=group_axes,
    )

    run = train_config.launch(prepare_inputs=True)
    try:
        result_timeout = None
        if result_deadline is not None:
            result_timeout = result_deadline - time.time()
            if result_timeout <= 0:
                raise TimeoutError("validation launch exceeded its result deadline")
        train_result = run.result(
            timeout=result_timeout,
            stop_app_on_success=False,
        )
    except BaseException:
        stop_app(run.modal_app_id)
        raise
    stop_app(run.modal_app_id)

    training_run = TrainingRun.from_id(train_result.training_run_id)
    return TutorialResult(
        base_model_name=plan.model_name,
        step_count=plan.step_count,
        training_run_id=train_result.training_run_id,
        training_run_status=training_run.status,
        total_duration_s=float(training_run.duration_seconds or 0.0),
        step_times=training_run.step_times,
        substep_times=training_run.substep_times,
        modal_app_id=run.modal_app_id or training_run.modal_app_id or "",
        modal_app_url=run.modal_app_url or training_run.modal_app_url or "",
    )


def status_label(result: TutorialResult) -> str:
    if result.succeeded:
        return "✅ completed"
    return f"❌ {result.training_run_status.value}"


def phase_timing_rows(
    result: TutorialResult,
) -> list[tuple[str, float | int | None]]:
    rows: list[tuple[str, float | int | None]] = []
    for key in _step_keys(result):
        step = (result.step_times or {}).get(key) or {}
        for name, entry in _ordered_substeps(
            (result.substep_times or {}).get(key) or {}
        ):
            rows.append((phase_label(key, name), entry.get("duration_s")))
        rows.append((f"Step {key}", step.get("duration_s")))
    rows.append(("Total duration", result.total_duration_s))
    return rows
