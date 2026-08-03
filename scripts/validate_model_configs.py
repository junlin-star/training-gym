"""
Input: string [ model name ]
Output: string [ Formatted test result ]
Optional args:
    -j: json formatted output
    -o: output file path
"""

import argparse
import inspect
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import modal_training_gym.common.models as models
from modal_training_gym.common.dataset import (
    DatasetConfig,
    HuggingFaceDataset,
    MultimodalDataset,
)
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

    @property
    def succeeded(self) -> bool:
        return self.training_run_status == TrainingRunStatus.COMPLETED

    def format_tutorial_result(self) -> None:
        print(f"Training run result for {self.training_run_id}")
        print("Parameters:")
        print(f"Base model name: {self.base_model_name}")
        print(f"Step count: {self.step_count}")
        print("Result:")
        print(f"Training run status: {self.training_run_status}")
        print(f"Total duration (s): {self.total_duration_s}")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["training_run_status"] = self.training_run_status.value
        data["succeeded"] = self.succeeded
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TutorialResult":
        return cls(
            base_model_name=data["base_model_name"],
            step_count=data["step_count"],
            training_run_id=data["training_run_id"],
            training_run_status=TrainingRunStatus(data["training_run_status"]),
            total_duration_s=data["total_duration_s"],
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
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds.map(lambda r: {"answer": r["answer"].split("####")[-1].strip()})


class LibriSpeechASRDataset(MultimodalDataset):
    """LibriSpeech ASR rows (prompt + audio data-URI + transcript label).

    Mirrors the 006_audio_asr tutorial dataset: audio models can't train on
    gsm8k, so they validate against a handful of LibriSpeech clips instead.
    """

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
        import base64 as b64
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset

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
            data_uri = "data:audio/wav;base64," + b64.b64encode(buf.getvalue()).decode(
                "ascii"
            )
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


def pick_dataset(model_config: ModelConfig) -> DatasetConfig:
    """Pick a validation dataset matching the base model's modality.

    Audio models (Qwen3-ASR) need speech clips, so they get LibriSpeech;
    everything else defaults to gsm8k.
    """
    if isinstance(model_config, models.Qwen3_ASR_1_7B):
        return LibriSpeechASRDataset(n_rows=8)
    return Gsm8kDataset(n_rows=10)


# Configs reachable only through a constructor argument, so class introspection alone
# can't find them. Keyed by the variant's ``catalog_name``.
_MODEL_VARIANTS: dict[str, Callable[[], ModelConfig]] = {
    "google/gemma-4-26b-a4b-it-vl": lambda: models.Gemma4_26B_A4B(vision=True),
}


def _model_config_registry() -> dict[str, Callable[[], ModelConfig]]:
    """Map normalized model names to a factory for their ModelConfig.

    Keys cover both the full HF repo id ("qwen/qwen3-4b") and the short
    repo name ("qwen3-4b"), all lowercased.

    A variant sharing an HF repo id with its base config (Gemma-4 text vs VL) is keyed
    by its ``catalog_name`` from ``_MODEL_VARIANTS``, so both stay distinguishable
    rather than one overwriting the other.
    """
    registry: dict[str, Callable[[], ModelConfig]] = {}
    for obj in vars(models).values():
        if (
            inspect.isclass(obj)
            and issubclass(obj, ModelConfig)
            and getattr(obj, "model_name", "")
        ):
            full = (getattr(obj, "catalog_name", None) or obj.model_name).lower()
            registry[full] = obj
            registry[full.rsplit("/", 1)[-1]] = obj
    for full, factory in _MODEL_VARIANTS.items():
        registry[full] = factory
        registry[full.rsplit("/", 1)[-1]] = factory
    return registry


def _supports_slime(model_config: ModelConfig) -> bool:
    """Whether a model has a base slime recipe, the only thing this script runs.

    Derived by attempting ``SlimeRecipe.get_base_recipe`` rather than encoding
    framework support on the model — the recipe registry is the source of truth.
    """
    try:
        SlimeRecipe.get_base_recipe(model_config)
    except Exception:
        return False
    return True


def available_model_names() -> list[str]:
    """Sorted short model names (e.g. "qwen3-4b") validatable on slime.

    Excludes models with no base slime recipe (e.g. Kimi on miles), since this
    script only runs base training on slime.
    """
    names = set()
    for factory in _model_config_registry().values():
        config = factory()
        if _supports_slime(config):
            names.add(
                (getattr(config, "catalog_name", None) or config.model_name).rsplit(
                    "/", 1
                )[-1]
            )
    return sorted(names)


def get_model_config_from_model_name(model_name: str) -> ModelConfig:
    registry = _model_config_registry()
    factory = registry.get(model_name.lower())
    if factory is None:
        raise ValueError(
            f"unknown model {model_name!r}; available: "
            f"{', '.join(available_model_names())}"
        )
    return factory()


def run_base_training_on_slime(
    model_name: str,
    step_count: int = 1,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_secret_name: str = "wandb-secret",
    eval_interval: int | None = None,
    save_interval: int | None = None,
    colocate: bool | None = None,
) -> TutorialResult:
    model_config = get_model_config_from_model_name(model_name)
    if not _supports_slime(model_config):
        raise ValueError(
            f"model {model_config.model_name!r} has no base slime recipe; "
            f"validatable models: {', '.join(available_model_names())}"
        )
    dataset = pick_dataset(model_config)
    dataset_name = getattr(dataset, "hf_repo", type(dataset).__name__).rsplit("/", 1)[
        -1
    ]
    model_short_name = model_config.model_name.rsplit("/", 1)[-1]
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
    if wandb_project is not None:
        train_recipe.wandb = WandbConfig(
            project=wandb_project
            or f"model-validation-{model_short_name}-{dataset_name}",
            group=wandb_group or f"model-validator-{model_short_name}-{dataset_name}",
            modal_wandb_secret_name=wandb_secret_name,
        )

    train_config = TrainConfig(
        model=model_config,
        dataset=dataset,
        recipe=train_recipe,
    )

    train_result = train_config.train()
    training_run = TrainingRun.from_id(train_result.training_run_id)

    return TutorialResult(
        base_model_name=model_name,
        step_count=step_count,
        training_run_id=train_result.training_run_id,
        training_run_status=training_run.status,
        total_duration_s=float(training_run.duration_seconds or 0.0),
    )


def summarize_results(results_dir: str) -> str:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        result = TutorialResult.from_dict(json.loads(path.read_text()))
        status = (
            "✅ completed"
            if result.succeeded
            else f"❌ {result.training_run_status.value}"
        )
        rows.append(
            f"| {result.base_model_name} | {status} "
            f"| {result.total_duration_s:.1f}s | {result.step_count} "
            f"| `{result.training_run_id}` |"
        )

    lines = [
        "<!-- validate-models-comment -->",
        "## Model Validation Results",
        "",
        "| Model | Status | Duration | Steps | Run |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(rows or ["| _no results_ | | | | |"])
    return "\n".join(lines)


def __main__():
    parser = argparse.ArgumentParser(
        description="Validate a model config by running base training on slime."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run base training for a single model."
    )
    check_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Base model name to run training on (e.g. qwen3-4b).",
    )
    check_parser.add_argument(
        "-n",
        "--num_steps",
        type=int,
        default=1,
        help="Number of training steps (rollouts) to run. Defaults to 1.",
    )
    check_parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Override the recipe eval_interval (eval every N rollouts).",
    )
    check_parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override the recipe save_interval (checkpoint every N rollouts).",
    )
    check_parser.add_argument(
        "--non-colocated",
        action="store_true",
        help="Allocate rollout GPUs separately from trainer GPUs.",
    )
    check_parser.add_argument(
        "-o",
        "--output",
        help="Write the result as JSON to this file path.",
    )
    check_parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print the result as JSON to stdout.",
    )
    check_parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project for validator runs. If omitted, W&B logging is disabled.",
    )
    check_parser.add_argument(
        "--wandb-group",
        default="",
        help="W&B group for validator runs. Defaults to model-validator-{model}-{dataset}.",
    )
    check_parser.add_argument(
        "--wandb-secret-name",
        default="wandb-secret",
        help="Modal Secret name containing WANDB_API_KEY.",
    )
    check_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging for this validator run.",
    )

    subparsers.add_parser(
        "list", help="Print available model names as a JSON array and exit."
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Render a markdown table from a directory of result JSON files.",
    )
    summarize_parser.add_argument(
        "-d",
        "--results-dir",
        required=True,
        help="Directory containing result JSON files written by `check --output`.",
    )

    args = parser.parse_args()

    if args.command == "list":
        print(json.dumps(available_model_names()))
        return

    if args.command == "summarize":
        print(summarize_results(args.results_dir))
        return

    tutorial_result = run_base_training_on_slime(
        args.model,
        args.num_steps,
        None if args.no_wandb else args.wandb_project,
        args.wandb_group,
        args.wandb_secret_name,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        colocate=False if args.non_colocated else None,
    )
    tutorial_result.format_tutorial_result()

    if args.output:
        Path(args.output).write_text(json.dumps(tutorial_result.to_dict()))
    if args.json:
        print(json.dumps(tutorial_result.to_dict()))

    if not tutorial_result.succeeded:
        print("Training run failed")
        exit(1)
    print("Training run completed successfully")
    exit(0)


if __name__ == "__main__":
    __main__()
