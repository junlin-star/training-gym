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
from dataclasses import asdict, dataclass
from pathlib import Path

import modal_training_gym.common.models as models
from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.model import ModelConfig
from modal_training_gym.train import TrainConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

# TODO(melody/joy): Add more granular result per step
# @dataclass
# class StepResult:
#     step_count: int
#     step_duration_s: float # Extract from the step update time
#     substep_duration_s: dict[TrainStepStatus, float] # Extract from the substep update time


@dataclass
class TutorialResult:
    base_model_name: str
    step_count: int
    training_run_id: str
    training_run_status: TrainingRunStatus
    total_duration_s: float
    # step_results: list[StepResult]

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


def _model_config_registry() -> dict[str, type[ModelConfig]]:
    """Map normalized model names to their ModelConfig subclass.

    Keys cover both the full HF repo id ("qwen/qwen3-4b") and the short
    repo name ("qwen3-4b"), all lowercased.
    """
    registry: dict[str, type[ModelConfig]] = {}
    for obj in vars(models).values():
        if (
            inspect.isclass(obj)
            and issubclass(obj, ModelConfig)
            and getattr(obj, "model_name", "")
        ):
            full = obj.model_name.lower()
            registry[full] = obj
            registry[full.rsplit("/", 1)[-1]] = obj
    return registry


def available_model_names() -> list[str]:
    """Sorted list of short model names (e.g. "qwen3-4b") that can be validated."""
    return sorted(
        {cls.model_name.rsplit("/", 1)[-1] for cls in _model_config_registry().values()}
    )


def get_model_config_from_model_name(model_name: str) -> ModelConfig:
    registry = _model_config_registry()
    config_cls = registry.get(model_name.lower())
    if config_cls is None:
        available = sorted({cls.model_name for cls in registry.values()})
        raise ValueError(
            f"unknown model {model_name!r}; available: {', '.join(available)}"
        )
    return config_cls()


def run_base_training_on_slime(model_name: str, step_count: int = 1) -> TutorialResult:
    model_config = get_model_config_from_model_name(model_name)
    train_recipe = SlimeRecipe.get_base_recipe(model_config)
    train_recipe.num_rollout = step_count
    train_recipe.rm_type = "deepscaler"

    train_config = TrainConfig(
        model=model_config,
        dataset=Gsm8kDataset(n_rows=10),
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

    tutorial_result = run_base_training_on_slime(args.model, args.num_steps)
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
