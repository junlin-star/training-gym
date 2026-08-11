"""Gemma-4-26B-A4B validation runs: two text paths and two vision-language paths.

Each task pairs a dataset with a reward built to leave the model *room to
improve*. The first pass of these runs scored 0.875-1.0 on GSM8K, so every GRPO
group was zero-variance and `rollout/advantages` sat at 0.000 for all 15 steps —
the run proved the plumbing and taught the model nothing. The rewards here aim
for a mean nearer 0.5 by combining harder rows with an answer format the model
does not satisfy for free.

Usage:

    uv run modal run scripts/validate_gemma4_runs.py --task gsm8k
    uv run modal run scripts/validate_gemma4_runs.py --task screenspot --steps 15
"""

import argparse
import base64
import io
import re

import modal

from modal_training_gym.common.dataset import HuggingFaceDataset, MultimodalDataset
from modal_training_gym.common.models import Gemma4_26B_A4B
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train import TrainConfig
from modal_training_gym.train_recipes.miles_recipe import Gemma4_26B_A4B_Recipe

WANDB_PROJECT = "gemma4-26b-a4b-validation"

# Answers must land in \boxed{}. The instruction is explicit, so a miss is a real
# failure to follow it rather than noise, and correct-but-unboxed still scores
# above wrong so the gradient does not collapse to the format alone.
_BOXED_INSTRUCTION = (
    "Solve the problem. Reason step by step, then give the final answer as "
    "\\boxed{answer} on the last line."
)

_CHOICE_INSTRUCTION = (
    "Answer the multiple-choice question. Reason step by step, then give the "
    "letter of the correct option as \\boxed{letter} on the last line."
)

_CHART_INSTRUCTION = (
    "Answer the question about the chart. Reply with only the answer, as "
    "\\boxed{answer}."
)

_GROUNDING_INSTRUCTION = (
    "Point at the described element in the screenshot. Reply with only the "
    "click position as normalized coordinates \\boxed{{x,y}}, each between 0 "
    "and 1.\n\nElement: {instruction}"
)


# ── Answer extraction ────────────────────────────────────────────────────────


def _boxed(text: str) -> str | None:
    """Contents of the last ``\\boxed{...}``, brace-balanced."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    depth, i = 0, start + len("\\boxed{") - 1
    for i in range(start + len("\\boxed{") - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{") : i].strip()
    return None


def _as_number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _last_number(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


# ── Datasets ─────────────────────────────────────────────────────────────────


class Gsm8kHardDataset(HuggingFaceDataset):
    """GSM8K restricted to its longest chains of reasoning.

    The base model clears short GSM8K problems almost every time. Row selection
    keys on the number of calculator annotations in the reference solution,
    which is the dataset's own proxy for how many steps the answer takes.
    """

    hf_repo = "openai/gsm8k"
    hf_config = "main"
    hf_split = "train"
    input_column = "question"
    output_column = "answer"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True
    min_steps = 5

    def load(self, split: str = "all"):
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.filter(lambda r: r["answer"].count("<<") >= self.min_steps)
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds.map(
            lambda r: {
                "question": f"{_BOXED_INSTRUCTION}\n\n{r['question']}",
                "answer": r["answer"].split("####")[-1].strip(),
            }
        )


class AquaRatDataset(HuggingFaceDataset):
    """AQuA-RAT algebraic word problems, five options each."""

    hf_repo = "deepmind/aqua_rat"
    hf_config = "raw"
    hf_split = "train"
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

        def _format(row):
            options = "\n".join(row["options"])
            return {
                "question": (
                    f"{_CHOICE_INSTRUCTION}\n\n{row['question']}\n\n{options}"
                ),
                "answer": row["correct"].strip().upper(),
            }

        return ds.map(_format)


class ScreenSpotDataset(MultimodalDataset):
    """GUI grounding: click the described element in a screenshot."""

    modality = "image"
    hf_repo = "rootsautomation/ScreenSpot"
    hf_split = "test"
    n_rows = 400
    row_offset = 0
    always_prepare = True
    apply_chat_template = True

    def __init__(self, **kwargs):
        super().__init__(rows=[], **kwargs)

    def _build_rows(self) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, split=self.hf_split)
        start = min(self.row_offset, len(ds))
        stop = min(start + self.n_rows, len(ds))
        rows = []
        for row in ds.select(range(start, stop)):
            left, top, right, bottom = row["bbox"]
            buf = io.BytesIO()
            row["image"].save(buf, format="PNG")
            data_uri = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
            rows.append(
                {
                    self.input_key: _GROUNDING_INSTRUCTION.format(
                        instruction=row["instruction"]
                    ),
                    self.media_column: [data_uri],
                    self.label_key: f"{left:.4f},{top:.4f},{right:.4f},{bottom:.4f}",
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


class ChartQADataset(MultimodalDataset):
    """ChartQA: read a value or comparison off a chart image."""

    modality = "image"
    hf_repo = "HuggingFaceM4/ChartQA"
    hf_split = "train"
    n_rows = 400
    row_offset = 0
    always_prepare = True
    apply_chat_template = True

    def __init__(self, **kwargs):
        super().__init__(rows=[], **kwargs)

    def _build_rows(self) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, split=self.hf_split)
        start = min(self.row_offset, len(ds))
        stop = min(start + self.n_rows, len(ds))
        rows = []
        for row in ds.select(range(start, stop)):
            answer = row["label"]
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            buf = io.BytesIO()
            row["image"].convert("RGB").save(buf, format="PNG")
            data_uri = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
            rows.append(
                {
                    self.input_key: f"{_CHART_INSTRUCTION}\n\n{row['query']}",
                    self.media_column: [data_uri],
                    self.label_key: str(answer).strip(),
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


# ── Rewards ──────────────────────────────────────────────────────────────────
#
# Every reward grades on two axes so a group keeps some spread even when all
# eight samples agree on correctness: whether the answer is right, and whether
# it arrived in the requested \boxed{} form.


def score_numeric(response: str, label: str) -> float:
    target = _as_number(label)
    if target is None:
        return 0.0
    boxed = _boxed(response)
    if boxed is not None:
        got = _as_number(boxed)
        if got is not None:
            return 1.0 if abs(got - target) < 1e-4 else 0.0
    fallback = _last_number(response)
    if fallback is None:
        return -0.5
    # Right answer, wrong shape: worth more than a wrong answer, less than a
    # clean one, so following the format stays worth learning.
    return 0.3 if abs(fallback - target) < 1e-4 else 0.0


def score_choice(response: str, label: str) -> float:
    target = label.strip().upper()[:1]
    if not target:
        return 0.0
    boxed = _boxed(response)
    if boxed:
        match = re.search(r"[A-E]", boxed.upper())
        if match:
            return 1.0 if match.group() == target else 0.0
    tail = re.findall(r"\b([A-E])\b", response.upper())
    if not tail:
        return -0.5
    return 0.3 if tail[-1] == target else 0.0


def score_chart(response: str, label: str) -> float:
    """ChartQA relaxed accuracy: numeric answers within 5%, else exact text."""
    boxed = _boxed(response)
    candidate = boxed if boxed is not None else response.strip()
    if not candidate:
        return -0.5
    target_num = _as_number(label)
    got_num = _as_number(candidate)
    if target_num is not None and got_num is not None:
        tolerance = abs(target_num) * 0.05
        hit = abs(got_num - target_num) <= max(tolerance, 1e-6)
    else:
        hit = candidate.strip().lower() == label.strip().lower()
    if not hit:
        return 0.0
    return 1.0 if boxed is not None else 0.3


def score_grounding(response: str, label: str) -> float:
    """Distance-to-box reward: +1 anywhere inside the element, decaying outside."""
    boxed = _boxed(response)
    source = boxed if boxed is not None else response
    nums = re.findall(r"-?\d*\.?\d+", source)
    if len(nums) < 2:
        return -1.0
    try:
        x, y = float(nums[0]), float(nums[1])
        left, top, right, bottom = (float(v) for v in label.split(","))
    except ValueError:
        return -1.0
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return -1.0

    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    outside = (dx * dx + dy * dy) ** 0.5
    if outside == 0.0:
        return 1.0 if boxed is not None else 0.6
    diagonal = ((right - left) ** 2 + (bottom - top) ** 2) ** 0.5
    margin = max(diagonal, 0.05)
    if outside >= margin:
        return -1.0
    return 1.0 - 2.0 * outside / margin


async def gsm8k_reward(args, sample, **kwargs) -> float:
    return score_numeric(
        getattr(sample, "response", "") or "", getattr(sample, "label", "") or ""
    )


async def aqua_reward(args, sample, **kwargs) -> float:
    return score_choice(
        getattr(sample, "response", "") or "", getattr(sample, "label", "") or ""
    )


async def chart_reward(args, sample, **kwargs) -> float:
    return score_chart(
        getattr(sample, "response", "") or "", getattr(sample, "label", "") or ""
    )


async def grounding_reward(args, sample, **kwargs) -> float:
    return score_grounding(
        getattr(sample, "response", "") or "", getattr(sample, "label", "") or ""
    )


# ── Tasks ────────────────────────────────────────────────────────────────────

TASKS = {
    "gsm8k": {
        "dataset": lambda: Gsm8kHardDataset(n_rows=512),
        "reward": gsm8k_reward,
        "max_response_len": 640,
    },
    "aqua": {
        "dataset": lambda: AquaRatDataset(n_rows=512),
        "reward": aqua_reward,
        "max_response_len": 640,
    },
    "screenspot": {
        "dataset": lambda: ScreenSpotDataset(n_rows=400),
        "reward": grounding_reward,
        "max_response_len": 64,
    },
    "chartqa": {
        "dataset": lambda: ChartQADataset(n_rows=400),
        "reward": chart_reward,
        "max_response_len": 96,
    },
}


cli_app = modal.App()


def _build(task: str, steps: int) -> TrainConfig:
    spec = TASKS[task]
    return TrainConfig(
        model=Gemma4_26B_A4B(),
        dataset=spec["dataset"](),
        recipe=Gemma4_26B_A4B_Recipe(
            custom_rm_function=spec["reward"],
            num_rollout=steps,
            # rm_type is the text path's maths default; these tasks all bring
            # their own reward, and leaving it set would double-score them.
            rm_type=None,
            rollout_max_response_len=spec["max_response_len"],
            save_interval=max(steps, 1),
            no_save_optim=True,
            wandb=WandbConfig(project=WANDB_PROJECT, group=task),
        ),
    )


def _run(task: str, steps: int) -> None:
    result = _build(task, steps).train()
    print(f"task={task} training_run_id={result.training_run_id}")


@cli_app.local_entrypoint()
def main(task: str = "gsm8k", steps: int = 15) -> None:
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; pick one of {sorted(TASKS)}")
    _run(task, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="gsm8k", choices=sorted(TASKS))
    parser.add_argument("--steps", type=int, default=15)
    parsed = parser.parse_args()
    _run(parsed.task, parsed.steps)
