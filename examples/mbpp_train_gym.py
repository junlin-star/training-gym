"""Train Qwen3 models on MBPP with Training Gym and sandbox rewards.

This script uses the same Modal Sandbox scorer as ``mbpp_eval_dashboard.py``,
but materializes a deterministic train/test split for SLIME GRPO training. The
training reward sees only train rows; the post-training dashboard eval uses the
held-out test split.

Usage:

    # Short smoke train for one model, then eval base and trained checkpoints:
    uv run python examples/mbpp_train_gym.py --models Qwen3_0_6B

    # Run all three model training/eval jobs from one process:
    uv run python examples/mbpp_train_gym.py \
        --models Qwen3_0_6B Qwen3_1_7B Qwen3_4B
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json
import os
import random
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from modal_training_gym import (
    DatasetConfig,
    DeploymentConfig,
    EvalConfigDurable,
    EvalResult,
    EvalRowResult,
    Qwen3_0_6B,
    Qwen3_1_7B,
    Qwen3_4B,
    SlimeRecipe,
    TrainConfig,
    extract_code,
    list_checkpoints,
)
from modal_training_gym.common.ids import create_hash
from huggingface_hub import hf_hub_download

MBPP_REPO = "Muennighoff/mbpp"
PROMPT_STYLE = "correctness-first-brevity-v1"
DEFAULT_BREVITY_WEIGHT = 0.1
SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the task by writing the "
    "shortest correct Python implementation you can. Correctness is required; "
    "brevity only matters after all public tests pass. Return only executable "
    "Python code."
)

MODEL_REGISTRY: dict[str, type] = {
    "Qwen3_0_6B": Qwen3_0_6B,
    "Qwen3_1_7B": Qwen3_1_7B,
    "Qwen3_4B": Qwen3_4B,
}

DEFAULT_SPLIT_SEED = 20260608
DEFAULT_TRAIN_SIZE = 327
DEFAULT_TEST_SIZE = 100


@dataclass(frozen=True)
class MBPPTask:
    task_id: int
    text: str
    reference_code: str
    test_setup_code: str
    test_list: tuple[str, ...]
    challenge_test_list: tuple[str, ...]


def model_slug(model_name: str) -> str:
    return model_name.lower().replace("_", "-")


def _str(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(_str(item) for item in value)
    return ()


def _task_from_sanitized(row: dict[str, Any]) -> MBPPTask:
    imports = row.get("test_imports", [])
    setup = "\n".join(imports) if isinstance(imports, list) else _str(imports)
    return MBPPTask(
        task_id=int(row["task_id"]),
        text=_str(row["prompt"]),
        reference_code=_str(row["code"]),
        test_setup_code=setup,
        test_list=_str_tuple(row["test_list"]),
        challenge_test_list=(),
    )


def _task_from_full(row: dict[str, Any]) -> MBPPTask:
    return MBPPTask(
        task_id=int(row["task_id"]),
        text=_str(row["text"]),
        reference_code=_str(row["code"]),
        test_setup_code=_str(row.get("test_setup_code")),
        test_list=_str_tuple(row["test_list"]),
        challenge_test_list=_str_tuple(row.get("challenge_test_list")),
    )


def load_mbpp_tasks(limit: int = 10, subset: str = "sanitized") -> list[MBPPTask]:
    if subset == "sanitized":
        path = hf_hub_download(
            MBPP_REPO, "data/sanitized-mbpp.json", repo_type="dataset"
        )
        with open(path) as f:
            rows: list[dict[str, Any]] = json.load(f)
        return [_task_from_sanitized(r) for r in rows[:limit]]

    path = hf_hub_download(MBPP_REPO, "data/mbpp.jsonl", repo_type="dataset")
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [_task_from_full(r) for r in rows[:limit]]


def build_prompt(task: MBPPTask) -> str:
    visible_tests = "\n".join(f"- `{t}`" for t in task.test_list)
    return (
        "/no_think\n"
        f"{SYSTEM_PROMPT} Start with a line containing "
        "exactly ```python and end with a line containing exactly ```. "
        "Do not explain your solution or include comments unless required.\n\n"
        f"Task {task.task_id}:\n{task.text}\n\n"
        f"Public tests that your code must pass:\n{visible_tests}"
    )


def extract_mbpp_code(completion: str) -> str:
    code = extract_code(completion)
    lines = code.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```python") and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip()
    if len(lines) >= 2 and lines[0] == "```" and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip()
    return code


def correctness_first_brevity_reward(
    *,
    passed: int,
    total: int,
    completion_chars: int,
    reference_chars: int,
    brevity_weight: float = DEFAULT_BREVITY_WEIGHT,
) -> tuple[float, dict[str, float | bool | int]]:
    if total <= 0:
        return 0.0, {
            "pass_frac": 0.0,
            "all_tests_passed": False,
            "brevity_bonus": 0.0,
            "completion_chars": completion_chars,
            "reference_chars": reference_chars,
        }

    pass_frac = passed / total
    all_tests_passed = passed == total
    if not all_tests_passed:
        return pass_frac, {
            "pass_frac": pass_frac,
            "all_tests_passed": False,
            "brevity_bonus": 0.0,
            "completion_chars": completion_chars,
            "reference_chars": reference_chars,
        }

    brevity_ratio = reference_chars / max(1, completion_chars)
    brevity_bonus = brevity_weight * min(1.0, brevity_ratio)
    return 1.0 + brevity_bonus, {
        "pass_frac": pass_frac,
        "all_tests_passed": True,
        "brevity_bonus": brevity_bonus,
        "completion_chars": completion_chars,
        "reference_chars": reference_chars,
    }


def reward_brevity_weight() -> float:
    return float(os.environ.get("MBPP_BREVITY_WEIGHT", DEFAULT_BREVITY_WEIGHT))


def run_mbpp_asserts_in_sandbox(
    *,
    code: str,
    task: MBPPTask,
    app_name: str = "training-gym-mbpp-openenv",
    timeout_sec: int = 10,
    sandbox_cpu: float = 1.0,
    sandbox_memory: int = 1024,
    python_version: str = "3.12",
) -> dict[str, object]:
    import modal

    runner = r"""
import json
import sys
import traceback

payload = json.loads(sys.argv[1])
namespace = {}
results = []
stdout = ""
stderr = ""
error = ""

try:
    setup_code = payload["test_setup_code"]
    solution_code = payload["code"]
    if setup_code:
        exec(compile(setup_code, "<mbpp-test-setup>", "exec"), namespace)
    exec(compile(solution_code, "<mbpp-solution>", "exec"), namespace)
    for assertion in payload["test_list"]:
        try:
            exec(compile(assertion, "<mbpp-assert>", "exec"), namespace)
            results.append({"assertion": assertion, "passed": True, "error": ""})
        except BaseException:
            results.append({
                "assertion": assertion,
                "passed": False,
                "error": traceback.format_exc(limit=2),
            })
except BaseException:
    error = traceback.format_exc(limit=5)

print(json.dumps({
    "results": results,
    "stdout": stdout,
    "stderr": stderr,
    "error": error,
}))
"""
    payload = json.dumps(
        {
            "code": code,
            "test_setup_code": task.test_setup_code,
            "test_list": list(task.test_list),
        }
    )

    app = modal.App.lookup(app_name, create_if_missing=True)
    image = modal.Image.debian_slim(python_version=python_version)
    try:
        sandbox = modal.Sandbox.create(
            "python",
            "-c",
            runner,
            payload,
            image=image,
            cpu=sandbox_cpu,
            memory=sandbox_memory,
            timeout=timeout_sec,
            app=app,
        )
        sandbox.wait()
        raw_stdout = sandbox.stdout.read()
        raw_stderr = sandbox.stderr.read()
    except Exception as exc:
        return {
            "passed": 0,
            "total": len(task.test_list),
            "stdout": "",
            "stderr": "",
            "error": f"{type(exc).__name__}: {exc}",
            "per_test": [],
        }

    try:
        decoded = json.loads(raw_stdout)
    except json.JSONDecodeError:
        return {
            "passed": 0,
            "total": len(task.test_list),
            "stdout": raw_stdout,
            "stderr": raw_stderr,
            "error": "sandbox returned non-JSON output",
            "per_test": [],
        }

    result_objects = decoded.get("results")
    per_test = result_objects if isinstance(result_objects, list) else []
    passed = sum(
        1 for result in per_test if isinstance(result, dict) and result.get("passed")
    )
    error_value = decoded.get("error")
    stdout_value = decoded.get("stdout")
    stderr_value = decoded.get("stderr")
    return {
        "passed": passed,
        "total": len(task.test_list),
        "stdout": stdout_value if isinstance(stdout_value, str) else "",
        "stderr": raw_stderr + (stderr_value if isinstance(stderr_value, str) else ""),
        "error": error_value if isinstance(error_value, str) else "",
        "per_test": per_test,
    }


def evaluate_one(
    task: MBPPTask,
    deployment,
    *,
    app_name: str,
    idx: int,
    total: int,
) -> EvalRowResult:
    prompt = build_prompt(task)
    response = deployment.generate(
        prompt,
        ensure_ready=False,
        max_tokens=512,
        temperature=0.0,
    )
    code = extract_mbpp_code(response)
    result = run_mbpp_asserts_in_sandbox(
        code=code,
        task=task,
        app_name=app_name,
        timeout_sec=10,
    )
    passed = int(result["passed"])
    total_tests = int(result["total"])
    reward, reward_parts = correctness_first_brevity_reward(
        passed=passed,
        total=total_tests,
        completion_chars=len(code.strip()),
        reference_chars=len(task.reference_code.strip()),
    )
    print(
        f"  [{idx}/{total}] task={task.task_id} "
        f"passed={passed}/{total_tests} reward={reward:.4f} "
        f"chars={len(code.strip())}"
    )
    return EvalRowResult(
        score=reward,
        response=response,
        prompt=prompt,
        metadata={
            "task_id": task.task_id,
            "passed": passed,
            "total": total_tests,
            "all_tests_passed": passed == total_tests and total_tests > 0,
            "completion_chars": len(code.strip()),
            "extracted_code": code,
            "reward_parts": reward_parts,
        },
    )


def split_mbpp_tasks(
    *,
    subset: str,
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[MBPPTask], list[MBPPTask]]:
    tasks = load_mbpp_tasks(limit=10_000, subset=subset)
    tasks = sorted(tasks, key=lambda task: task.task_id)
    random.Random(seed).shuffle(tasks)
    if train_size + test_size > len(tasks):
        raise ValueError(
            f"Requested train_size + test_size = {train_size + test_size}, "
            f"but MBPP {subset!r} only has {len(tasks)} tasks."
        )
    return tasks[:train_size], tasks[train_size : train_size + test_size]


def task_to_label(task: MBPPTask) -> str:
    return json.dumps(dataclasses.asdict(task), separators=(",", ":"))


def task_from_label(label: str) -> MBPPTask:
    data = json.loads(label)
    return MBPPTask(
        task_id=int(data["task_id"]),
        text=str(data["text"]),
        reference_code=str(data["reference_code"]),
        test_setup_code=str(data["test_setup_code"]),
        test_list=tuple(str(item) for item in data["test_list"]),
        challenge_test_list=tuple(str(item) for item in data["challenge_test_list"]),
    )


class MBPPSplitDataset(DatasetConfig):
    input_key = "messages"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True

    def __init__(
        self,
        *,
        subset: str,
        train_size: int,
        test_size: int,
        split_seed: int,
        train_repeats: int,
        path_suffix: str,
    ) -> None:
        self.subset = subset
        self.train_size = train_size
        self.test_size = test_size
        self.split_seed = split_seed
        self.train_repeats = train_repeats
        self.eval_repeats = 1
        self.path_suffix = path_suffix
        split_name = f"mbpp-{subset}-seed{split_seed}-train{train_size}-test{test_size}"
        self.hf_repo = f"{split_name}-{path_suffix}-{PROMPT_STYLE}"
        super().__init__(dataset_id=f"{split_name}-{PROMPT_STYLE}")

    def load(
        self, split: Literal["all", "train", "eval"] = "all"
    ) -> list[MBPPTask] | tuple[list[MBPPTask], list[MBPPTask]]:
        train_tasks, test_tasks = split_mbpp_tasks(
            subset=self.subset,
            train_size=self.train_size,
            test_size=self.test_size,
            seed=self.split_seed,
        )
        if split == "train":
            return train_tasks
        if split == "eval":
            return test_tasks
        return train_tasks, test_tasks

    @staticmethod
    def _row(task: MBPPTask) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": build_prompt(task)}],
            "label": task_to_label(task),
        }

    @staticmethod
    def _write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        train_tasks = self.load("train")
        test_tasks = self.load("eval")
        assert isinstance(train_tasks, list)
        assert isinstance(test_tasks, list)

        train_rows = [
            self._row(task)
            for task in train_tasks
            for _ in range(max(1, int(self.train_repeats)))
        ]
        eval_rows = [self._row(task) for task in test_tasks]

        self._write_jsonl(path, train_rows)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_jsonl(eval_path, eval_rows)


async def mbpp_rm(args, sample, **kwargs) -> float:
    task = task_from_label(sample.label)
    code = extract_mbpp_code(sample.response)
    result = await asyncio.to_thread(
        run_mbpp_asserts_in_sandbox,
        code=code,
        task=task,
        app_name="mbpp-train-sandbox",
        timeout_sec=10,
    )
    passed = int(result["passed"])
    total = int(result["total"])
    reward, reward_parts = correctness_first_brevity_reward(
        passed=passed,
        total=total,
        completion_chars=len(code.strip()),
        reference_chars=len(task.reference_code.strip()),
        brevity_weight=reward_brevity_weight(),
    )
    sample.metadata = {
        **(getattr(sample, "metadata", None) or {}),
        "mbpp": {
            "task_id": task.task_id,
            "passed": passed,
            "total": total,
            "all_tests_passed": passed == total and total > 0,
            "completion_chars": len(code.strip()),
            "reward_parts": reward_parts,
        },
    }
    return float(reward)


def build_recipe(args: argparse.Namespace) -> SlimeRecipe:
    return SlimeRecipe(
        custom_rm_function=mbpp_rm,
        gpu_type=args.gpu_type,
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=args.num_rollout,
        rollout_batch_size=args.rollout_batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
        rollout_max_response_len=args.max_response_len,
        rollout_temperature=args.temperature,
        rollout_stop_token_ids=[128001],
        global_batch_size=args.global_batch_size,
        save_interval=args.save_interval,
        eval_interval=args.save_interval,
        eval_max_response_len=args.max_response_len,
        n_samples_per_eval_prompt=args.n_samples_per_eval_prompt,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        apply_chat_template_kwargs='{"enable_thinking": false}',
        app_tags={
            "task": "mbpp",
            "split_seed": str(args.split_seed),
            "prompt_style": PROMPT_STYLE,
        },
        train_env_vars={"MBPP_BREVITY_WEIGHT": str(args.brevity_weight)},
        image_overlay=lambda image: image.run_commands(
            "uv pip install --system modal>=1.4.0"
        ),
    )


def save_eval_result(
    *,
    model_key: str,
    deployment,
    rows: list[EvalRowResult],
    split_seed: int,
    train_size: int,
    test_size: int,
    eval_kind: str,
    training_run_id: str = "",
    brevity_weight: float = DEFAULT_BREVITY_WEIGHT,
    run_label: str = "",
) -> EvalResult:
    dataset_name = f"mbpp-sanitized-seed{split_seed}-test{test_size}"
    eval_config_id = create_hash(
        "eval-config",
        "MBPPTrainEval",
        model_key,
        "mbpp_sandbox",
        (
            f"{dataset_name}:{PROMPT_STYLE}:{eval_kind}:{training_run_id}:"
            f"{brevity_weight}:{run_label}"
        ),
    )
    EvalConfigDurable(
        eval_config_id=eval_config_id,
        dataset_name=dataset_name,
        eval_fn_name="mbpp_sandbox",
        generate_kwargs={
            "model": model_key,
            "eval_kind": eval_kind,
            "training_run_id": training_run_id,
            "split_seed": split_seed,
            "train_size": train_size,
            "test_size": test_size,
            "prompt_style": PROMPT_STYLE,
            "brevity_weight": brevity_weight,
            "run_label": run_label,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ).save()
    eval_id = create_hash(
        "eval",
        eval_config_id,
        deployment.deployment_id,
        training_run_id,
        eval_kind,
    )
    result = EvalResult(
        eval_id=eval_id,
        eval_config_id=eval_config_id,
        deployment_id=deployment.deployment_id,
        created_at=datetime.datetime.now(datetime.UTC),
        rows=rows,
    )
    result.save()
    return result


def run_dashboard_eval(
    *,
    model_key: str,
    deployment,
    tasks: list[MBPPTask],
    split_seed: int,
    train_size: int,
    test_size: int,
    eval_kind: str,
    max_concurrency: int,
    training_run_id: str = "",
    brevity_weight: float = DEFAULT_BREVITY_WEIGHT,
    run_label: str = "",
) -> EvalResult:
    rows: list[EvalRowResult] = []
    total = len(tasks)

    def _eval_indexed(item: tuple[int, MBPPTask]) -> EvalRowResult:
        idx, task = item
        return evaluate_one(
            task,
            deployment,
            app_name=f"mbpp-{eval_kind}-{model_slug(model_key)}",
            idx=idx,
            total=total,
        )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        for row in executor.map(_eval_indexed, enumerate(tasks, 1)):
            rows.append(row)

    result = save_eval_result(
        model_key=model_key,
        deployment=deployment,
        rows=rows,
        split_seed=split_seed,
        train_size=train_size,
        test_size=test_size,
        eval_kind=eval_kind,
        training_run_id=training_run_id,
        brevity_weight=brevity_weight,
        run_label=run_label,
    )
    all_pass = sum(1 for row in rows if row.metadata.get("all_tests_passed"))
    avg_chars = sum(int(row.metadata.get("completion_chars", 0)) for row in rows) / len(
        rows
    )
    print(
        f"{model_key} {eval_kind}: mean={result.mean:.4f} "
        f"all_pass={all_pass}/{len(rows)} avg_chars={avg_chars:.1f} "
        f"eval_id={result.eval_id}"
    )
    return result


def eval_summary(result: EvalResult) -> dict[str, float | int | str]:
    all_pass = sum(1 for row in result.rows if row.metadata.get("all_tests_passed"))
    avg_chars = sum(
        int(row.metadata.get("completion_chars", 0)) for row in result.rows
    ) / len(result.rows)
    passing_rows = [row for row in result.rows if row.metadata.get("all_tests_passed")]
    passing_avg_chars = (
        sum(int(row.metadata.get("completion_chars", 0)) for row in passing_rows)
        / len(passing_rows)
        if passing_rows
        else 0.0
    )
    return {
        "eval_id": result.eval_id,
        "mean": result.mean,
        "all_pass": all_pass,
        "total": len(result.rows),
        "avg_chars": avg_chars,
        "passing_avg_chars": passing_avg_chars,
    }


def train_and_eval_model(model_key: str, args: argparse.Namespace) -> dict[str, object]:
    model_cls = MODEL_REGISTRY[model_key]
    model = model_cls()
    slug = model_slug(model_key)
    run_slug = slug
    if args.run_label:
        run_slug = f"{slug}-{args.run_label.lower().replace('_', '-')}"
    train_tasks, test_tasks = split_mbpp_tasks(
        subset=args.subset,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.split_seed,
    )
    print(
        f"{model_key}: train={len(train_tasks)} test={len(test_tasks)} "
        f"split_seed={args.split_seed}"
    )

    base_eval_summary: dict[str, float | int | str] | None = None
    if not args.skip_base_eval:
        print(f"{model_key}: serving base model for held-out eval")
        base_deployment = DeploymentConfig(
            model=model,
            app_name=f"{run_slug}-mbpp-base-serve",
            served_model_name=f"{run_slug}-mbpp-base",
        ).serve()
        base_deployment.wait_until_ready()
        base_eval = run_dashboard_eval(
            model_key=model_key,
            deployment=base_deployment,
            tasks=test_tasks,
            split_seed=args.split_seed,
            train_size=args.train_size,
            test_size=args.test_size,
            eval_kind="base-test",
            brevity_weight=args.brevity_weight,
            run_label=args.run_label,
            max_concurrency=args.max_concurrency,
        )
        base_eval_summary = eval_summary(base_eval)

    dataset = MBPPSplitDataset(
        subset=args.subset,
        train_size=args.train_size,
        test_size=args.test_size,
        split_seed=args.split_seed,
        train_repeats=args.train_repeats,
        path_suffix=run_slug,
    )
    training_run = TrainConfig(
        model=model,
        dataset=dataset,
        recipe=build_recipe(args),
    )
    print(f"{model_key}: starting training run {training_run.training_run_id}")
    train_result = training_run.train()
    checkpoints = list_checkpoints(train_result.training_run_id)
    if not checkpoints:
        raise RuntimeError(f"{model_key}: no checkpoints found after training")
    checkpoint = checkpoints[-1]
    print(f"{model_key}: latest checkpoint {checkpoint.path}")

    trained_deployment = DeploymentConfig(
        model=model_cls(),
        checkpoint=checkpoint,
        app_name=f"{run_slug}-mbpp-trained-serve",
        served_model_name=f"{run_slug}-mbpp-trained",
    ).serve()
    trained_deployment.wait_until_ready()
    trained_eval = run_dashboard_eval(
        model_key=model_key,
        deployment=trained_deployment,
        tasks=test_tasks,
        split_seed=args.split_seed,
        train_size=args.train_size,
        test_size=args.test_size,
        eval_kind="trained-test",
        training_run_id=train_result.training_run_id,
        brevity_weight=args.brevity_weight,
        run_label=args.run_label,
        max_concurrency=args.max_concurrency,
    )
    return {
        "model": model_key,
        "run_label": args.run_label,
        "num_rollout": args.num_rollout,
        "brevity_weight": args.brevity_weight,
        "training_run_id": train_result.training_run_id,
        "base_eval": base_eval_summary,
        "trained_eval": eval_summary(trained_eval),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B"],
        choices=list(MODEL_REGISTRY),
    )
    parser.add_argument("--subset", default="sanitized")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-repeats", type=int, default=1)
    parser.add_argument("--num-rollout", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--n-samples-per-prompt", type=int, default=4)
    parser.add_argument("--n-samples-per-eval-prompt", type=int, default=4)
    parser.add_argument("--max-response-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--brevity-weight", type=float, default=DEFAULT_BREVITY_WEIGHT)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--max-tokens-per-gpu", type=int, default=8192)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--skip-base-eval", action="store_true")
    args = parser.parse_args()

    for model_key in args.models:
        train_and_eval_model(model_key, args)


if __name__ == "__main__":
    main()
