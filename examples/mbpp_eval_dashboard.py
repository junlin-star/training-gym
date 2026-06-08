"""Evaluate models on MBPP and publish results to the Training Gym dashboard.

Usage:

    # Evaluate Qwen3-0.6B and Qwen3-4B on 50 sanitized MBPP tasks:
    uv run python examples/mbpp_eval_dashboard.py --limit 50

    # Single model, all 427 sanitized tasks:
    uv run python examples/mbpp_eval_dashboard.py --models Qwen3_0_6B --limit 427

Models are deployed via Modal (requires GPU), completions are scored in
Modal Sandboxes, and results are saved to the training-gym metadata volume
so they appear in the gym dashboard at https://gym.modal.dev.
"""

from __future__ import annotations

import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor

from modal_training_gym import DeploymentConfig, Qwen3_0_6B, Qwen3_1_7B, Qwen3_4B
from modal_training_gym.common.eval import (
    EvalConfigDurable,
    EvalResult,
    EvalRowResult,
)
from modal_training_gym.common.ids import create_hash

from examples.mbpp_openenv_sandbox import (
    MBPPTask,
    correctness_first_brevity_reward,
    extract_mbpp_code,
    load_mbpp_tasks,
    run_mbpp_asserts_in_sandbox,
)

MODEL_REGISTRY: dict[str, type] = {
    "Qwen3_0_6B": Qwen3_0_6B,
    "Qwen3_1_7B": Qwen3_1_7B,
    "Qwen3_4B": Qwen3_4B,
}

PROMPT_STYLE = "correctness-first-brevity-v1"
SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the task by writing the "
    "shortest correct Python implementation you can. Correctness is required; "
    "brevity only matters after all public tests pass. Return only executable "
    "Python code."
)


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


def run_eval(
    model_name: str,
    tasks: list[MBPPTask],
    *,
    subset: str,
    limit: int,
    max_concurrency: int = 4,
) -> None:
    model_cls = MODEL_REGISTRY[model_name]
    model_config = model_cls()
    print(f"\n{'=' * 60}")
    print(f"Deploying {model_name} ({model_config.model_name})...")

    deployment = DeploymentConfig(model=model_config).serve()
    deployment.wait_until_ready()
    print(f"  Ready at {deployment.url}")

    dataset_name = f"mbpp-{subset}-first-{limit}"
    eval_config_id = create_hash(
        "eval-config",
        "MBPPEval",
        model_name,
        "mbpp_sandbox",
        f"{dataset_name}:{PROMPT_STYLE}",
    )
    eval_config = EvalConfigDurable(
        eval_config_id=eval_config_id,
        dataset_name=dataset_name,
        eval_fn_name="mbpp_sandbox",
        generate_kwargs={
            "model": model_name,
            "subset": subset,
            "limit": limit,
            "prompt_style": PROMPT_STYLE,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    )
    eval_config.save()

    rows: list[EvalRowResult] = []
    total = len(tasks)

    def _eval_indexed(item: tuple[int, MBPPTask]) -> tuple[int, EvalRowResult]:
        idx, task = item
        return idx, evaluate_one(
            task,
            deployment,
            app_name=f"mbpp-eval-{model_name.lower()}",
            idx=idx,
            total=total,
        )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        for idx, result in executor.map(_eval_indexed, enumerate(tasks, 1)):
            rows.append(result)

    created_at = datetime.datetime.now(datetime.UTC)
    eval_id = create_hash(
        "eval",
        eval_config_id,
        deployment.deployment_id,
        "",
        "",
    )
    eval_result = EvalResult(
        eval_id=eval_id,
        eval_config_id=eval_config_id,
        deployment_id=deployment.deployment_id,
        created_at=created_at,
        rows=rows,
    )
    eval_result.save()

    all_pass_count = sum(1 for r in rows if r.metadata.get("all_tests_passed"))
    print(f"\n{model_name} results:")
    print(f"  Mean reward:    {eval_result.mean:.4f}")
    print(
        f"  All-pass rate:  {all_pass_count}/{total} ({100 * all_pass_count / total:.1f}%)"
    )
    print(f"  Eval ID:        {eval_id}")
    print("  Saved to dashboard ✓")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B"],
        choices=list(MODEL_REGISTRY),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--subset", default="sanitized")
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args()

    tasks = load_mbpp_tasks(limit=args.limit, subset=args.subset)
    print(f"Loaded {len(tasks)} MBPP tasks ({args.subset})")

    for model_name in args.models:
        run_eval(
            model_name,
            tasks,
            subset=args.subset,
            limit=args.limit,
            max_concurrency=args.max_concurrency,
        )

    print("\nAll evals complete. View at https://gym.modal.dev")


if __name__ == "__main__":
    main()
