"""Train Qwen3 models on MBPP with OpenEnv as the reward substrate.

This is intentionally separate from ``mbpp_train_gym.py``.  The original file
uses a direct SLIME custom reward function that calls the Modal Sandbox scorer.
This variant routes each rollout through ``MBPPCodingEnv.reset(...)`` and
``MBPPCodingEnv.step(...)`` so the training reward is produced by the OpenEnv
environment abstraction.

Usage:

    uv run python examples/mbpp_train_openenv_gym.py \
        --models Qwen3_0_6B --num-rollout 10
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from modal_training_gym import (
    DeploymentConfig,
    SlimeRecipe,
    TrainConfig,
    list_checkpoints,
)

from mbpp_openenv_sandbox import (
    MBPPCodingEnv,
    MBPPSubmitCode,
    MBPPTask as OpenEnvMBPPTask,
)
from mbpp_train_gym import (
    DEFAULT_BREVITY_WEIGHT,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    MODEL_REGISTRY,
    PROMPT_STYLE,
    MBPPTask,
    MBPPSplitDataset,
    eval_summary,
    model_slug,
    reward_brevity_weight,
    run_dashboard_eval,
    split_mbpp_tasks,
    task_from_label,
)


EXAMPLES_DIR = Path(__file__).resolve().parent


def to_openenv_task(task: MBPPTask) -> OpenEnvMBPPTask:
    return OpenEnvMBPPTask(
        task_id=task.task_id,
        text=task.text,
        reference_code=task.reference_code,
        test_setup_code=task.test_setup_code,
        test_list=task.test_list,
        challenge_test_list=task.challenge_test_list,
    )


def score_with_openenv(task: MBPPTask, completion: str) -> object:
    env = MBPPCodingEnv(
        [to_openenv_task(task)],
        app_name="mbpp-train-openenv-sandbox",
        sandbox_timeout_sec=10,
        brevity_weight=reward_brevity_weight(),
    )
    env.reset(task_id=task.task_id)
    return env.step(MBPPSubmitCode(completion=completion))


async def mbpp_openenv_rm(args, sample, **kwargs) -> float:
    timing_profile: dict[str, object] = {}
    total_start = time.perf_counter()
    task = task_from_label(sample.label)
    phase_start = time.perf_counter()
    observation = await asyncio.to_thread(score_with_openenv, task, sample.response)
    timing_profile["openenv_step_sec"] = time.perf_counter() - phase_start

    observation_metadata = observation.metadata if observation.metadata else {}
    reward_parts = observation_metadata.get("reward_parts", {})
    timing = observation_metadata.get("timing_profile", {})
    if isinstance(timing, dict):
        timing_profile["openenv"] = timing
    timing_profile["rm_total_sec"] = time.perf_counter() - total_start

    base_metadata = sample.metadata if sample.metadata else {}
    sample.metadata = {
        **base_metadata,
        "mbpp": {
            "task_id": task.task_id,
            "passed": observation.passed,
            "total": observation.total,
            "all_tests_passed": observation.all_tests_passed,
            "completion_chars": observation.completion_chars,
            "reward_parts": reward_parts,
            "timing_profile": timing_profile,
            "reward_substrate": "openenv",
        },
    }
    return float(observation.reward if observation.reward is not None else 0.0)


def add_openenv_sources(image):
    return (
        image.run_commands("uv pip install --system modal>=1.4.0 openenv==0.3.1")
        .add_local_file(
            str(EXAMPLES_DIR / "mbpp_openenv_sandbox.py"),
            remote_path="/root/mbpp_openenv_sandbox.py",
            copy=True,
        )
        .add_local_file(
            str(EXAMPLES_DIR / "mbpp_train_gym.py"),
            remote_path="/root/mbpp_train_gym.py",
            copy=True,
        )
    )


def build_openenv_recipe(args: argparse.Namespace) -> SlimeRecipe:
    return SlimeRecipe(
        custom_rm_function=mbpp_openenv_rm,
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
            "task": "mbpp-openenv",
            "split_seed": str(args.split_seed),
            "prompt_style": PROMPT_STYLE,
        },
        train_env_vars={"MBPP_BREVITY_WEIGHT": str(args.brevity_weight)},
        image_overlay=add_openenv_sources,
    )


def train_and_eval_model(model_key: str, args: argparse.Namespace) -> dict[str, object]:
    model_cls = MODEL_REGISTRY[model_key]
    model = model_cls()
    run_label = args.run_label or "openenv"
    run_slug = f"{model_slug(model_key)}-{run_label.lower().replace('_', '-')}"
    train_tasks, test_tasks = split_mbpp_tasks(
        subset=args.subset,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.split_seed,
    )
    print(
        f"{model_key}: OpenEnv training substrate; "
        f"train={len(train_tasks)} test={len(test_tasks)} split_seed={args.split_seed}"
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
            eval_kind="base-test-openenv",
            brevity_weight=args.brevity_weight,
            run_label=run_label,
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
        recipe=build_openenv_recipe(args),
    )
    print(
        f"{model_key}: starting OpenEnv-backed training {training_run.training_run_id}"
    )
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
        eval_kind="trained-test-openenv",
        training_run_id=train_result.training_run_id,
        brevity_weight=args.brevity_weight,
        run_label=run_label,
        max_concurrency=args.max_concurrency,
    )
    return {
        "model": model_key,
        "run_label": run_label,
        "reward_substrate": "openenv",
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
        default=["Qwen3_0_6B"],
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
    parser.add_argument("--run-label", default="openenv")
    parser.add_argument("--max-tokens-per-gpu", type=int, default=8192)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--skip-base-eval", action="store_true")
    args = parser.parse_args()

    for model_key in args.models:
        train_and_eval_model(model_key, args)


if __name__ == "__main__":
    main()
