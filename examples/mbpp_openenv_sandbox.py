"""MBPP OpenEnv prototype backed by Modal Sandboxes.

Run a 10-task smoke test with reference MBPP solutions:

    uv run --with openenv python examples/mbpp_openenv_sandbox.py --limit 10

The environment is intentionally one-shot: reset selects an MBPP task, and step
accepts one model completion, extracts Python code, runs MBPP asserts in a fresh
Modal Sandbox, and returns a correctness-first reward.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from huggingface_hub import hf_hub_download
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action,
    EnvironmentMetadata,
    Observation,
    State,
)
from pydantic import Field

from modal_training_gym import extract_code

MBPP_REPO = "Muennighoff/mbpp"
DEFAULT_APP_NAME = "training-gym-mbpp-openenv"
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the task by writing the "
    "shortest correct Python implementation you can. Return only Python code "
    "inside one ```python code fence."
)


@dataclass(frozen=True)
class MBPPTask:
    task_id: int
    text: str
    reference_code: str
    test_setup_code: str
    test_list: tuple[str, ...]
    challenge_test_list: tuple[str, ...]

    @property
    def prompt(self) -> str:
        visible_tests = "\n".join(f"- `{test}`" for test in self.test_list)
        return (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"Task {self.task_id}:\n{self.text}\n\n"
            "Public tests that your code must pass:\n"
            f"{visible_tests}"
        )


class MBPPSubmitCode(Action):
    completion: str = Field(description="Raw model completion containing Python code")


class MBPPCodingObservation(Observation):
    task_id: int | None = None
    prompt: str = ""
    extracted_code: str = ""
    passed: int = 0
    total: int = 0
    all_tests_passed: bool = False
    completion_chars: int = 0
    reference_chars: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str = ""


def _str(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(_str(item) for item in value)
    return ()


def _task_from_sanitized(row: dict[str, Any]) -> MBPPTask:
    """Parse a row from data/sanitized-mbpp.json (prompt, test_imports)."""
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
    """Parse a row from data/mbpp.jsonl (text, test_setup_code, challenge_test_list)."""
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


def correctness_first_brevity_reward(
    *,
    passed: int,
    total: int,
    completion_chars: int,
    reference_chars: int,
    brevity_weight: float = 0.1,
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


def run_mbpp_asserts_in_sandbox(
    *,
    code: str,
    task: MBPPTask,
    app_name: str = DEFAULT_APP_NAME,
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
        1
        for result in per_test
        if isinstance(result, dict) and result.get("passed") is True
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


class MBPPCodingEnv(Environment[MBPPSubmitCode, MBPPCodingObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        tasks: list[MBPPTask],
        *,
        app_name: str = DEFAULT_APP_NAME,
        sandbox_timeout_sec: int = 10,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("MBPPCodingEnv requires at least one task")
        self._tasks_by_id = {task.task_id: task for task in tasks}
        self._tasks = tasks
        self._app_name = app_name
        self._sandbox_timeout_sec = sandbox_timeout_sec
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = tasks[0]

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: object,
    ) -> MBPPCodingObservation:
        task_id = kwargs.get("task_id")
        if isinstance(task_id, int):
            self._current_task = self._tasks_by_id[task_id]
        elif seed is not None:
            self._current_task = self._tasks[seed % len(self._tasks)]
        else:
            self._current_task = self._tasks[0]

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return MBPPCodingObservation(
            task_id=self._current_task.task_id,
            prompt=self._current_task.prompt,
            total=len(self._current_task.test_list),
            reference_chars=len(self._current_task.reference_code.strip()),
            done=False,
            reward=None,
        )

    def step(
        self,
        action: MBPPSubmitCode,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> MBPPCodingObservation:
        self._state.step_count += 1
        code = extract_code(action.completion)
        result = run_mbpp_asserts_in_sandbox(
            code=code,
            task=self._current_task,
            app_name=self._app_name,
            timeout_sec=int(timeout_s or self._sandbox_timeout_sec),
        )
        passed = int(result["passed"])
        total = int(result["total"])
        reward, reward_parts = correctness_first_brevity_reward(
            passed=passed,
            total=total,
            completion_chars=len(code.strip()),
            reference_chars=len(self._current_task.reference_code.strip()),
        )
        return MBPPCodingObservation(
            task_id=self._current_task.task_id,
            prompt=self._current_task.prompt,
            extracted_code=code,
            passed=passed,
            total=total,
            all_tests_passed=passed == total and total > 0,
            completion_chars=len(code.strip()),
            reference_chars=len(self._current_task.reference_code.strip()),
            stdout=_str(result["stdout"]),
            stderr=_str(result["stderr"]),
            error=_str(result["error"]),
            done=True,
            reward=reward,
            metadata={"reward_parts": reward_parts, "per_test": result["per_test"]},
        )

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="MBPPCodingEnv",
            description=(
                "One-shot MBPP coding environment with Modal Sandbox execution "
                "and correctness-first brevity rewards."
            ),
            version="0.1.0",
            documentation_url="https://huggingface.co/datasets/Muennighoff/mbpp",
        )


def score_mbpp_completion(
    *,
    task: MBPPTask,
    completion: str,
    app_name: str = DEFAULT_APP_NAME,
    timeout_sec: int = 10,
) -> tuple[float, dict[str, object]]:
    code = extract_code(completion)
    result = run_mbpp_asserts_in_sandbox(
        code=code,
        task=task,
        app_name=app_name,
        timeout_sec=timeout_sec,
    )
    passed = int(result["passed"])
    total = int(result["total"])
    reward, reward_parts = correctness_first_brevity_reward(
        passed=passed,
        total=total,
        completion_chars=len(code.strip()),
        reference_chars=len(task.reference_code.strip()),
    )
    result["reward_parts"] = reward_parts
    result["extracted_code"] = code
    return reward, result


def run_reference_smoke(limit: int, subset: str, app_name: str) -> None:
    tasks = load_mbpp_tasks(limit=limit, subset=subset)
    env = MBPPCodingEnv(tasks, app_name=app_name)
    all_passed = 0
    for task in tasks:
        env.reset(task_id=task.task_id)
        observation = env.step(MBPPSubmitCode(completion=task.reference_code))
        if observation.all_tests_passed:
            all_passed += 1
        print(
            f"task={task.task_id} "
            f"passed={observation.passed}/{observation.total} "
            f"reward={observation.reward:.4f} "
            f"chars={observation.completion_chars}"
        )
        if observation.error:
            print(f"  error: {observation.error[:200]}")

    print(f"\nreference smoke: {all_passed}/{len(tasks)} tasks passed all public tests")


def check_reward_gate() -> None:
    short_failing, short_meta = correctness_first_brevity_reward(
        passed=2,
        total=3,
        completion_chars=1,
        reference_chars=100,
    )
    long_passing, long_meta = correctness_first_brevity_reward(
        passed=3,
        total=3,
        completion_chars=1000,
        reference_chars=100,
    )
    if not long_passing > short_failing:
        raise AssertionError(
            f"passing reward {long_passing} must beat failing reward {short_failing}"
        )
    if short_meta["brevity_bonus"] != 0.0:
        raise AssertionError("failing completions must not receive brevity reward")
    print(
        "reward gate ok: "
        f"short_failing={short_failing:.4f}, long_passing={long_passing:.4f}, "
        f"long_passing_bonus={long_meta['brevity_bonus']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--subset", default="sanitized")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument(
        "--check-reward-gate",
        action="store_true",
        help="Check gated brevity reward invariants before launching sandboxes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_reward_gate:
        check_reward_gate()
    run_reference_smoke(limit=args.limit, subset=args.subset, app_name=args.app_name)


if __name__ == "__main__":
    main()
