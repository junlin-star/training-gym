"""SWE-Rebench V2 data and fresh-sandbox execution primitives.

The common layer owns benchmark data, task normalization, Modal Sandbox
lifecycle, and held-out grading. The slime-specific mini-swe adapter lives in
``modal_training_gym.frameworks.slime.agentic_rl``.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.environments.base import (
    EvalVerdict,
    Observation,
    SandboxEnvironment,
    StepResult,
    ToolCall,
)
from modal_training_gym.common.errors import TrainingGymConfigError

SUPPORTED_LOG_PARSERS = frozenset(
    {
        "parse_log_pytest",
        "parse_log_pytest_options",
        "parse_log_pytest_v2",
    }
)
_APPLY = "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn"
_EXEC_GRACE_SECONDS = 30

SWE_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute one bash command in the task repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

SWE_SYSTEM_TEMPLATE = """\
You are a software engineer with access to one bash tool. Think briefly, then
make exactly one bash tool call per response. Inspect the repository, implement
the requested source-code fix, and verify it without modifying tests.
"""

SWE_INSTANCE_TEMPLATE = """\
Solve the following software-engineering task in {{cwd}}:

<task>
{{task}}
</task>

Use bash to inspect and edit the repository. Do not modify tests or commit.
When the fix is ready:
1. Write only the intended source changes to patch.txt with `git diff`.
2. Inspect patch.txt.
3. In a separate final tool call, run exactly:
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`
"""

SWE_OBSERVATION_TEMPLATE = """\
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>{{ output.output }}</output>
{%- else -%}
<output_head>{{ output.output[:5000] }}</output_head>
<elided_chars>{{ output.output | length - 10000 }}</elided_chars>
<output_tail>{{ output.output[-5000:] }}</output_tail>
{%- endif %}
"""

SWE_SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def extract_swe_submission(output: str, returncode: int) -> str | None:
    """Return the submitted patch when output follows the SWE sentinel contract."""

    lines = output.lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == SWE_SUBMIT_SENTINEL and returncode == 0:
        return "".join(lines[1:])
    return None


@dataclass(frozen=True)
class SweRebenchV2Config:
    hf_repo: str = "nebius/SWE-rebench-V2"
    hf_revision: str = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
    hf_split: str = "train"
    prefilter_repo: str | None = "junlin-modal/swe-rebench-v2"
    prefilter_revision: str | None = "f2cf9141b7fff2febb748d4859b9b5d0d2aacda1"
    prefilter_path: str = "swe_rebench_v2/prefilter_ids.json"
    language: str = "python"
    n_tasks: int | None = None
    scan_limit: int | None = None


@dataclass(frozen=True)
class SweEnvironmentConfig:
    sandbox_app: str = "training-gym-swe-rebench-sandboxes"
    lifetime: int = 2100
    exec_timeout: int = 120
    grade_timeout: int = 1800
    boot_retries: int = 2
    cpu: float | None = None
    memory_mb: int | None = None


DEFAULT_SWE_ENVIRONMENT_CONFIG = SweEnvironmentConfig()


def repo_workdir(repo: str) -> str:
    name = (repo or "").rstrip("/").split("/")[-1]
    if not name:
        raise TrainingGymConfigError(f"Cannot derive a workdir from repo {repo!r}")
    return f"/{name}"


def test_files_from_patch(test_patch: str) -> list[str]:
    return [
        line[6:].strip()
        for line in (test_patch or "").splitlines()
        if line.startswith("+++ b/")
    ]


def passed_pytest_tests(log: str) -> set[str]:
    passed = set()
    for line in log.splitlines():
        if line.startswith("PASSED"):
            parts = line.split()
            if len(parts) >= 2:
                passed.add(parts[1])
    return passed


def normalize_swe_task(raw: dict[str, Any]) -> dict[str, Any]:
    install_config = raw.get("install_config") or {}
    parser = install_config.get("log_parser")
    if parser not in SUPPORTED_LOG_PARSERS:
        raise TrainingGymConfigError(f"Unsupported SWE log parser {parser!r}")
    if not install_config.get("test_cmd"):
        raise TrainingGymConfigError("SWE task is missing install_config.test_cmd")
    if not raw.get("image_name"):
        raise TrainingGymConfigError("SWE task is missing image_name")
    if not raw.get("repo"):
        raise TrainingGymConfigError("SWE task is missing repo")
    if not raw.get("FAIL_TO_PASS"):
        raise TrainingGymConfigError("SWE task has no FAIL_TO_PASS tests")
    if not raw.get("test_patch"):
        raise TrainingGymConfigError("SWE task has no held-out test patch")

    return {
        "task_type": "swerebench",
        "instance_id": raw["instance_id"],
        "image_name": raw["image_name"],
        "repo": raw["repo"],
        "workdir": raw.get("workdir") or repo_workdir(raw["repo"]),
        "problem_statement": raw["problem_statement"],
        "install_config": install_config,
        "test_patch": raw["test_patch"],
        "FAIL_TO_PASS": list(raw["FAIL_TO_PASS"]),
        "PASS_TO_PASS": list(raw.get("PASS_TO_PASS") or []),
    }


def parse_swe_task_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return normalize_swe_task(metadata)
    label = row.get("label")
    if isinstance(label, str):
        try:
            decoded = json.loads(label)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            return normalize_swe_task(decoded)
    raise TrainingGymConfigError("SWE row has no task metadata")


class SweRebenchV2Dataset(DatasetConfig):
    """A filtered, directly runnable Python subset of SWE-Rebench V2."""

    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = False
    writes_eval_paths = False
    always_prepare = True

    def __init__(
        self,
        *,
        config: SweRebenchV2Config | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or SweRebenchV2Config()
        super().__init__(
            dataset_id=kwargs.pop(
                "dataset_id",
                f"swe-rebench-v2-agentic-{self.config.n_tasks or 'all'}",
            ),
            **kwargs,
        )

    def _prefilter_ids(self) -> set[str] | None:
        if not self.config.prefilter_repo:
            return None
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            self.config.prefilter_repo,
            self.config.prefilter_path,
            repo_type="dataset",
            revision=self.config.prefilter_revision,
        )
        return set(json.loads(Path(path).read_text())["instance_ids"])

    def _load_rows(self) -> list[dict[str, Any]]:
        from datasets import load_dataset

        prefilter_ids = self._prefilter_ids()
        rows: list[dict[str, Any]] = []
        source = load_dataset(
            self.config.hf_repo,
            split=self.config.hf_split,
            revision=self.config.hf_revision,
            streaming=True,
        )
        for scanned, raw in enumerate(source, start=1):
            if self.config.scan_limit and scanned > self.config.scan_limit:
                break
            if prefilter_ids is not None and raw["instance_id"] not in prefilter_ids:
                continue
            if (raw.get("language") or "").lower() != self.config.language:
                continue
            try:
                task = normalize_swe_task(dict(raw))
            except TrainingGymConfigError:
                continue
            rows.append(
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": task["problem_statement"],
                        }
                    ],
                    "label": task["instance_id"],
                    "metadata": task,
                }
            )
            if self.config.n_tasks and len(rows) >= self.config.n_tasks:
                break
        if self.config.n_tasks and len(rows) < self.config.n_tasks:
            raise TrainingGymConfigError(
                f"Only found {len(rows)}/{self.config.n_tasks} compatible SWE tasks"
            )
        return rows

    def load(
        self, split: Literal["all", "train", "eval"] = "all"
    ) -> list[dict[str, Any]]:
        return self._load_rows()

    def prepare(
        self,
        path: str,
        eval_paths: dict[str, str] | None = None,
    ) -> None:
        rows = self._load_rows()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


def _run_with_timeout(fn, timeout_seconds: float, operation: str):
    result: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            result["value"] = fn()
        except BaseException as error:
            result["error"] = error
        finally:
            done.set()

    threading.Thread(target=run, name="swe-modal-rpc", daemon=True).start()
    if not done.wait(timeout_seconds):
        raise TimeoutError(f"{operation} exceeded {timeout_seconds}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


class SweEnvironment(SandboxEnvironment):
    """One mutable SWE repository in a Modal Sandbox."""

    def __init__(
        self,
        sandbox: Any,
        *,
        task: dict[str, Any],
        config: SweEnvironmentConfig,
        boot_time: float = 0.0,
    ) -> None:
        super().__init__(sandbox)
        self.task = task
        self.config = config
        self.workdir = task["workdir"]
        self.boot_time = boot_time
        self.exec_time = 0.0
        self.exec_timeouts = 0
        self.deadline: float | None = None

    @classmethod
    def create(
        cls,
        task: dict[str, Any],
        *,
        config: SweEnvironmentConfig = DEFAULT_SWE_ENVIRONMENT_CONFIG,
        lifetime: int | None = None,
    ) -> "SweEnvironment":
        import modal

        normalized = normalize_swe_task(task)
        cpu = config.cpu
        memory_mb = config.memory_mb
        if cpu is None and os.environ.get("SLIME_AGENT_SANDBOX_CPU"):
            cpu = float(os.environ["SLIME_AGENT_SANDBOX_CPU"])
        if memory_mb is None and os.environ.get("SLIME_AGENT_SANDBOX_MEMORY_MB"):
            memory_mb = int(os.environ["SLIME_AGENT_SANDBOX_MEMORY_MB"])

        app = modal.App.lookup(config.sandbox_app, create_if_missing=True)
        create = getattr(modal.Sandbox, "_experimental_create", modal.Sandbox.create)
        kwargs: dict[str, Any] = {
            "image": modal.Image.from_registry(normalized["image_name"]),
            "app": app,
            "timeout": lifetime or config.lifetime,
        }
        if cpu is not None:
            kwargs["cpu"] = cpu
        if memory_mb is not None:
            kwargs["memory"] = memory_mb

        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(config.boot_retries + 1):
            try:
                sandbox = create("sleep", "infinity", **kwargs)
                return cls(
                    sandbox,
                    task=normalized,
                    config=config,
                    boot_time=time.perf_counter() - started,
                )
            except Exception as error:
                last_error = error
                if attempt < config.boot_retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"SWE sandbox boot after {config.boot_retries + 1} attempts: {last_error}"
        )

    def execute_bash(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        started = time.perf_counter()
        budget = timeout or self.config.exec_timeout
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                return 124, "command not run: agent time budget exhausted"
            budget = min(budget, max(1, int(remaining)))
        command = command.replace("\x00", "")
        workdir = shlex.quote(cwd or self.workdir)

        def execute() -> tuple[int, str]:
            process = self.sandbox.exec(
                "bash",
                "-lc",
                f"cd {workdir} && {command}",
                timeout=budget,
                text=False,
            )
            output = (process.stdout.read() + process.stderr.read()).decode(
                "utf-8",
                errors="replace",
            )
            return process.wait(), output

        try:
            return _run_with_timeout(
                execute,
                budget + _EXEC_GRACE_SECONDS,
                f"exec({command[:80]})",
            )
        except TimeoutError:
            self.exec_timeouts += 1
            return 124, f"command timed out after {budget}s (sandbox unresponsive)"
        finally:
            self.exec_time += time.perf_counter() - started

    def write_file(self, path: str, content: str) -> None:
        def write() -> None:
            process = self.sandbox.exec(
                "bash",
                "-lc",
                f"cat > {shlex.quote(path)}",
                text=False,
            )
            data = content.encode()
            for offset in range(0, len(data), 1 << 20):
                process.stdin.write(data[offset : offset + (1 << 20)])
                process.stdin.drain()
            process.stdin.write_eof()
            process.stdin.drain()
            process.wait()

        _run_with_timeout(
            write,
            self.config.exec_timeout,
            f"write_file({path})",
        )

    def step(self, action: ToolCall) -> StepResult:
        if action.name != "bash":
            return StepResult(
                observation=Observation(
                    text=f"Unsupported tool {action.name!r}",
                    is_error=True,
                )
            )
        returncode, output = self.execute_bash(str(action.arguments.get("command", "")))
        return StepResult(
            observation=Observation(
                text=output,
                is_error=returncode != 0,
                metadata={"returncode": returncode},
            )
        )

    def capture_patch(self) -> str:
        _, patch = self.execute_bash(
            "git add -A && git diff --cached HEAD",
            timeout=120,
        )
        return patch

    def evaluate(self, patch: str | None = None) -> EvalVerdict:
        return grade_swe_patch(
            self.task,
            patch if patch is not None else self.capture_patch(),
            config=self.config,
        )

    def get_template_vars(self) -> dict[str, str]:
        return {
            "system": "Linux",
            "release": "",
            "version": "",
            "machine": "x86_64",
            "cwd": self.workdir,
        }

    def serialize(self) -> dict[str, Any]:
        return {}


def grade_swe_patch(
    task: dict[str, Any],
    model_patch: str,
    *,
    config: SweEnvironmentConfig = DEFAULT_SWE_ENVIRONMENT_CONFIG,
    timeout: int | None = None,
) -> EvalVerdict:
    """Apply and grade a patch in a sandbox separate from the agent."""

    normalized = normalize_swe_task(task)
    if not model_patch.strip():
        return EvalVerdict(
            passed=False,
            detail="The agent produced an empty patch.",
            metadata={"dense_reward": 0.0, "missing": normalized["FAIL_TO_PASS"]},
        )

    timeout = timeout or config.grade_timeout
    grader = SweEnvironment.create(
        normalized,
        config=config,
        lifetime=2 * timeout + 120,
    )
    try:
        grader.write_file("/tmp/model.patch", model_patch)
        grader.write_file("/tmp/test.patch", normalized["test_patch"])
        test_cmd = normalized["install_config"]["test_cmd"]
        test_commands = test_cmd if isinstance(test_cmd, list) else [test_cmd]
        reset_test_files = [
            f"git checkout HEAD -- {shlex.quote(path)} 2>/dev/null "
            f"|| rm -f {shlex.quote(path)}"
            for path in test_files_from_patch(normalized["test_patch"])
        ]

        baseline_script = "\n".join(
            [
                "set -e",
                "git reset --hard HEAD",
                f"{_APPLY} /tmp/test.patch",
                "set +e",
                *test_commands,
            ]
        )
        _, baseline_output = grader.execute_bash(
            baseline_script,
            timeout=timeout,
        )

        patched_script = "\n".join(
            [
                "set -e",
                "git reset --hard HEAD",
                f"{_APPLY} /tmp/model.patch",
                *reset_test_files,
                f"{_APPLY} /tmp/test.patch",
                "set +e",
                *test_commands,
            ]
        )
        _, output = grader.execute_bash(patched_script, timeout=timeout)
    except Exception as error:
        return EvalVerdict(
            passed=False,
            detail=f"SWE grading harness: {type(error).__name__}: {error}",
            harness_error=True,
            metadata={"dense_reward": 0.0},
        )
    finally:
        grader.close()

    baseline_passed = passed_pytest_tests(baseline_output)
    passed = passed_pytest_tests(output)
    fail_to_pass = normalized["FAIL_TO_PASS"]
    pass_to_pass = normalized["PASS_TO_PASS"]
    required = fail_to_pass + pass_to_pass
    missing = [test for test in required if test not in passed]
    solved = bool(fail_to_pass) and not missing

    fixable = [test for test in fail_to_pass if test not in baseline_passed]
    progress = (
        len([test for test in fixable if test in passed]) / len(fixable)
        if fixable
        else (1.0 if solved else 0.0)
    )
    stable = [test for test in pass_to_pass if test in baseline_passed]
    stable_fraction = (
        len([test for test in stable if test in passed]) / len(stable)
        if stable
        else 1.0
    )
    dense_reward = round(progress * stable_fraction, 4)

    return EvalVerdict(
        passed=solved,
        detail="" if solved else f"Missing {len(missing)}/{len(required)} tests",
        metadata={
            "dense_reward": dense_reward,
            "passed": sorted(passed),
            "baseline_passed": sorted(baseline_passed),
            "required": required,
            "missing": missing,
            "progress": round(progress, 4),
            "pass_to_pass_fraction": round(stable_fraction, 4),
            "output": output,
        },
    )
