from __future__ import annotations

import json
import sys
import types

import pytest

from modal_training_gym import (
    SweEnvironment,
    SweRebenchV2Config,
    SweRebenchV2Dataset,
)
from modal_training_gym.common.environments import swerebench
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ToolCall


def _task(**overrides):
    task = {
        "instance_id": "django__django-123",
        "image_name": "example/swe-image:latest",
        "repo": "django/django",
        "problem_statement": "Fix the bug.",
        "install_config": {
            "log_parser": "parse_log_pytest",
            "test_cmd": "pytest -rA tests/test_fix.py",
        },
        "test_patch": "+++ b/tests/test_fix.py\n",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_bug"],
        "PASS_TO_PASS": ["tests/test_fix.py::test_existing"],
    }
    task.update(overrides)
    return task


class _Stream:
    def __init__(self, value: bytes):
        self.value = value

    def read(self):
        return self.value


class _Process:
    def __init__(self, output: str, returncode: int = 0):
        self.stdout = _Stream(output.encode())
        self.stderr = _Stream(b"")
        self.returncode = returncode

    def wait(self):
        return self.returncode


class _Sandbox:
    def __init__(self, output: str = ""):
        self.output = output
        self.commands = []
        self.terminated = False

    def exec(self, *args, **kwargs):
        self.commands.append((args, kwargs))
        return _Process(self.output)

    def terminate(self):
        self.terminated = True


def test_normalize_swe_task_and_workdir():
    normalized = swerebench.normalize_swe_task(_task())

    assert normalized["task_type"] == "swerebench"
    assert normalized["workdir"] == "/django"
    assert swerebench.repo_workdir("org/repo") == "/repo"


def test_normalize_swe_task_rejects_unsupported_parser():
    task = _task(install_config={"log_parser": "parse_log_js", "test_cmd": "npm test"})

    with pytest.raises(TrainingGymConfigError, match="Unsupported SWE log parser"):
        swerebench.normalize_swe_task(task)


def test_pytest_and_patch_parsers():
    output = "\n".join(
        [
            "PASSED tests/test_fix.py::test_bug",
            "FAILED tests/test_fix.py::test_existing - AssertionError",
        ]
    )
    patch = "\n".join(
        [
            "diff --git a/tests/a.py b/tests/a.py",
            "--- a/tests/a.py",
            "+++ b/tests/a.py",
            "diff --git a/tests/b.py b/tests/b.py",
            "--- /dev/null",
            "+++ b/tests/b.py",
        ]
    )

    assert swerebench.passed_pytest_tests(output) == {"tests/test_fix.py::test_bug"}
    assert swerebench.test_files_from_patch(patch) == [
        "tests/a.py",
        "tests/b.py",
    ]


def test_swe_environment_executes_bash_tool():
    sandbox = _Sandbox(output="ok")
    environment = SweEnvironment(
        sandbox,
        task=swerebench.normalize_swe_task(_task()),
        config=swerebench.SweEnvironmentConfig(),
    )

    result = environment.step(ToolCall(name="bash", arguments={"command": "pwd"}))

    assert result.observation.text == "ok"
    assert result.observation.is_error is False
    assert sandbox.commands[0][0][-1] == "cd /django && pwd"


def test_grade_swe_patch_uses_fresh_environment(monkeypatch):
    outputs = iter(
        [
            "FAILED tests/test_fix.py::test_bug\n"
            "PASSED tests/test_fix.py::test_existing\n",
            "PASSED tests/test_fix.py::test_bug\n"
            "PASSED tests/test_fix.py::test_existing\n",
        ]
    )

    class _Grader:
        def __init__(self):
            self.files = {}
            self.closed = False

        def write_file(self, path, content):
            self.files[path] = content

        def execute_bash(self, command, timeout=None):
            return 0, next(outputs)

        def close(self):
            self.closed = True

    grader = _Grader()
    monkeypatch.setattr(
        swerebench.SweEnvironment,
        "create",
        classmethod(lambda cls, task, **kwargs: grader),
    )

    verdict = swerebench.grade_swe_patch(_task(), "diff --git a/a.py b/a.py")

    assert verdict.passed is True
    assert verdict.metadata["dense_reward"] == 1.0
    assert grader.closed is True
    assert set(grader.files) == {"/tmp/model.patch", "/tmp/test.patch"}


def test_filtered_dataset_materializes_native_rows(monkeypatch, tmp_path):
    raw = _task(language="python")
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *args, **kwargs: iter([raw])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    config = SweRebenchV2Config(
        n_tasks=1,
        prefilter_repo=None,
    )
    dataset = SweRebenchV2Dataset(config=config)
    output = tmp_path / "train.jsonl"
    dataset.prepare(str(output))

    row = json.loads(output.read_text())
    assert row["prompt"] == "Fix the bug."
    assert row["label"] == "django__django-123"
    assert row["metadata"]["task_type"] == "swerebench"
    assert dataset.input_key == "prompt"
    assert dataset.apply_chat_template is False


def test_top_level_swe_exports():
    assert SweEnvironment is swerebench.SweEnvironment
    assert SweRebenchV2Dataset is swerebench.SweRebenchV2Dataset
