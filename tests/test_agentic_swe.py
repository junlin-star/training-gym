from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from modal_training_gym.common.environments.base import EvalVerdict
from modal_training_gym.frameworks.slime.agentic_rl import grade, metrics
from modal_training_gym.frameworks.slime.agentic_rl.model import RecordingModel
from modal_training_gym.frameworks.slime.agentic_rl.sandbox import Sandbox


def _import_with_stubs(module_name: str, monkeypatch):
    for _ in range(40):
        try:
            return importlib.import_module(module_name)
        except ImportError as error:
            missing = error.name
            if not missing or missing in sys.modules:
                raise
            stub = types.ModuleType(missing)
            stub.__path__ = []
            stub.__getattr__ = lambda _name: MagicMock()
            monkeypatch.setitem(sys.modules, missing, stub)
    return importlib.import_module(module_name)


class _Tokenizer:
    chat_template = "{{ preserve_thinking }}"

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 1, "<|endoftext|>": 2}[token]

    def apply_chat_template(self, messages, **kwargs):
        return "".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def _recording_model():
    return RecordingModel(
        _Tokenizer(),
        {"max_new_tokens": 32},
        "http://router",
        "observation",
        "session",
    )


def test_recording_model_parses_qwen_bash_tool_call():
    model = _recording_model()
    response = (
        "<tool_call><function=bash><parameter=command>"
        "pytest -q"
        "</parameter></function></tool_call>"
    )

    assert model._parse_actions(response, "stop") == [{"command": "pytest -q"}]


def test_sandbox_adapter_delegates_to_native_environment(monkeypatch):
    class Environment:
        workdir = "/repo"
        config = SimpleNamespace(exec_timeout=120)
        boot_time = 1.0
        exec_time = 2.0
        exec_timeouts = 0
        deadline = None
        closed = False

        def execute_bash(self, command, cwd=None, timeout=None):
            return 0, "ok"

        def write_file(self, path, content):
            self.file = (path, content)

        def get_template_vars(self):
            return {"cwd": self.workdir}

        def close(self):
            self.closed = True

    environment = Environment()
    monkeypatch.setattr(
        "modal_training_gym.frameworks.slime.agentic_rl.sandbox.SweEnvironment.create",
        classmethod(lambda cls, task, **kwargs: environment),
    )

    adapter = Sandbox({"instance_id": "task"})

    assert adapter.exec("pwd") == (0, "ok")
    assert adapter.get_template_vars() == {"cwd": "/repo"}
    adapter.terminate()
    assert environment.closed is True


def test_grade_adapter_uses_native_verdict(monkeypatch):
    verdict = EvalVerdict(
        passed=True,
        metadata={
            "dense_reward": 0.75,
            "passed": ["a"],
            "baseline_passed": [],
            "required": ["a"],
            "missing": [],
            "progress": 0.75,
            "pass_to_pass_fraction": 1.0,
            "output": "PASSED a",
        },
    )
    monkeypatch.setattr(grade, "grade_swe_patch", lambda *args, **kwargs: verdict)

    result = grade.grade_detailed(
        {"FAIL_TO_PASS": ["a"]},
        "diff",
    )

    assert result["reward"] == 1.0
    assert result["dense"] == 0.75


def test_agentic_metrics_report_policy_lag():
    now = 1000.0
    samples = [
        SimpleNamespace(
            metadata={"agentic": {"gen_timestamp": 900.0}},
            weight_versions=["1", "2"],
        ),
        SimpleNamespace(
            metadata={"agentic": {"gen_timestamp": 950.0}},
            weight_versions=["2"],
        ),
    ]

    result = metrics._async_metrics(samples, now)

    assert result["async/version_span/max"] == 2.0
    assert result["async/version_lag/max"] == 1.0
    assert result["async/sample_age_sec/max"] == 100.0


def test_generate_hook_imports_with_remote_slime_dependencies_stubbed(monkeypatch):
    module_name = "modal_training_gym.frameworks.slime.agentic_rl.generate"
    sys.modules.pop(module_name, None)

    module = _import_with_stubs(module_name, monkeypatch)

    assert callable(module.generate)
