from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from modal_training_gym.common.agents import MiniSweEnvironmentAdapter
from modal_training_gym.frameworks.slime.swe_agent.model import RecordingModel


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


def test_mini_swe_adapter_delegates_to_native_environment():
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
    adapter = MiniSweEnvironmentAdapter(environment)

    assert adapter.exec("pwd") == (0, "ok")
    assert adapter.get_template_vars() == {"cwd": "/repo"}
    adapter.terminate()
    assert environment.closed is True


def test_generate_hook_imports_with_remote_slime_dependencies_stubbed(monkeypatch):
    module_name = "modal_training_gym.frameworks.slime.swe_agent.generate"
    sys.modules.pop(module_name, None)

    module = _import_with_stubs(module_name, monkeypatch)

    assert callable(module.generate)
