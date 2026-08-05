from __future__ import annotations

from types import SimpleNamespace

import modal
import pytest

from modal_training_gym.common.sandbox_scoring import extract_code, score_in_sandbox


def test_extract_code_takes_fenced_block() -> None:
    text = "Here you go:\n```python\nprint('hi')\n```\nHope that helps."

    assert extract_code(text) == "print('hi')"


def test_extract_code_drops_reasoning_and_chat_markers() -> None:
    text = (
        "<|im_start|>assistant\n<think>weighing options</think>\n"
        "```python\nprint(2 + 2)\n```<|im_end|>"
    )

    assert extract_code(text) == "print(2 + 2)"


def test_extract_code_prefers_tool_call_argument() -> None:
    class _ToolCall:
        arguments = {"code": "print('from tool call')"}

    class _Parsed:
        tool_calls = [_ToolCall()]
        content = "```python\nprint('from content')\n```"

    class _Model:
        def parse_response(self, text: str) -> _Parsed:
            return _Parsed()

    assert extract_code("ignored", model=_Model()) == "print('from tool call')"


def test_score_in_sandbox_returns_pass_rate_for_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStdin:
        def write(self, _data: bytes) -> None:
            return None

        def write_eof(self) -> None:
            return None

        def drain(self) -> None:
            return None

    class _FakeSandbox:
        def __init__(self) -> None:
            self.stdin = _FakeStdin()
            self.stdout = SimpleNamespace(
                read=lambda: (
                    '{"ran": true, "stdout": "ok\\n", "stderr": ""}\n'
                    '{"ran": true, "stdout": "bad\\n", "stderr": ""}\n'
                )
            )
            self.stderr = SimpleNamespace(read=lambda: "")

        def wait(self) -> None:
            return None

    monkeypatch.setattr(
        "modal_training_gym.common.sandbox_scoring.modal.App.lookup",
        lambda *_, **__: object(),
    )
    monkeypatch.setattr(
        "modal_training_gym.common.sandbox_scoring.modal.Image.debian_slim",
        lambda **__: object(),
    )
    monkeypatch.setattr(
        "modal_training_gym.common.sandbox_scoring.modal.Sandbox.create",
        lambda *_args, **_kwargs: _FakeSandbox(),
    )

    score, metadata = score_in_sandbox(
        "print('hi')",
        test_cases=[
            {"expected_output": "ok"},
            {"expected_output": "ok"},
        ],
    )

    assert score == 0.5
    assert metadata["per_case"][0]["passed"] is True
    assert metadata["per_case"][1]["passed"] is False


def test_score_in_sandbox_returns_partial_results_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSandbox:
        stdin = SimpleNamespace(
            write=lambda _data: None,
            write_eof=lambda: None,
            drain=lambda: None,
        )
        stdout = SimpleNamespace(
            read=lambda: '{"ran": true, "stdout": "ok\\n", "stderr": ""}\n'
        )
        stderr = SimpleNamespace(read=lambda: "")

        def wait(self) -> None:
            raise modal.exception.SandboxTimeoutError

    monkeypatch.setattr(
        "modal_training_gym.common.sandbox_scoring.modal.App.lookup",
        lambda *_, **__: object(),
    )
    monkeypatch.setattr(
        "modal_training_gym.common.sandbox_scoring.modal.Image.debian_slim",
        lambda **__: object(),
    )
    monkeypatch.setattr(
        "modal_training_gym.common.sandbox_scoring.modal.Sandbox.create",
        lambda *_args, **_kwargs: _FakeSandbox(),
    )

    score, metadata = score_in_sandbox(
        "while True: pass",
        test_cases=[
            {"expected_output": "ok"},
            {"expected_output": "ok"},
        ],
        timeout_sec=3,
    )

    assert score == 0.5
    assert metadata["error"] == "sandbox timed out after 3s"
    assert metadata["per_case"] == [{"stdout": "ok\n", "stderr": "", "passed": True}]
