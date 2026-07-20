from types import SimpleNamespace

from modal_training_gym.common import deployment as deployment_module
from modal_training_gym.common.deployment import ModelDeployment
from modal_training_gym.common.environments import bfcl
from modal_training_gym.common.environments.base import (
    EvalVerdict,
    Observation,
    StepResult,
    ToolCall,
)


class _FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[ToolCall] = []

    def step(self, action: ToolCall) -> StepResult:
        self.actions.append(action)
        return StepResult(observation=Observation(text=f"result:{action.name}"))

    def evaluate(self) -> EvalVerdict:
        return EvalVerdict(passed=True)


def test_run_bfcl_episode_executes_calls_and_appends_observations(monkeypatch) -> None:
    environment = _FakeEnvironment()
    monkeypatch.setattr(bfcl, "build_env", lambda label, start_step: environment)
    monkeypatch.setattr(
        bfcl,
        "build_prefix_messages",
        lambda label, start_step: [{"role": "user", "content": "start"}],
    )
    monkeypatch.setattr(
        bfcl,
        "tool_schemas_to_openai",
        lambda schemas: [{"type": "function", "function": {"name": "lookup"}}],
    )

    responses = iter(
        [
            {
                "content": "",
                "actions": [ToolCall(name="lookup", arguments={"key": "value"})],
            },
            {"content": "done", "actions": []},
        ]
    )
    generated_messages: list[list[dict]] = []

    def generate(messages: list[dict], tools: list[dict]) -> dict:
        generated_messages.append(list(messages))
        assert tools[0]["function"]["name"] == "lookup"
        return next(responses)

    result = bfcl.run_bfcl_episode(
        {},
        start_step=2,
        generate=generate,
        parse_response=lambda message: (
            message["content"],
            message["actions"],
        ),
        max_turns=3,
    )

    assert result.verdict.passed
    assert result.exit_reason == "no_further_calls"
    assert result.final_response == "done"
    assert result.first_call == {"name": "lookup", "arguments": {"key": "value"}}
    assert result.execution_successes == [True]
    assert environment.actions[0].name == "lookup"
    assert generated_messages[1][-1] == {
        "role": "tool",
        "tool_call_id": "call_t0_0",
        "content": "result:lookup",
    }


def test_model_deployment_chat_preserves_structured_message(monkeypatch) -> None:
    structured_message = {
        "content": "",
        "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
    }

    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": structured_message}]}

    request_body = {}

    def post(url, *, json, timeout, headers):
        request_body.update(json)
        return _Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(deployment_module, "_modal_proxy_auth_headers", lambda: {})
    deployment = ModelDeployment.model_construct(
        deployment_id="test",
        deployment_config=SimpleNamespace(served_model_name="test-model"),
        url="https://example.test",
    )

    message = deployment.chat(
        [{"role": "user", "content": "look it up"}],
        ensure_ready=False,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )

    assert message == structured_message
    assert request_body["model"] == "test-model"
    assert request_body["tools"][0]["function"]["name"] == "lookup"
