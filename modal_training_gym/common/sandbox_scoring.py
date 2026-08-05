from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import modal

from modal_training_gym.common.errors import TrainingGymConfigError

if TYPE_CHECKING:
    from modal_training_gym.common.models.base import ModelConfig

_CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)

_MODAL_DEFAULT_CPU_REQUEST_CORES = 0.125
_MODAL_DEFAULT_MEMORY_REQUEST_MIB = 128


RESOURCE_POLICIES = ("reserve", "limit", "ignore")


def extract_code(text: str, model: "ModelConfig | None" = None) -> str:
    """Extract Python code from an LLM response.

    When *model* is provided, use its response parser and prefer a tool call's
    ``code`` argument. Otherwise, strip common chat artifacts and extract the
    first Python code fence.
    """
    if model is not None:
        parsed = model.parse_response(text)
        for tool_call in parsed.tool_calls:
            code = tool_call.arguments.get("code", "")
            if code:
                return code
        content = parsed.content
    else:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if "<|im_start|>assistant" in normalized:
            normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
        if "</think>" in normalized:
            normalized = normalized.split("</think>", 1)[-1]
        normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
        content = normalized

    if match := _CODE_FENCE_RE.search(content):
        return match.group(1).strip()
    return content


def _sandbox_resource(
    value: float, policy: str, default_request: float
) -> float | tuple[float, float] | None:
    if policy == "reserve":
        return value
    if policy == "limit":
        return (min(default_request, value), value)
    if policy == "ignore":
        return None
    raise TrainingGymConfigError(
        f"invalid resource policy {policy!r}; expected one of {RESOURCE_POLICIES}"
    )


def score_in_sandbox(
    code: str,
    *,
    test_cases: list[dict[str, str]],
    timeout_sec: int = 60,
    sandbox_cpu: float = 1.0,
    sandbox_memory: int = 1024,
    python_version: str = "3.11",
    cpu_policy: str = "limit",
    memory_policy: str = "limit",
) -> tuple[float, dict[str, Any]]:
    """Run *code* against *test_cases* within one total sandbox timeout."""
    if not test_cases:
        return 0.0, {"error": "no test cases"}

    runner = (
        "import json, subprocess, sys\n"
        "cases = json.load(sys.stdin)\n"
        "for case in cases:\n"
        "    try:\n"
        "        proc = subprocess.run(\n"
        '            [sys.executable, "-c", case["code"]],\n'
        '            input=case["input"], text=True, capture_output=True,\n'
        '            timeout=case["timeout"],\n'
        "        )\n"
        "        stdout, stderr = proc.stdout, proc.stderr\n"
        "        ok = proc.returncode == 0\n"
        "    except Exception as exc:\n"
        '        stdout, stderr, ok = "", str(exc), False\n'
        '    result = {"ran": ok, "stdout": stdout, "stderr": stderr}\n'
        "    print(json.dumps(result), flush=True)\n"
    )

    case_timeout = max(1, timeout_sec // len(test_cases))
    sandbox_timeout = timeout_sec
    cases_payload = json.dumps(
        [
            {
                "code": code,
                "input": tc.get("input", ""),
                "timeout": case_timeout,
            }
            for tc in test_cases
        ]
    ).encode()

    app = modal.App.lookup("training-gym-sandbox-rm", create_if_missing=True)
    image = modal.Image.debian_slim(python_version=python_version)

    resource_kwargs: dict[str, Any] = {}
    cpu_arg = _sandbox_resource(
        sandbox_cpu, cpu_policy, _MODAL_DEFAULT_CPU_REQUEST_CORES
    )
    if cpu_arg is not None:
        resource_kwargs["cpu"] = cpu_arg
    memory_arg = _sandbox_resource(
        sandbox_memory, memory_policy, _MODAL_DEFAULT_MEMORY_REQUEST_MIB
    )
    if memory_arg is not None:
        resource_kwargs["memory"] = memory_arg

    sb = modal.Sandbox.create(
        "python",
        "-c",
        runner,
        image=image,
        timeout=sandbox_timeout,
        app=app,
        block_network=True,
        **resource_kwargs,
    )
    sb.stdin.write(cases_payload)
    sb.stdin.write_eof()
    sb.stdin.drain()
    timed_out = False
    try:
        sb.wait()
    except modal.exception.SandboxTimeoutError:
        timed_out = True

    stdout = sb.stdout.read()
    stderr = sb.stderr.read()

    metadata: dict[str, Any] = {"stderr": stderr}
    if timed_out:
        metadata["error"] = f"sandbox timed out after {sandbox_timeout}s"
    try:
        results = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        if len(results) > len(test_cases) or (
            not timed_out and len(results) != len(test_cases)
        ):
            raise ValueError
        for result, test_case in zip(results, test_cases):
            result["passed"] = (
                result.pop("ran", False)
                and result["stdout"].strip()
                == test_case.get("expected_output", "").strip()
            )
        passed = sum(1 for result in results if result["passed"])
        metadata["per_case"] = results
        return passed / len(test_cases), metadata
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError):
        metadata["raw_stdout"] = stdout
        return 0.0, metadata
