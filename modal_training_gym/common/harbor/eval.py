from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modal_training_gym.common.eval import EvalConfig, EvalRowResult
from modal_training_gym.common.dataset import DatasetRow

if TYPE_CHECKING:
    from modal_training_gym.common.deployment import ModelDeployment
    from modal_training_gym.common.models.base import ModelConfig



_CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, model: "ModelConfig | None" = None) -> str:
    """Extract Python code from an LLM response.

    When *model* is provided, uses ``model.parse_response`` to strip
    thinking tags and chat-template artifacts, and checks tool-call
    arguments for a ``code`` key.  Falls back to regex heuristics when
    *model* is ``None``.
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


# Modal's per-container default resource request (modal.com/docs/guide/resources).
# Used as the request "floor" for the "limit" enforcement policy so sandboxes bill by
# actual CPU-/RAM-second usage rather than a static reservation.
_MODAL_DEFAULT_CPU_REQUEST = 0.125
_MODAL_DEFAULT_MEMORY_REQUEST = 128  # MiB

#: Accepted CPU/memory enforcement policies, mirroring Harbor v0.8.0's ``--cpus`` /
#: ``--memory`` flags. Modal bills for ``max(request, actual usage)``, so reserving more
#: than a sandbox uses over-provisions and inflates cost.
RESOURCE_POLICIES = ("reserve", "limit", "ignore")


def _sandbox_resource(
    value: float, policy: str, default_request: float
) -> float | tuple[float, float] | None:
    """Translate a Harbor-style enforcement *policy* into a Modal cpu/memory kwarg.

    - ``"reserve"`` — reserve *value* outright (billed for the full reservation, even
      when idle). This is the static-reservation behavior that over-provisions on Modal.
    - ``"limit"`` — request the small Modal default and cap bursting at *value*, so the
      sandbox is billed by actual usage up to that ceiling.
    - ``"ignore"`` — no enforcement; returns ``None`` so the caller omits the kwarg and
      the sandbox bursts freely on Modal's default request, billed by actual usage.
    """
    if policy == "reserve":
        return value
    if policy == "limit":
        return (min(default_request, value), value)
    if policy == "ignore":
        return None
    raise ValueError(
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
    """Run *code* against *test_cases* in a Modal sandbox.

    Each test case is a dict with ``input`` and ``expected_output`` keys.
    The code is executed once per test case with the input piped to stdin.
    Returns ``(fraction_passed, metadata_dict)``.

    ``cpu_policy`` and ``memory_policy`` control how ``sandbox_cpu`` / ``sandbox_memory``
    are enforced on Modal (see :data:`RESOURCE_POLICIES`). The default ``"limit"`` treats
    them as burst ceilings rather than reservations, so the sandbox is billed by actual
    CPU-/RAM-second usage instead of over-provisioning a static reservation. Use
    ``"ignore"`` to let tasks burst above the configured values, or ``"reserve"`` for the
    legacy fixed-reservation behavior.
    """
    import modal

    if not test_cases:
        return 0.0, {"error": "no test cases"}

    runner = (
        "import sys, json, io, contextlib\n"
        "cases = json.loads(sys.argv[1])\n"
        "results = []\n"
        "for case in cases:\n"
        "    old_stdin = sys.stdin\n"
        '    sys.stdin = io.StringIO(case["input"])\n'
        "    buf = io.StringIO()\n"
        "    ok = False\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(buf):\n"
        '            exec(compile(case["code"], "<solution>", "exec"))\n'
        '        ok = buf.getvalue().strip() == case["expected_output"].strip()\n'
        "    except Exception as exc:\n"
        '        buf.write(f"ERROR: {exc}")\n'
        "    finally:\n"
        "        sys.stdin = old_stdin\n"
        '    results.append({"passed": ok, "stdout": buf.getvalue()})\n'
        "print(json.dumps(results))\n"
    )

    cases_payload = json.dumps(
        [
            {
                "code": code,
                "input": tc.get("input", ""),
                "expected_output": tc.get("expected_output", ""),
            }
            for tc in test_cases
        ]
    )

    app = modal.App.lookup("training-gym-sandbox-rm", create_if_missing=True)
    image = modal.Image.debian_slim(python_version=python_version)

    resource_kwargs: dict[str, Any] = {}
    cpu_arg = _sandbox_resource(sandbox_cpu, cpu_policy, _MODAL_DEFAULT_CPU_REQUEST)
    if cpu_arg is not None:
        resource_kwargs["cpu"] = cpu_arg
    memory_arg = _sandbox_resource(
        sandbox_memory, memory_policy, _MODAL_DEFAULT_MEMORY_REQUEST
    )
    if memory_arg is not None:
        resource_kwargs["memory"] = memory_arg

    sb = modal.Sandbox._experimental_create(
        "python",
        "-c",
        runner,
        cases_payload,
        image=image,
        timeout=timeout_sec,
        app=app,
        **resource_kwargs,
    )
    sb.wait()

    stdout = sb.stdout.read()
    stderr = sb.stderr.read()

    metadata: dict[str, Any] = {"stderr": stderr}
    try:
        results = json.loads(stdout)
        passed = sum(1 for r in results if r.get("passed"))
        metadata["per_case"] = results
        return passed / len(test_cases), metadata
    except (json.JSONDecodeError, TypeError):
        metadata["raw_stdout"] = stdout
        return 0.0, metadata


@dataclass
class HarborEval(EvalConfig):
    """Evaluate a deployed model on a Harbor dataset using sandbox execution.

    Automates the common pattern of generating code from a Harbor task,
    extracting it from the LLM response, running it in a Modal sandbox,
    and comparing stdout against expected test-case outputs.

    When neither ``eval_fn`` nor ``eval_response_fn`` is provided, a
    default sandbox-backed scorer is used automatically.  Pass
    ``extract_code_fn`` to override how code is pulled from the model
    response, or supply your own ``eval_fn`` to take full control.
    """

    model: "ModelConfig | None" = None
    test_cases: list[dict[str, str]] | None = None
    sandbox_timeout: int = 60
    sandbox_cpu: float = 1.0
    sandbox_memory: int = 1024
    sandbox_cpu_policy: str = "limit"
    sandbox_memory_policy: str = "limit"
    sandbox_python_version: str = "3.11"
    extract_code_fn: Callable[[str], str] | None = None

    def _resolve_test_cases(self, example: DatasetRow) -> list[dict[str, str]]:
        label = example.get("label", {})
        if isinstance(label, str):
            try:
                label = json.loads(label)
            except (json.JSONDecodeError, ValueError):
                label = {}
        if isinstance(label, dict):
            cases = label.get("test_cases")
            if isinstance(cases, list) and cases:
                return cases
        if self.test_cases is not None:
            return self.test_cases
        return []

    def _extract_code(self, text: str) -> str:
        if self.extract_code_fn is not None:
            return self.extract_code_fn(text)
        return extract_code(text, model=self.model)

    def _build_messages(self, example: DatasetRow, prompt: str) -> list[dict[str, str]]:
        messages = example.get("messages")
        if isinstance(messages, list) and messages:
            return messages
        msgs: list[dict[str, str]] = []
        sys_prompt = getattr(self.dataset, "system_prompt", "")
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _harbor_eval_fn(
        self,
        deployment: "ModelDeployment",
        example: DatasetRow,
    ) -> EvalRowResult:
        prompt = self.build_prompt(example)
        messages = self._build_messages(example, prompt)
        response = deployment.generate(
            prompt,
            ensure_ready=False,
            messages=messages,
            **self.generate_kwargs,
        )
        code = self._extract_code(response)
        test_cases = self._resolve_test_cases(example)
        score, metadata = score_in_sandbox(
            code,
            test_cases=test_cases,
            timeout_sec=self.sandbox_timeout,
            sandbox_cpu=self.sandbox_cpu,
            sandbox_memory=self.sandbox_memory,
            cpu_policy=self.sandbox_cpu_policy,
            memory_policy=self.sandbox_memory_policy,
            python_version=self.sandbox_python_version,
        )

        parsed = self.model.parse_response(response) if self.model is not None else None

        return EvalRowResult(
            score=score,
            response=response,
            prompt=prompt,
            parsed_response=parsed,
            metadata=metadata,
        )

    def __post_init__(self) -> None:
        if self.eval_fn is None and self.eval_response_fn is None:
            self.eval_fn = self._harbor_eval_fn
        super().__post_init__()
