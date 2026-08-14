"""Small mini-swe protocol adapter shared by framework integrations."""

from __future__ import annotations

from typing import Any

BASH_TOOL = {
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

SYSTEM_TEMPLATE = """\
You are a software engineer with access to one bash tool. Think briefly, then
make exactly one bash tool call per response. Inspect the repository, implement
the requested source-code fix, and verify it without modifying tests.
"""

INSTANCE_TEMPLATE = """\
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

OBSERVATION_TEMPLATE = """\
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>{{ output.output }}</output>
{%- else -%}
<output_head>{{ output.output[:5000] }}</output_head>
<elided_chars>{{ output.output | length - 10000 }}</elided_chars>
<output_tail>{{ output.output[-5000:] }}</output_tail>
{%- endif %}
"""

FORMAT_ERROR_TEMPLATE = """\
Format error: {{ error }}
Respond with brief reasoning and exactly one bash tool call.
"""

SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class MiniSweEnvironmentAdapter:
    """Duck-typed mini-swe environment over a Training Gym environment."""

    config = None

    def __init__(self, environment: Any) -> None:
        self.environment = environment

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        return self.environment.execute_bash(command, cwd=cwd, timeout=timeout)

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict:
        returncode, output = self.exec(
            action.get("command", ""),
            cwd=cwd or self.cwd,
            timeout=timeout or self.exec_timeout,
        )
        lines = output.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_SENTINEL and returncode == 0:
            from minisweagent.exceptions import Submitted

            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": submission,
                    },
                }
            )
        return {
            "output": output,
            "returncode": returncode,
            "exception_info": "",
        }

    def get_template_vars(self, **kwargs) -> dict:
        return self.environment.get_template_vars()

    def serialize(self) -> dict:
        return {}

    def terminate(self) -> None:
        self.environment.close()

    @property
    def cwd(self) -> str:
        return self.environment.workdir

    @property
    def exec_timeout(self) -> int:
        return self.environment.config.exec_timeout

    @property
    def boot_time(self) -> float:
        return self.environment.boot_time

    @property
    def exec_time(self) -> float:
        return self.environment.exec_time

    @property
    def exec_timeouts(self) -> int:
        return self.environment.exec_timeouts

    @property
    def deadline(self) -> float | None:
        return self.environment.deadline

    @deadline.setter
    def deadline(self, value: float | None) -> None:
        self.environment.deadline = value
