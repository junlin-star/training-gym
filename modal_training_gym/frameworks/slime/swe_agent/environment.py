"""mini-swe protocol adapter for Training Gym's SWE environment."""

from __future__ import annotations

from typing import Any

from modal_training_gym.common.environments.swerebench import (
    extract_swe_submission,
)
from modal_training_gym.common.models import ToolCall


class MiniSweEnvironmentAdapter:
    """Expose a Training Gym SWE environment through mini-swe's protocol."""

    config = None

    def __init__(self, environment: Any) -> None:
        self.environment = environment

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict:
        result = self.environment.step(ToolCall(name="bash", arguments=dict(action)))
        returncode = int(result.observation.metadata.get("returncode", 0))
        output = result.observation.text
        submission = extract_swe_submission(output, returncode)
        if submission is not None:
            from minisweagent.exceptions import Submitted

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
