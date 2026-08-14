"""Small mini-swe protocol adapter shared by framework integrations."""

from __future__ import annotations

from typing import Any

from modal_training_gym.common.models import ToolCall


class MiniSweEnvironmentAdapter:
    """Duck-typed mini-swe environment over a Training Gym environment."""

    config = None

    def __init__(self, environment: Any, *, tool_name: str = "bash") -> None:
        self.environment = environment
        self.tool_name = tool_name

    def _step(self, action: dict):
        return self.environment.step(
            ToolCall(name=self.tool_name, arguments=dict(action))
        )

    @staticmethod
    def _format_result(result) -> dict:
        returncode = int(result.observation.metadata.get("returncode", 0))
        return {
            "output": result.observation.text,
            "returncode": returncode,
            "exception_info": "",
        }

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict:
        return self._format_result(self._step(action))

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
