"""mini-swe environment adapter over Training Gym's SWE environment."""

from minisweagent.exceptions import Submitted

from modal_training_gym.common.environments.swerebench import (
    SweEnvironment,
    SweEnvironmentConfig,
)

from .prompts import SUBMIT_SENTINEL


class Sandbox:
    config = None  # mini-swe Environment protocol

    def __init__(
        self,
        task: dict,
        *,
        lifetime: int = 1800,
        exec_timeout: int = 120,
        app_name: str = "training-gym-swe-rebench-sandboxes",
        cpu: float | None = None,
        memory_mb: int | None = None,
        boot_retries: int = 2,
    ) -> None:
        config = SweEnvironmentConfig(
            sandbox_app=app_name,
            lifetime=lifetime,
            exec_timeout=exec_timeout,
            boot_retries=boot_retries,
            cpu=cpu,
            memory_mb=memory_mb,
        )
        self.environment = SweEnvironment.create(
            task,
            config=config,
            lifetime=lifetime,
        )

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        return self.environment.execute_bash(command, cwd=cwd, timeout=timeout)

    def write_file(self, path: str, content: str) -> None:
        self.environment.write_file(path, content)

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
