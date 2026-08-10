"""Contracts for deterministic Training Gym runtime image dependencies."""

from __future__ import annotations

from modal_training_gym.frameworks.slime.launcher import (
    READABLE_ID_INSTALL_COMMAND,
    READABLE_ID_PACKAGES,
    _add_training_gym_runtime,
)


def test_runtime_source_and_readable_id_dependencies_are_pinned() -> None:
    class ImageProbe:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        def add_local_python_source(
            self,
            *modules: object,
            **kwargs: object,
        ) -> ImageProbe:
            self.calls.append(("add_local_python_source", modules, kwargs))
            return self

        def run_commands(self, *commands: object) -> ImageProbe:
            self.calls.append(("run_commands", commands, {}))
            return self

        def pip_install(self, *packages: object) -> ImageProbe:
            raise AssertionError(
                f"bare-python pip helper must not be used: {packages!r}"
            )

        def uv_pip_install(self, *packages: object) -> ImageProbe:
            raise AssertionError(f"mutable uv helper must not be used: {packages!r}")

    assert READABLE_ID_PACKAGES == (
        "randomname==0.2.1",
        "fire==0.7.1",
        "termcolor==3.3.0",
    )
    assert READABLE_ID_INSTALL_COMMAND == (
        "python3 -m pip install randomname==0.2.1 fire==0.7.1 termcolor==3.3.0"
    )
    probe = ImageProbe()
    assert _add_training_gym_runtime(probe) is probe
    assert probe.calls == [
        ("add_local_python_source", ("modal_training_gym",), {"copy": True}),
        ("run_commands", (READABLE_ID_INSTALL_COMMAND,), {}),
    ]
