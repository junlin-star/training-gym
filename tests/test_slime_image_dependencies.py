"""Contracts for deterministic dependencies in the slime runtime image."""

from __future__ import annotations

from modal_training_gym.frameworks.slime.launcher import (
    READABLE_ID_PACKAGES,
    _install_readable_id_dependency,
)


def test_readable_id_dependency_is_pinned_and_uses_base_image_pip() -> None:
    class ImageProbe:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def pip_install(self, *packages: object) -> ImageProbe:
            self.calls.append(("pip_install", packages))
            return self

        def uv_pip_install(self, *packages: object) -> ImageProbe:
            raise AssertionError(f"mutable uv helper must not be used: {packages!r}")

    assert READABLE_ID_PACKAGES == (
        "randomname==0.2.1",
        "fire==0.7.1",
        "termcolor==3.3.0",
    )
    probe = ImageProbe()
    assert _install_readable_id_dependency(probe) is probe
    assert probe.calls == [("pip_install", READABLE_ID_PACKAGES)]
