from __future__ import annotations

from modal_training_gym.frameworks.slime import launcher


def _build_commands(monkeypatch, mode):
    commands = []

    class FakeImage:
        @classmethod
        def from_registry(cls, image):
            return cls()

        def entrypoint(self, command):
            return self

        def run_commands(self, *values):
            commands.extend(values)
            return self

    monkeypatch.setattr(launcher, "Image", FakeImage)
    launcher._build_slime_base_image(mode)
    return commands


def test_pinned_image_timing_patch_is_strict_in_require(monkeypatch):
    commands = _build_commands(monkeypatch, "require")

    assert commands[-1].startswith("echo ")
    assert "TRAINING_GYM_SUBSTEP_TIMING=require" in commands[-1]
    assert "TG_BEST_EFFORT_ENTRYPOINTS=1" not in commands[-1]
    assert all(
        "TG_BEST_EFFORT_ENTRYPOINTS=1" not in command for command in commands[:-1]
    )


def test_pinned_image_timing_patch_is_best_effort_otherwise(monkeypatch):
    for mode in ("auto", "off"):
        commands = _build_commands(monkeypatch, mode)
        assert commands[-1].startswith("echo ")
        assert f"TRAINING_GYM_SUBSTEP_TIMING={mode}" in commands[-1]
        assert "TG_BEST_EFFORT_ENTRYPOINTS=1" in commands[-1]
        assert all(
            "TG_BEST_EFFORT_ENTRYPOINTS=1" not in command for command in commands[:-1]
        )
