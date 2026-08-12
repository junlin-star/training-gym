from __future__ import annotations

from modal_training_gym.frameworks.slime import launcher


def _build_commands(monkeypatch):
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
    launcher._build_slime_base_image()
    return commands


def test_pinned_image_timing_patch_command_has_no_mode_environment(monkeypatch):
    commands = _build_commands(monkeypatch)

    assert commands[-1].startswith("echo ")
    assert "TRAINING_GYM_SUBSTEP_TIMING" not in commands[-1]
    assert "TG_BEST_EFFORT_ENTRYPOINTS" not in commands[-1]
