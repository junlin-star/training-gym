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


def test_pinned_image_timing_patch_command_is_mode_independent(monkeypatch):
    auto_commands = _build_commands(monkeypatch, "auto")
    off_commands = _build_commands(monkeypatch, "off")

    assert auto_commands == off_commands
    assert auto_commands[-1].startswith("echo ")
    assert "TRAINING_GYM_SUBSTEP_TIMING=" not in auto_commands[-1]
    assert "TG_BEST_EFFORT_ENTRYPOINTS=1" not in auto_commands[-1]
