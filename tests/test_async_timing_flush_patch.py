from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_async_timing_flush,
)


def test_async_timing_flush_patch_is_idempotent(tmp_path):
    actor = tmp_path / "actor.py"
    actor.write_text(
        "    def train(self):\n"
        "        result = None\n"
        "        return result\n\n"
        "    def train_critic(self):\n"
        "        pass\n"
    )

    patch_async_timing_flush._patch_file(actor)
    patched = actor.read_text()
    patch_async_timing_flush._patch_file(actor)

    assert "        flush_async_timing_events()\n" in patched
    assert actor.read_text() == patched
