from pathlib import Path

from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_async_timing_flush,
    patch_training_substep_timing,
)


def test_async_timing_flush_patch_is_idempotent(tmp_path):
    actor = tmp_path / "actor.py"
    actor.write_text(
        "    def train(self):\n"
        '        if self.role == "critic":\n'
        "            result = self.train_critic()\n"
        "        else:\n"
        "            result = None\n"
        "        return result\n\n"
        "    def train_critic(self):\n"
        "        pass\n"
    )

    patch_async_timing_flush._patch_file(actor)
    patched = actor.read_text()
    patch_async_timing_flush._patch_file(actor)

    assert "        flush_async_timing_events()\n" in patched
    assert patched.index("flush_async_timing_events()") > patched.index(
        "result = self.train_critic()"
    )
    assert patched.index("flush_async_timing_events()") < patched.index("return result")
    assert actor.read_text() == patched


def test_critic_timing_is_flushed_in_pinned_actor(tmp_path):
    actor = tmp_path / "actor.py"
    actor.write_text(
        (
            Path(__file__).parent
            / "testdata"
            / "slime_actor"
            / "megatron_actor.py.input"
        ).read_text()
    )

    patch_training_substep_timing._patch_file(actor)
    patch_async_timing_flush._patch_file(actor)
    patched = actor.read_text()

    assert patched.index("result = self.train_critic(rollout_id, rollout_data)") < (
        patched.index("flush_async_timing_events()")
    )
    compile(patched, str(actor), "exec")
