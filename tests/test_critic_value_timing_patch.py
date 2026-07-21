from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_critic_value_timing,
)


def test_critic_value_timing_patch_is_idempotent(tmp_path):
    actor = tmp_path / "actor.py"
    actor.write_text(
        "import torch.distributed as dist\n"
        "from slime.utils.types import RolloutBatch\n\n"
        "class Actor:\n"
        "    def train_critic(self, rollout_id, rollout_data):\n"
        "        rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))\n"
    )

    patch_critic_value_timing._patch_file(actor)
    patched = actor.read_text()
    patch_critic_value_timing._patch_file(actor)

    assert '                "value_inference",\n' in patched
    assert '                "phase_start",\n' in patched
    assert '                "phase_finish",\n' in patched
    assert '                timeline_lane="training",\n' in patched
    assert '                parent_phase="training",\n' in patched
    assert '                display_name="Critic value inference",\n' in patched
    assert patched.count("rollout_data.update(forward_only(get_values") == 1
    assert actor.read_text() == patched
