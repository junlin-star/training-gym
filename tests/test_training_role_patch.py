from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_training_role,
)


def test_training_role_patch_is_idempotent(tmp_path):
    actor = tmp_path / "train_actor.py"
    actor.write_text(
        "    def init(self, args, role):\n"
        "        self.args = args\n"
        "        self.role = role\n"
    )

    patch_training_role._patch_file(actor)
    patched = actor.read_text()
    patch_training_role._patch_file(actor)

    assert "        self.args.training_gym_role = role\n" in patched
    assert actor.read_text() == patched
