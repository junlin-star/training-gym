from pathlib import Path

from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_training_role,
)


TESTDATA = Path(__file__).parent / "testdata" / "slime_actor"


def test_training_role_patch_matches_pinned_slime(tmp_path, request):
    actor = tmp_path / "train_actor.py"
    actor.write_text((TESTDATA / "train_actor.py.input").read_text())

    patch_training_role._patch_file(actor)
    patched = actor.read_text()
    if request.config.getoption("--rewrite"):
        (TESTDATA / "train_actor.py.output").write_text(patched)
    else:
        assert patched == (TESTDATA / "train_actor.py.output").read_text()

    patch_training_role._patch_file(actor)

    assert "        self.args.training_gym_role = role\n" in patched
    assert actor.read_text() == patched
