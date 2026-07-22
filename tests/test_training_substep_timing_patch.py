from pathlib import Path

from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_training_substep_timing,
)


TESTDATA = Path(__file__).parent / "testdata" / "slime_actor"


def test_training_substep_timing_patch_matches_pinned_slime(tmp_path, request):
    source = (TESTDATA / "megatron_actor.py.input").read_text()
    actor = tmp_path / "actor.py"
    actor.write_text(source)

    patch_training_substep_timing._patch_file(actor)
    patched = actor.read_text()
    if request.config.getoption("--rewrite"):
        (TESTDATA / "megatron_actor.py.output").write_text(patched)
    else:
        assert patched == (TESTDATA / "megatron_actor.py.output").read_text()
    assert patched.index("result = self.train_critic(rollout_id, rollout_data)") < (
        patched.index("_tg_flush_timings()")
    )
    assert patched.index("_tg_flush_timings()") < patched.index("return result")

    patch_training_substep_timing._patch_file(actor)
    assert actor.read_text() == patched
