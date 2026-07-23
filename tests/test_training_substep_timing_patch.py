from pathlib import Path

from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_training_substep_timing,
)


TESTDATA = Path(__file__).parent / "testdata" / "slime_actor"


def test_pinned_megatron_native_timing_hook_order():
    assert (TESTDATA / "megatron_model_hooks.txt.input").read_text().splitlines() == [
        "[log_prob]",
        "# Turn on evaluation mode which disables dropout.",
        "custom_before_log_prob_hook(args, model, store_prefix)",
        "forward_backward_func = get_forward_backward_func()",
        "forward_data_store += forward_backward_func(",
        "[train_step]",
        "# Set grad to zero.",
        "optimizer.zero_grad()",
        "custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)",
        "# Forward pass.",
        "losses_reduced = forward_backward_func(",
    ]


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
