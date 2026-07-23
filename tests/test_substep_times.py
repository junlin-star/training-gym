"""Golden-file test for the Slime rollout-status patcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_rollout_status_reporting as patcher,
)

TESTDATA = Path(__file__).parent / "testdata"


@pytest.fixture(scope="session")
def slime_inputs() -> dict[str, str]:
    inputs = sorted(TESTDATA.glob("*.input"))
    assert inputs
    return {path.name.removesuffix(".input"): path.read_text() for path in inputs}


def test_missing_patch_target_is_skipped(tmp_path, capsys):
    missing = tmp_path / "train_async.py"
    patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping rollout-status patch" in capsys.readouterr().out


def test_patch_matches_golden(slime_inputs, tmp_path, request):
    rewrite_goldens = request.config.getoption("--rewrite")
    for name, source in slime_inputs.items():
        golden_path = TESTDATA / f"{name}.output"
        work = tmp_path / name
        work.write_text(source)
        patcher._patch_file(work)
        actual = work.read_text()

        if rewrite_goldens:
            golden_path.write_text(actual)
            continue

        assert golden_path.exists(), (
            f"Golden output file does not exist: {golden_path}. "
            "Regenerate and review the expected patch output with "
            "uv run pytest tests/test_substep_times.py --rewrite."
        )
        expected = golden_path.read_text()
        assert actual == expected, (
            f"golden mismatch for {name}; rerun with --rewrite to accept"
        )
        if name == "train_async.py":
            generation_finished = actual.index(
                "_tg_report('generate_rollouts', args, rollout_id + 1, 'phase_finish')"
            )
            weight_sync_started = actual.index(
                "'weight_sync', args, rollout_id, 'phase_start'"
            )
            weight_update = actual.index(
                "actor_model.update_weights()", weight_sync_started
            )
            weight_sync_finished = actual.index(
                "_tg_report('weight_sync', args, rollout_id, 'phase_finish')"
            )
            assert (
                generation_finished
                < weight_sync_started
                < weight_update
                < weight_sync_finished
            )

        patcher._patch_file(work)
        assert work.read_text() == actual, f"patch is not idempotent for {name}"


def test_async_patch_does_not_write_partial_changes(slime_inputs, tmp_path, capsys):
    source = slime_inputs["train_async.py"].replace(
        "            ray.get(rollout_manager.eval.remote(rollout_id))\n",
        "            ray.get(rollout_manager.changed_eval.remote(rollout_id))\n",
    )
    work = tmp_path / "train_async.py"
    work.write_text(source)

    patcher._patch_file(work)

    assert work.read_text() == source
    assert "Could not patch train_async.py" in capsys.readouterr().out
