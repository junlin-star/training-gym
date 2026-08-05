"""Golden-file test for the substep-timing patcher.

Anchors are literal source lines, so the test worth having is one that runs them
over the sources they were written for: ``tests/testdata`` holds slime's
``train.py`` after the rollout-status patcher (the state the timing patcher sees)
and miles' two entrypoints as shipped in the pinned image. Regenerate the
expected output with ``uv run pytest tests/test_substep_timing_patch.py
--rewrite``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TESTDATA = Path(__file__).parent / "testdata"
PATCHER_PATH = (
    Path(__file__).parents[1]
    / "modal_training_gym"
    / "common"
    / "patch_scripts"
    / "patch_substep_timing.py"
)

# fixture, entrypoint, golden output, driver phases the loop must record.
DRIVERS = [
    (
        "train.py.output",
        "train.py",
        "slime/train.py.timing.output",
        {
            "evaluate_rollouts",
            "generate_rollouts",
            "offload_rollout",
            "train_models",
            "checkpoint_save",
            "offload_train",
            "weight_sync",
            "evaluate_rollouts_end",
        },
    ),
    (
        "miles/train.py.input",
        "train.py",
        "miles/train.py.timing.output",
        {
            "evaluate_rollouts",
            "generate_rollouts",
            "offload_rollout",
            "train_models",
            "checkpoint_save",
            "offload_train",
            "weight_sync",
            "evaluate_rollouts_end",
        },
    ),
    (
        "miles/train_async.py.input",
        "train_async.py",
        "miles/train_async.py.timing.output",
        {
            "wait_for_rollout",
            "train_models",
            "checkpoint_save",
            "weight_sync",
            "evaluate_rollouts_end",
        },
    ),
]


@pytest.fixture(scope="session")
def patcher():
    """Load the patch script by path: it runs standalone in the image, and is
    deliberately not importable as part of the package."""
    spec = importlib.util.spec_from_file_location("patch_substep_timing", PATCHER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass resolves annotations here
    spec.loader.exec_module(module)
    return module


def _patched(patcher, tmp_path, fixture: str, entrypoint: str) -> str:
    entrypoints = (
        patcher.MILES_ENTRYPOINTS
        if fixture.startswith("miles/")
        else patcher.SLIME_ENTRYPOINTS
    )
    work = tmp_path / entrypoint
    work.write_text((TESTDATA / fixture).read_text())
    patcher._patch_file(work, entrypoints[entrypoint])
    return work.read_text()


@pytest.mark.parametrize("fixture, entrypoint, golden, expected_phases", DRIVERS)
def test_patch_matches_golden(
    patcher, tmp_path, request, fixture, entrypoint, golden, expected_phases
):
    patched = _patched(patcher, tmp_path, fixture, entrypoint)
    golden_path = TESTDATA / golden

    if request.config.getoption("--rewrite"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(patched)
        return

    assert patched == golden_path.read_text(), (
        f"golden mismatch for {golden}; rerun with --rewrite to accept"
    )
    assert "with _tg_role('driver', rollout_id) as _tg_rec:" in patched
    for phase in expected_phases:
        assert f"with _tg_rec.phase('{phase}'):" in patched
    compile(patched, entrypoint, "exec")


def test_a_conditional_phase_is_timed_inside_its_branch(patcher, tmp_path):
    """A skipped save must record nothing, not a 0s bar on every rollout.

    Its condition spans three lines, so the closing ``):`` sits at the ``if``'s
    own indent and must not be read as the start of another clause.
    """
    patched = _patched(patcher, tmp_path, "train.py.output", "train.py")
    save = patched.split("if release_train or should_run_periodic_action(")[1]
    before_wrap = save.split("with _tg_rec.phase('checkpoint_save'):")[0]
    assert before_wrap.split("#")[0].rstrip().endswith("):")


def test_patching_twice_is_a_no_op(patcher, tmp_path, capsys):
    work = tmp_path / "train.py"
    work.write_text((TESTDATA / "miles/train.py.input").read_text())
    patcher._patch_file(work, patcher.MILES_ENTRYPOINTS["train.py"])
    once = work.read_text()
    patcher._patch_file(work, patcher.MILES_ENTRYPOINTS["train.py"])
    assert work.read_text() == once
    assert "already patched" in capsys.readouterr().out


def test_a_moved_anchor_fails_the_build(patcher, tmp_path):
    """Half-instrumented timing is worse than none: a lane would just be absent."""
    work = tmp_path / "train.py"
    source = (TESTDATA / "miles/train.py.input").read_text()
    work.write_text(
        source.replace("await offload_train()", "await offload_train(args)")
    )
    with pytest.raises(RuntimeError, match="expected 1 occurrence"):
        patcher._patch_file(work, patcher.MILES_ENTRYPOINTS["train.py"])


def test_missing_package_file_fails_the_build(patcher, tmp_path):
    with pytest.raises(RuntimeError, match="layout changed"):
        patcher.patch_package_file(tmp_path, patcher.MILES_PACKAGE_TARGETS[0])
