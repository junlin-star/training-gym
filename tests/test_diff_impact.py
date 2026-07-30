from __future__ import annotations

import scripts.diff_impact as diff_impact
from scripts.diff_impact import analyze_diff


def test_model_file_diff_maps_to_related_tutorials() -> None:
    diff = (
        "diff --git a/modal_training_gym/common/models/qwen3_8b.py "
        "b/modal_training_gym/common/models/qwen3_8b.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/modal_training_gym/common/models/qwen3_8b.py\n"
        "+++ b/modal_training_gym/common/models/qwen3_8b.py\n"
        "@@ -1,3 +1,3 @@\n"
    )

    report = analyze_diff(diff)

    assert "Qwen3_8B" in report.affected_classes
    tutorial_slugs = {slug for slug, _, _ in report.affected_tutorials}
    assert "agent/000_agent_sandbox" in tutorial_slugs
    assert "rl/003_on_policy_distillation" in tutorial_slugs


def test_generated_tutorial_diff_maps_back_to_source() -> None:
    diff = (
        "diff --git a/tutorials/rl/003_on_policy_distillation/"
        "003_on_policy_distillation.py "
        "b/tutorials/rl/003_on_policy_distillation/003_on_policy_distillation.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/tutorials/rl/003_on_policy_distillation/"
        "003_on_policy_distillation.py\n"
        "+++ b/tutorials/rl/003_on_policy_distillation/"
        "003_on_policy_distillation.py\n"
        "@@ -1,3 +1,3 @@\n"
    )

    report = analyze_diff(diff)

    assert "rl/003_on_policy_distillation" in {
        slug for slug, _, _ in report.affected_tutorials
    }


def test_harness_skip_does_not_affect_model_specific_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        diff_impact,
        "_model_index",
        lambda: (
            {"SkippedClass": frozenset({"Skipped"})},
            frozenset({"Skipped", "Validated"}),
        ),
    )

    harness_diff = (
        "diff --git a/scripts/diff_impact.py b/scripts/diff_impact.py\n"
        "--- a/scripts/diff_impact.py\n"
        "+++ b/scripts/diff_impact.py\n"
    )
    assert diff_impact.affected_models(
        harness_diff,
        skip_harness_validation_models=["org/Skipped"],
    ) == ("Validated",)

    monkeypatch.setattr(
        diff_impact,
        "analyze_diff",
        lambda _diff: diff_impact.ImpactReport(
            affected_classes=("SkippedClass",),
            affected_tutorials=(),
        ),
    )
    model_diff = "diff --git a/model.py b/model.py\n--- a/model.py\n+++ b/model.py\n"
    assert diff_impact.affected_models(
        model_diff,
        skip_harness_validation_models=["org/Skipped"],
    ) == ("Skipped",)
