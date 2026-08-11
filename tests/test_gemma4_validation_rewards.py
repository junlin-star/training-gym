"""Reward scoring for the Gemma-4 validation runs.

These decide whether a run produces gradient signal at all, so they are worth
pinning before spending a cluster on them: a reward that cannot distinguish
right from wrong, or that every sample scores identically on, yields
zero-variance GRPO groups and a flat curve.
"""

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "validate_gemma4_runs",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "validate_gemma4_runs.py",
)
runs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runs)


# ── Boxed-answer extraction ──────────────────────────────────────────────────


def test_boxed_takes_the_last_answer():
    assert runs._boxed("first \\boxed{1} then \\boxed{2}") == "2"


def test_boxed_handles_nested_braces():
    assert runs._boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_boxed_returns_none_without_a_box():
    assert runs._boxed("the answer is 42") is None


def test_boxed_returns_none_on_an_unclosed_box():
    assert runs._boxed("\\boxed{42") is None


# ── Numeric (GSM8K) ──────────────────────────────────────────────────────────


def test_numeric_full_credit_for_a_boxed_correct_answer():
    assert runs.score_numeric("...so \\boxed{72}", "72") == 1.0


def test_numeric_partial_credit_when_correct_but_unboxed():
    assert runs.score_numeric("the answer is 72", "72") == 0.3


def test_numeric_zero_for_a_boxed_wrong_answer():
    assert runs.score_numeric("\\boxed{71}", "72") == 0.0


def test_numeric_penalises_an_unparseable_response():
    assert runs.score_numeric("I don't know", "72") == -0.5


def test_numeric_ignores_thousands_separators_and_currency():
    assert runs.score_numeric("\\boxed{$1,234}", "1234") == 1.0


def test_numeric_boxed_wrong_beats_nothing_but_loses_to_unboxed_right():
    """The ordering is what creates spread inside a group."""
    right_boxed = runs.score_numeric("\\boxed{72}", "72")
    right_plain = runs.score_numeric("answer: 72", "72")
    wrong = runs.score_numeric("\\boxed{13}", "72")
    unparseable = runs.score_numeric("hmm", "72")

    assert right_boxed > right_plain > wrong > unparseable


# ── Multiple choice (AQuA-RAT) ───────────────────────────────────────────────


def test_choice_full_credit_for_the_boxed_letter():
    assert runs.score_choice("so \\boxed{C}", "C") == 1.0


def test_choice_is_case_insensitive():
    assert runs.score_choice("\\boxed{c}", "C") == 1.0


def test_choice_partial_credit_for_a_trailing_letter():
    assert runs.score_choice("I pick D", "D") == 0.3


def test_choice_zero_for_the_wrong_letter():
    assert runs.score_choice("\\boxed{A}", "B") == 0.0


def test_choice_penalises_a_response_with_no_option():
    assert runs.score_choice("cannot tell", "B") == -0.5


# ── ChartQA ──────────────────────────────────────────────────────────────────


def test_chart_accepts_a_value_within_five_percent():
    assert runs.score_chart("\\boxed{102}", "100") == 1.0


def test_chart_rejects_a_value_outside_five_percent():
    assert runs.score_chart("\\boxed{120}", "100") == 0.0


def test_chart_matches_text_answers_case_insensitively():
    assert runs.score_chart("\\boxed{Yes}", "yes") == 1.0


def test_chart_partial_credit_when_unboxed():
    assert runs.score_chart("100", "100") == 0.3


def test_chart_penalises_an_empty_response():
    assert runs.score_chart("", "100") == -0.5


# ── Grounding (ScreenSpot) ───────────────────────────────────────────────────


BOX = "0.20,0.20,0.40,0.40"


def test_grounding_full_credit_inside_the_box():
    assert runs.score_grounding("\\boxed{0.3,0.3}", BOX) == 1.0


def test_grounding_partial_credit_inside_but_unboxed():
    assert runs.score_grounding("0.3, 0.3", BOX) == 0.6


def test_grounding_floor_far_outside_the_box():
    assert runs.score_grounding("\\boxed{0.95,0.95}", BOX) == -1.0


def test_grounding_decays_between_edge_and_margin():
    near = runs.score_grounding("\\boxed{0.42,0.30}", BOX)
    far = runs.score_grounding("\\boxed{0.50,0.30}", BOX)

    assert -1.0 < far < near < 1.0


def test_grounding_rejects_unparseable_points():
    assert runs.score_grounding("no idea", BOX) == -1.0


# ── Spread ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "score,label,responses",
    [
        (
            runs.score_numeric,
            "72",
            ["\\boxed{72}", "answer: 72", "\\boxed{13}", "no idea"],
        ),
        (runs.score_choice, "C", ["\\boxed{C}", "I pick C", "\\boxed{A}", "unsure"]),
        (
            runs.score_grounding,
            BOX,
            ["\\boxed{0.3,0.3}", "0.3,0.3", "\\boxed{0.45,0.3}", "nope"],
        ),
    ],
)
def test_a_mixed_group_is_not_zero_variance(score, label, responses):
    """The failure mode these runs exist to avoid: every sample scoring alike."""
    scores = [score(r, label) for r in responses]
    assert len(set(scores)) > 1


# ── Gemma's 0-1000 grounding grid ────────────────────────────────────────────


def test_grounding_accepts_gemmas_thousand_grid():
    """Gemma answers 933,56 rather than 0.933,0.056; both must score alike."""
    assert runs.score_grounding("\\boxed{300,300}", BOX) == 1.0
    assert runs.score_grounding("\\boxed{0.3,0.3}", BOX) == 1.0


def test_grounding_thousand_grid_still_penalises_a_miss():
    assert runs.score_grounding("\\boxed{950,950}", BOX) == -1.0


def test_grounding_rejects_coordinates_past_the_grid():
    assert runs.score_grounding("\\boxed{1400,300}", BOX) == -1.0


def test_grounding_observed_pixel_answer_is_scored_not_floored():
    """Regression: this exact response scored the -1.0 floor on the first run."""
    assert runs.score_grounding("\\boxed{933,56}", "0.90,0.02,0.99,0.10") == 1.0
