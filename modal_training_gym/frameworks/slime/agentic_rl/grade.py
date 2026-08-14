"""Slime reward adapter over Training Gym's fresh-sandbox SWE grader."""

from modal_training_gym.common.environments.swerebench import grade_swe_patch


def grade_detailed(task: dict, model_patch: str, *, timeout: int = 1800) -> dict:
    verdict = grade_swe_patch(task, model_patch, timeout=timeout)
    metadata = verdict.metadata
    baseline_passed = set(metadata.get("baseline_passed", []))
    fixable = [test for test in task["FAIL_TO_PASS"] if test not in baseline_passed]
    return {
        "reward": 1.0 if verdict.passed else 0.0,
        "dense": float(metadata.get("dense_reward", 0.0)),
        "passed": metadata.get("passed", []),
        "base_passed": metadata.get("baseline_passed", []),
        "required": metadata.get("required", []),
        "missing": metadata.get("missing", []),
        "n_fixable": len(fixable),
        "progress": float(metadata.get("progress", 0.0)),
        "p2p_frac": float(metadata.get("pass_to_pass_fraction", 1.0)),
        "output": metadata.get("output", verdict.detail),
    }


def grade(task: dict, model_patch: str, *, timeout: int = 1800) -> float:
    return grade_detailed(task, model_patch, timeout=timeout)["reward"]
