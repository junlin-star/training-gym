"""
Input: string [ model name ]
Output: string [ Formatted test result ]
Optional args:
    -j: json formatted output
    -o: output file path
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum

from modal_training_gym.common.train import TrainStepStatus

class TutorialStatus(Enum):
    SUCCESS = "success"
    OUT_OF_MEMORY = "oom"
    GYM_ERROR = "gym_error"
    SLIME_ERROR = "slime_error"
    TIMEOUT = "timeout"


@dataclass
class StepResult:
    step_count: int
    step_duration_s: float # Extract from the step update time
    substep_duration_s: dict[TrainStepStatus, float] # Extract from the substep update time


@dataclass
class TutorialResult:
    status: TutorialStatus
    total_duration_s: float
    step_results: list[StepResult]
    training_run_id: str # <- format it into a url

    @classmethod
    def format_tutorial_result(cls, tutorial_result: "TutorialResult") -> str:
        is_valid = duration_is_valid(tutorial_result)
        pass

# TODO(melody): Implement
def duration_is_valid(tutorial_result: TutorialResult) -> bool:
    return True



# uv run python tutorials/rl/001_sandboxes/001_sandboxes.py


def test_model(model_name: ModelConfig, step_count = 1) -> TutorialResult:
    # default slime recipe for this model

    train_config = TrainConfig(
            model=model_name,
            dataset=MathDataset(n_rows=10), 
        )

    training_run_id = train_config.training_run_id
    run = train_config.train()

    dashboard_url = get_dashboard_url()
    dashboard_url_base = dashboard_url.rstrip("/")

    run = TrainingRun.from_id(training_run_id)

    status = run.status
    framework_status = run.framework_status
    updated_at = run.updated_at

    progress = (run.metadata or {}).get("framework_progress") or {}
    phase = progress.get("phase")
    phase_updated_at = progress.get("updated_at")
    current = progress.get("current")
    total = progress.get("total")
    rollout_id = progress.get("rollout_id")
    step_id = progress.get("step_id")
    is_active = progress.get("is_active")

    

    run = get_json(f"{dashboard_url_base}/api/training-runs/{training_run_id}")

    start = time.time()

    # grab the training run id
    web.post("/api/framework-status")

    training_run_id = str(payload.get("training_run_id", "") or "").strip()
    run = TrainingRun.from_id(training_run_id)
    # look it up via dashboard api and use api routes to get all this metadata

    step_results = []
    for step in range(step_count):
        Step = StepResult()
        last_updated = None
        last_phase = None
        while True: 
            progress = (run.metadata or {}).get("framework_progress") or {}
            if (progress.phase != last_phase):
                Step.substep_duration_s[progress.get("status")] = progress.get("updated_at") - last_updated
                last_updated = progress.get("updated_at")
                last_phase = progress.phase


            step_results.append(Step)


    return TutorialResult(
        step_results=step_results
    )
