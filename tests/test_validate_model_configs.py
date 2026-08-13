from modal_training_gym.common.run import TrainingRunStatus


def test_warns_for_empty_timings_on_completed_run(capsys):
    from scripts.validate_model_configs import _warn_if_timings_missing

    _warn_if_timings_missing("run-123", TrainingRunStatus.COMPLETED, {}, {})

    assert (
        "warning: no timing records found for completed run run-123"
        in capsys.readouterr().out
    )


def test_does_not_warn_for_incomplete_run_or_partial_timings(capsys):
    from scripts.validate_model_configs import _warn_if_timings_missing

    _warn_if_timings_missing("run-123", TrainingRunStatus.FAILED, {}, {})
    _warn_if_timings_missing("run-123", TrainingRunStatus.COMPLETED, {"0": {}}, {})

    assert capsys.readouterr().out == ""
