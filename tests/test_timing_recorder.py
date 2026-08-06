from __future__ import annotations

import threading

from modal_training_gym.common import status_reporter
from modal_training_gym.common.timing_recorder import RoleRecorder


def test_missing_timing_route_latches_off_and_warns_for_require(monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "require")
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "run-1")

    posted = []
    first_post = threading.Event()

    def post_item_result(item):
        posted.append(item)
        first_post.set()
        return "not_found"

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)

    recorder = RoleRecorder("driver", 0)
    with recorder.phase("train"):
        pass
    assert first_post.wait(timeout=1)

    with recorder.phase("train"):
        pass
    recorder.__exit__(None, None, None)
    if recorder._poster is not None:
        recorder._poster.join(timeout=1)

    assert len(posted) == 1
    assert (
        "substep_timing='require' was rejected with HTTP 404" in capsys.readouterr().out
    )
