from __future__ import annotations

import threading
import time
from urllib.error import HTTPError

import pytest

from modal_training_gym.common import status_reporter
from modal_training_gym.common import timing_recorder
from modal_training_gym.common.timing_recorder import RoleRecorder


@pytest.fixture(autouse=True)
def _reset_timing_mode_cache():
    timing_recorder.reset_timing_mode_cache()
    yield
    timing_recorder.reset_timing_mode_cache()


def test_status_reporter_retries_throttled_errors(monkeypatch):
    def raise_http_error(code):
        raise HTTPError("https://dashboard.test", code, "test", {}, None)

    for code in (408, 425, 429):
        monkeypatch.setattr(
            status_reporter,
            "urlopen",
            lambda request, timeout, code=code: raise_http_error(code),
        )
        assert (
            status_reporter._post(
                {"_url": "https://dashboard.test", "payload": "value"}
            )
            == "failed"
        )

    for code in (401, 403):
        monkeypatch.setattr(
            status_reporter,
            "urlopen",
            lambda request, timeout, code=code: raise_http_error(code),
        )
        assert (
            status_reporter._post(
                {"_url": "https://dashboard.test", "payload": "value"}
            )
            == "auth_failed"
        )

    monkeypatch.setattr(
        status_reporter,
        "urlopen",
        lambda request, timeout: raise_http_error(422),
    )
    assert (
        status_reporter._post({"_url": "https://dashboard.test", "payload": "value"})
        == "permanent"
    )


def test_auth_rejections_latch_lane(monkeypatch):
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "auth-run")
    posted = []
    done = threading.Event()

    def post_item_result(item):
        posted.append(item)
        done.set()
        return "auth_failed"

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)
    recorder = RoleRecorder("driver", 0)
    with recorder.phase("train"):
        pass
    assert done.wait(timeout=1)
    recorder.__exit__(None, None, None)
    if recorder._poster is not None:
        recorder._poster.join(timeout=1)
    assert len(posted) == timing_recorder.AUTH_REJECTION_LATCH_THRESHOLD
    assert recorder._auth_rejected

    with recorder.phase("forward_backward"):
        pass
    assert len(posted) == timing_recorder.AUTH_REJECTION_LATCH_THRESHOLD


def test_transient_failures_do_not_latch_auth(monkeypatch):
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "auth-run")
    attempts = iter(["auth_failed", "failed", "auth_failed", "failed", "auth_failed"])
    posted = []
    done = threading.Event()

    def post_item_result(item):
        posted.append(item)
        done.set()
        return next(attempts)

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)
    recorder = RoleRecorder("driver", 0)
    with recorder.phase("train"):
        pass
    assert done.wait(timeout=1)
    recorder.__exit__(None, None, None)
    if recorder._poster is not None:
        recorder._poster.join(timeout=1)
    assert not recorder._auth_rejected


def test_success_resets_auth_rejections(monkeypatch):
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "auth-run")
    attempts = iter(["auth_failed", "ok", "auth_failed", "ok"])
    posted = []
    done = threading.Event()

    def post_item_result(item):
        posted.append(item)
        done.set()
        return next(attempts)

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)
    recorder = RoleRecorder("driver", 0)
    with recorder.phase("train"):
        pass
    assert done.wait(timeout=1)
    recorder.__exit__(None, None, None)
    if recorder._poster is not None:
        recorder._poster.join(timeout=1)
    assert recorder._auth_rejections == 0
    assert not recorder._auth_rejected


def test_missing_timing_route_latches_off_and_warns(monkeypatch, capsys):
    monkeypatch.setattr(timing_recorder, "_UNSUPPORTED", False)
    monkeypatch.setattr(timing_recorder, "_NOT_FOUND_COUNT", 0)
    monkeypatch.setattr(timing_recorder, "_FAILURE_REPORTED", False)
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "auto")
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "run-1")

    posted = []
    first_post = threading.Event()

    def post_item_result(item):
        posted.append(item)
        first_post.set()
        return "not_found"

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)

    for _ in range(timing_recorder.NOT_FOUND_LATCH_THRESHOLD):
        recorder = RoleRecorder("driver", _)
        with recorder.phase("train"):
            pass
        assert first_post.wait(timeout=1)
        first_post.clear()
        recorder.__exit__(None, None, None)
        if recorder._poster is not None:
            recorder._poster.join(timeout=1)
        assert len(posted) == _ + 1

    recorder = RoleRecorder("driver", timing_recorder.NOT_FOUND_LATCH_THRESHOLD)
    with recorder.phase("train"):
        pass
    assert len(posted) == timing_recorder.NOT_FOUND_LATCH_THRESHOLD
    assert timing_recorder._UNSUPPORTED
    output = capsys.readouterr().out
    assert output.count("WARNING: this dashboard is too old for substep timing") == 1


def test_off_skips_lane_creation_and_phase_timing(monkeypatch):
    def fail_if_called(*args):
        raise AssertionError("off must not create a RoleRecorder")

    def fail_timestamp():
        raise AssertionError("off must not timestamp")

    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "off")
    monkeypatch.setattr(timing_recorder, "RoleRecorder", fail_if_called)
    monkeypatch.setattr(timing_recorder.time, "monotonic", fail_timestamp)

    with timing_recorder.recording_lane("driver", 0) as recorder:
        with recorder.phase("train"):
            pass
    with timing_recorder.time_phase("forward_backward"):
        pass


def test_failed_snapshot_retries_without_close_duplicate(monkeypatch):
    monkeypatch.setattr(timing_recorder, "_UNSUPPORTED", False)
    monkeypatch.setattr(timing_recorder, "_NOT_FOUND_COUNT", 0)
    monkeypatch.setattr(timing_recorder, "_FAILURE_REPORTED", False)
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "run-1")

    posted = []
    attempts = iter(["failed", "ok"])
    second_post = threading.Event()

    def post_item_result(item):
        posted.append(item)
        if len(posted) == 2:
            second_post.set()
        return next(attempts)

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)

    recorder = RoleRecorder("driver", 0)
    with recorder.phase("train"):
        pass
    recorder.__exit__(None, None, None)

    assert second_post.wait(timeout=1)
    if recorder._poster is not None:
        recorder._poster.join(timeout=1)
    assert len(posted) == 2


def test_closing_lane_retries_final_snapshot_after_exhaustion(monkeypatch):
    monkeypatch.setattr(timing_recorder, "MAX_POST_RETRIES", 2)
    monkeypatch.setattr(timing_recorder, "MAX_RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "closing-run")

    posted = []
    first_post = threading.Event()
    release_first_post = threading.Event()
    delivered = threading.Event()
    attempts = iter(["failed", "failed", "ok"])

    def post_item_result(item):
        posted.append(item)
        if len(posted) == 1:
            first_post.set()
            release_first_post.wait(timeout=1)
        result = next(attempts)
        if result == "ok":
            delivered.set()
        return result

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)

    recorder = RoleRecorder("rollout", 0)
    with recorder.phase("generate_samples"):
        pass
    recorder.__exit__(None, None, None)
    assert first_post.wait(timeout=1)
    release_first_post.set()

    assert delivered.wait(timeout=1)
    if recorder._poster is not None:
        recorder._poster.join(timeout=1)
    assert len(posted) == 3
    assert posted[-1]["phases"]["generate_samples"]["count"] == 1


def test_closing_lane_retry_deadline_bounds_shutdown(monkeypatch):
    monkeypatch.setattr(timing_recorder, "MAX_POST_RETRIES", 1)
    monkeypatch.setattr(timing_recorder, "MAX_RETRY_BACKOFF_S", 0.01)
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setattr(timing_recorder, "TIMING_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "closing-timeout-run")

    first_post = threading.Event()
    release_first_post = threading.Event()

    def post_item_result(item):
        first_post.set()
        release_first_post.wait(timeout=1)
        return "failed"

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)

    recorder = RoleRecorder("rollout", 0)
    with recorder.phase("generate_samples"):
        pass
    recorder.__exit__(None, None, None)
    assert first_post.wait(timeout=1)
    release_first_post.set()

    started = time.monotonic()
    timing_recorder._drain_closed_posters()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert recorder._poster is not None
    assert not recorder._poster.is_alive()


def test_preloop_lanes_accumulate_on_one_recorder(monkeypatch):
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "preloop-run")
    timing_recorder._PRELOOP_RECORDERS.clear()
    timing_recorder._CLOSED_POSTERS.clear()
    posted = []
    posted_event = threading.Event()

    def post_item_result(item):
        posted.append(item)
        posted_event.set()
        return "ok"

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)

    for phase in ("initial_weight_sync", "evaluate_rollouts", "evaluate_rollouts_end"):
        with timing_recorder.recording_lane("driver", None) as recorder:
            with recorder.phase(phase):
                pass
        assert posted_event.wait(timeout=1)
        posted_event.clear()

    shared = timing_recorder._PRELOOP_RECORDERS["driver"]
    assert recorder is shared
    assert set(shared.phases) == {
        "initial_weight_sync",
        "evaluate_rollouts",
        "evaluate_rollouts_end",
    }
    assert all(
        item["lane_start_unix_s"] == posted[0]["lane_start_unix_s"] for item in posted
    )
    assert not shared._closed

    shared._close()
    if shared._poster is not None:
        shared._poster.join(timeout=1)


def test_permanent_rejection_latches_recorder(monkeypatch, capsys):
    monkeypatch.setattr(timing_recorder, "_UNSUPPORTED", False)
    monkeypatch.setattr(timing_recorder, "_NOT_FOUND_COUNT", 0)
    monkeypatch.setattr(timing_recorder, "_FAILURE_REPORTED", False)
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "rejected-run")

    posted = []
    first_post = threading.Event()

    def post_item_result(item):
        posted.append(item)
        first_post.set()
        return "permanent"

    monkeypatch.setattr(status_reporter, "post_item_result", post_item_result)

    recorder = RoleRecorder("driver", 0)
    with recorder.phase("train"):
        pass
    assert first_post.wait(timeout=1)
    with recorder.phase("forward_backward"):
        pass
    recorder.__exit__(None, None, None)
    if recorder._poster is not None:
        recorder._poster.join(timeout=1)

    assert len(posted) == 1
    assert recorder._permanent_rejected
    assert capsys.readouterr().out.count("permanent client error") == 1
