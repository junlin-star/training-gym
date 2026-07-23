import pytest

from modal_training_gym.frameworks.slime import reporting


TIMING_EVENTS_URL = "https://dashboard.test/api/timing-events"


def _timing_batch() -> dict[str, object]:
    return {
        "_url": TIMING_EVENTS_URL,
        "_timeout": 2.0,
        "training_run_id": "run-1",
        "training_attempt": 2,
        "training_role": "driver",
        "rollout_id": 0,
        "events": [],
    }


def test_supported_timing_endpoint_receives_batch_unchanged(monkeypatch):
    posted = []
    batch = _timing_batch()
    monkeypatch.setattr(
        reporting,
        "_TRAINING_TIMING_ENDPOINT_UNSUPPORTED",
        False,
    )

    def post(item):
        posted.append(dict(item))
        return True, None

    monkeypatch.setattr(reporting, "_post", post)

    assert reporting._post_with_retries(
        batch,
        attempts=2,
        retry_delay_seconds=0,
        detect_unsupported_timing_endpoint=True,
    )
    assert posted == [batch]
    assert not reporting._training_timing_endpoint_is_unsupported()


@pytest.mark.parametrize("unsupported_status_code", [401, 405])
def test_definitive_unsupported_status_disables_timing_endpoint(
    monkeypatch,
    unsupported_status_code,
):
    posted = []
    monkeypatch.setattr(
        reporting,
        "_TRAINING_TIMING_ENDPOINT_UNSUPPORTED",
        False,
    )

    def post(item):
        posted.append(dict(item))
        return False, unsupported_status_code

    monkeypatch.setattr(reporting, "_post", post)

    assert reporting._post_with_retries(
        _timing_batch(),
        attempts=3,
        retry_delay_seconds=0,
        detect_unsupported_timing_endpoint=True,
    )
    assert len(posted) == 1
    assert {item["_url"] for item in posted} == {TIMING_EVENTS_URL}
    assert reporting._training_timing_endpoint_is_unsupported()


def test_not_found_disables_timing_endpoint_only_after_retries(monkeypatch):
    responses = iter([(False, 404), (True, None)])
    monkeypatch.setattr(
        reporting,
        "_TRAINING_TIMING_ENDPOINT_UNSUPPORTED",
        False,
    )
    monkeypatch.setattr(reporting, "_post", lambda _item: next(responses))

    assert reporting._post_with_retries(
        _timing_batch(),
        attempts=3,
        retry_delay_seconds=0,
        detect_unsupported_timing_endpoint=True,
    )
    assert not reporting._training_timing_endpoint_is_unsupported()

    posted = []

    def post(item):
        posted.append(dict(item))
        return False, 404

    monkeypatch.setattr(reporting, "_post", post)

    assert reporting._post_with_retries(
        _timing_batch(),
        attempts=3,
        retry_delay_seconds=0,
        detect_unsupported_timing_endpoint=True,
    )
    assert len(posted) == 3
    assert reporting._training_timing_endpoint_is_unsupported()


@pytest.mark.parametrize("failure_status_code", [None, 403])
def test_delivery_failure_does_not_disable_timing_endpoint(
    monkeypatch,
    failure_status_code,
):
    posted = []
    monkeypatch.setattr(
        reporting,
        "_TRAINING_TIMING_ENDPOINT_UNSUPPORTED",
        False,
    )

    def post(item):
        posted.append(dict(item))
        return False, failure_status_code

    monkeypatch.setattr(reporting, "_post", post)

    assert not reporting._post_with_retries(
        _timing_batch(),
        attempts=2,
        retry_delay_seconds=0,
        detect_unsupported_timing_endpoint=True,
    )
    assert len(posted) == 2
    assert not reporting._training_timing_endpoint_is_unsupported()
