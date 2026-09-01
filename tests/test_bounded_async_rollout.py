from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from modal_training_gym.frameworks.slime import bounded_async_rollout


@pytest.fixture
def fake_slime(monkeypatch: pytest.MonkeyPatch) -> None:
    sglang = ModuleType("slime.rollout.sglang_rollout")

    class _GenerateState:
        def __init__(self, args) -> None:
            self.sampling_params = {}

    sglang.GenerateState = _GenerateState
    for name, module in {
        "slime": ModuleType("slime"),
        "slime.rollout": ModuleType("slime.rollout"),
        "slime.rollout.sglang_rollout": sglang,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_dead_producer_thread_surfaces_original_error(fake_slime: None) -> None:
    args = SimpleNamespace(rollout_batch_size=1, rollout_max_staleness=None)
    worker = bounded_async_rollout.AsyncRolloutWorker(args, None, concurrency=1)

    async def fail() -> None:
        raise ValueError("producer boom")

    worker._loop = fail
    worker.start()
    assert worker.worker_thread is not None
    worker.worker_thread.join(timeout=2)

    with pytest.raises(RuntimeError, match="producer failed") as exc_info:
        worker.raise_if_failed()
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "producer boom"
