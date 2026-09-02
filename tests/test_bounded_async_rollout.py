from __future__ import annotations

import asyncio
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
    misc = ModuleType("slime.utils.misc")
    misc.load_function = lambda path: None
    for name, module in {
        "slime": ModuleType("slime"),
        "slime.rollout": ModuleType("slime.rollout"),
        "slime.rollout.sglang_rollout": sglang,
        "slime.utils": ModuleType("slime.utils"),
        "slime.utils.misc": misc,
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


def test_groups_older_than_max_staleness_are_dropped(
    fake_slime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        rollout_batch_size=1,
        rollout_max_staleness=2,
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
    )
    pending = [(0, [SimpleNamespace(index=0)]), (1, [SimpleNamespace(index=1)])]
    worker = SimpleNamespace(
        completed_buffer={},
        launch_rid={0: 0, 1: 1},
        inflight_gids=set(),
        raise_if_failed=lambda: None,
        queue_size=lambda: 0,
        get_completed_groups=lambda: [pending.pop(0)] if pending else [],
    )
    monkeypatch.setattr(
        bounded_async_rollout, "_get_global_worker", lambda a, d: worker
    )
    logged: dict[str, float] = {}
    monkeypatch.setattr(
        bounded_async_rollout,
        "_log_rollout_metrics",
        lambda a, rid, metrics: logged.update(metrics),
    )

    out = asyncio.run(bounded_async_rollout._generate_rollout_async(args, 3, None))

    assert [s.index for group in out for s in group] == [1]
    assert logged["rollout/stale_groups_dropped"] == 1


def test_requeued_group_drops_launch_rid(
    fake_slime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    types_mod = ModuleType("slime.utils.types")
    types_mod.Sample = SimpleNamespace(Status=SimpleNamespace(PENDING=0, ABORTED=1))
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)
    args = SimpleNamespace(rollout_batch_size=1, rollout_max_staleness=None)
    data_buffer = SimpleNamespace(add_samples=lambda groups: None)
    worker = bounded_async_rollout.AsyncRolloutWorker(args, data_buffer, concurrency=1)
    worker.launch_rid[0] = 0

    async def run() -> None:
        task = asyncio.ensure_future(asyncio.sleep(0, result="not a list"))
        await task
        worker._make_done_cb(0, [SimpleNamespace()])(task)

    asyncio.run(run())

    assert 0 not in worker.launch_rid
    assert worker.output_queue.empty()
