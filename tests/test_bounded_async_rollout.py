from __future__ import annotations

import importlib
import sys
import types
from enum import Enum
from types import SimpleNamespace


def _import_bounded_rollout(monkeypatch):
    class FakeSample:
        class Status(Enum):
            PENDING = "pending"
            COMPLETED = "completed"
            ABORTED = "aborted"

        def __init__(self, status):
            self.status = status

    modules = {
        "slime": types.ModuleType("slime"),
        "slime.rollout": types.ModuleType("slime.rollout"),
        "slime.rollout.sglang_rollout": types.ModuleType(
            "slime.rollout.sglang_rollout"
        ),
        "slime.utils": types.ModuleType("slime.utils"),
        "slime.utils.async_utils": types.ModuleType("slime.utils.async_utils"),
        "slime.utils.http_utils": types.ModuleType("slime.utils.http_utils"),
        "slime.utils.misc": types.ModuleType("slime.utils.misc"),
        "slime.utils.types": types.ModuleType("slime.utils.types"),
    }
    modules["slime"].__path__ = []
    modules["slime.rollout"].__path__ = []
    modules["slime.utils"].__path__ = []
    modules["slime.rollout.sglang_rollout"].GenerateState = lambda args: (
        SimpleNamespace(sampling_params={})
    )
    modules["slime.rollout.sglang_rollout"].generate_and_rm_group = None
    modules["slime.utils.async_utils"].run = lambda value: value
    modules["slime.utils.http_utils"].get_rollout_num_engines = lambda args: 1
    modules["slime.utils.misc"].load_function = lambda path: None
    modules["slime.utils.types"].Sample = FakeSample
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    target = "modal_training_gym.frameworks.slime.bounded_async_rollout"
    sys.modules.pop(target, None)
    return importlib.import_module(target), FakeSample


def test_bounded_pool_uses_staleness_window(monkeypatch):
    rollout, _ = _import_bounded_rollout(monkeypatch)
    args = SimpleNamespace(rollout_max_staleness=2, rollout_batch_size=4)

    worker = rollout.AsyncRolloutWorker(args, object(), concurrency=64)

    assert worker.pool_limit == 8


def test_aborted_group_is_fully_regenerated(monkeypatch):
    rollout, Sample = _import_bounded_rollout(monkeypatch)

    class Buffer:
        def __init__(self):
            self.groups = []

        def add_samples(self, groups):
            self.groups.extend(groups)

    buffer = Buffer()
    args = SimpleNamespace(rollout_max_staleness=2, rollout_batch_size=4)
    worker = rollout.AsyncRolloutWorker(args, buffer, concurrency=64)
    group = [
        Sample(Sample.Status.ABORTED),
        Sample(Sample.Status.COMPLETED),
    ]
    worker.inflight_gids.add(7)

    worker._make_done_cb(7)(SimpleNamespace(result=lambda: group))

    assert buffer.groups == [group]
    assert all(sample.status is Sample.Status.PENDING for sample in group)
    assert 7 not in worker.inflight_gids
