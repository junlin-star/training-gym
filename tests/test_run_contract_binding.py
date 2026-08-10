from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from modal_training_gym.frameworks.slime.launcher import (
    _image_overlay_contract,
    _persist_and_verify_d1a_terminal_success,
    _pop_train_function_timeout,
    _require_d1a_function_call_binding,
    _restore_d1a_terminal_binding,
    _serialize_slime_params,
    _validate_committed_dataset_inputs,
)
from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.common.train import (
    TrainConfig,
    _d1a_pre_spawn_binding_polls,
    _persist_and_verify_d1a_initial_run_record,
    _persist_and_verify_d1a_pre_spawn_app_binding,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


def _overlay(image):
    return image


def test_d1a_pre_spawn_binding_precedes_paid_spawn_and_non_d1_is_unchanged() -> None:
    source = inspect.getsource(TrainConfig.launch)
    assert source.index("_persist_and_verify_d1a_pre_spawn_app_binding") < source.index(
        "app.train.spawn("
    )
    dashboard_call = source.index("ensure_dashboard_deployed()")
    assert source.rfind(
        "if d1a_pre_spawn_binding_polls is None", 0, dashboard_call
    ) != -1
    post_spawn = source[source.index("app.train.spawn(") :]
    assert post_spawn.index("if d1a_pre_spawn_binding_polls is None") < (
        post_spawn.index("run_record.save()")
    )
    assert _d1a_pre_spawn_binding_polls(SimpleNamespace(image_env={})) is None


def test_d1a_pre_spawn_binding_retries_transient_metadata_io() -> None:
    class Record:
        save_calls = 0
        load_calls = 0

        def __init__(self) -> None:
            self.training_run_id = "d1a-matrix-aaaaaaaaaaaa"
            self.modal_app_id = "ap-exact"
            self.modal_app_url = "https://modal.com/apps/exact"
            self.function_call_id = ""
            self.metadata = {
                "event_journal_enabled": True,
                "event_journal_contract": "d1a_legacy_single_attempt_v1",
            }

        def save(self) -> None:
            type(self).save_calls += 1
            if type(self).save_calls == 1:
                raise RuntimeError("transient metadata save")

        @classmethod
        def from_id(cls, _run_id: str) -> Record:
            cls.load_calls += 1
            if cls.load_calls == 1:
                raise RuntimeError("transient metadata read")
            return cls()

    observed = _persist_and_verify_d1a_pre_spawn_app_binding(
        Record(),
        polls=3,
        poll_seconds=0,
        sleep=lambda _seconds: None,
    )
    assert observed.modal_app_id == "ap-exact"
    assert Record.save_calls == 2
    assert Record.load_calls == 2


def test_d1a_initial_record_retries_non_runtime_transport_errors() -> None:
    class Record:
        save_calls = 0
        load_calls = 0

        def __init__(self) -> None:
            self.training_run_id = "d1a-matrix-aaaaaaaaaaaa"
            self.modal_app_id = ""
            self.function_call_id = ""
            self.metadata = {
                "event_journal_enabled": True,
                "event_journal_contract": "d1a_legacy_single_attempt_v1",
            }

        def save(self) -> None:
            type(self).save_calls += 1
            if type(self).save_calls == 1:
                raise OSError("transient volume transport")

        @classmethod
        def from_id(cls, _run_id: str) -> Record:
            cls.load_calls += 1
            if cls.load_calls == 1:
                raise OSError("transient canonical read")
            return cls()

    observed = _persist_and_verify_d1a_initial_run_record(
        Record(),
        polls=2,
        poll_seconds=0,
        sleep=lambda _seconds: None,
    )
    assert observed.training_run_id == "d1a-matrix-aaaaaaaaaaaa"
    assert Record.save_calls == 2
    assert Record.load_calls == 2


def test_d1a_pre_spawn_binding_reads_canonical_after_summary_save_failure() -> None:
    class Record:
        save_calls = 0
        load_calls = 0

        def __init__(self) -> None:
            self.training_run_id = "d1a-matrix-aaaaaaaaaaaa"
            self.modal_app_id = "ap-exact"
            self.modal_app_url = "https://modal.com/apps/exact"
            self.function_call_id = ""
            self.metadata = {
                "event_journal_enabled": True,
                "event_journal_contract": "d1a_legacy_single_attempt_v1",
            }

        def save(self) -> None:
            type(self).save_calls += 1
            # TrainingRun.save() has already written the canonical item when
            # the subsequent disposable-summary upsert raises this error.
            raise OSError("summary refresh failed after canonical write")

        @classmethod
        def from_id(cls, _run_id: str) -> Record:
            cls.load_calls += 1
            return cls()

    observed = _persist_and_verify_d1a_pre_spawn_app_binding(
        Record(),
        polls=900,
        poll_seconds=1,
        sleep=lambda _seconds: pytest.fail("durable canonical state must not sleep"),
    )

    assert observed.modal_app_id == "ap-exact"
    assert Record.save_calls == 1
    assert Record.load_calls == 1


def test_d1a_initial_record_reads_canonical_after_summary_save_failure() -> None:
    class Record:
        save_calls = 0
        load_calls = 0

        def __init__(self) -> None:
            self.training_run_id = "d1a-matrix-aaaaaaaaaaaa"
            self.modal_app_id = ""
            self.function_call_id = ""
            self.metadata = {
                "event_journal_enabled": True,
                "event_journal_contract": "d1a_legacy_single_attempt_v1",
            }

        def save(self) -> None:
            type(self).save_calls += 1
            raise OSError("summary refresh failed after canonical write")

        @classmethod
        def from_id(cls, _run_id: str) -> Record:
            cls.load_calls += 1
            return cls()

    observed = _persist_and_verify_d1a_initial_run_record(
        Record(),
        polls=900,
        poll_seconds=1,
        sleep=lambda _seconds: pytest.fail("durable canonical state must not sleep"),
    )

    assert observed.training_run_id == "d1a-matrix-aaaaaaaaaaaa"
    assert Record.save_calls == 1
    assert Record.load_calls == 1


def test_d1a_pre_spawn_binding_stops_at_monotonic_deadline() -> None:
    class Clock:
        value = 10.0
        sleeps: list[float] = []

        @classmethod
        def monotonic(cls) -> float:
            return cls.value

        @classmethod
        def sleep(cls, seconds: float) -> None:
            cls.sleeps.append(seconds)
            cls.value += seconds

    class Record:
        save_calls = 0
        load_calls = 0
        training_run_id = "d1a-matrix-aaaaaaaaaaaa"
        modal_app_id = "ap-exact"
        modal_app_url = "https://modal.com/apps/exact"
        function_call_id = ""
        metadata = {}

        def save(self) -> None:
            type(self).save_calls += 1

        @classmethod
        def from_id(cls, _run_id: str) -> Record:
            cls.load_calls += 1
            return cls()

    with pytest.raises(RuntimeError, match="before paid task spawn"):
        _persist_and_verify_d1a_pre_spawn_app_binding(
            Record(),
            polls=900,
            poll_seconds=1,
            sleep=Clock.sleep,
            monotonic=Clock.monotonic,
            deadline_monotonic=12.5,
        )

    assert Record.save_calls == 3
    assert Record.load_calls == 3
    assert Clock.sleeps == [1.0, 1.0, 0.5]
    assert Clock.value == 12.5


def test_d1a_pre_spawn_binding_rejects_readback_completed_after_deadline() -> None:
    class Clock:
        value = 20.0

        @classmethod
        def monotonic(cls) -> float:
            return cls.value

    class Record:
        training_run_id = "d1a-matrix-aaaaaaaaaaaa"
        modal_app_id = "ap-exact"
        modal_app_url = "https://modal.com/apps/exact"
        function_call_id = ""
        metadata = {
            "event_journal_enabled": True,
            "event_journal_contract": "d1a_legacy_single_attempt_v1",
        }

        def save(self) -> None:
            Clock.value = 21.0

        @classmethod
        def from_id(cls, _run_id: str) -> Record:
            return cls()

    with pytest.raises(RuntimeError, match="before paid task spawn"):
        _persist_and_verify_d1a_pre_spawn_app_binding(
            Record(),
            polls=1,
            poll_seconds=0,
            monotonic=Clock.monotonic,
            deadline_monotonic=21.0,
        )


def test_d1a_launch_shares_one_deadline_across_both_pre_spawn_writes() -> None:
    source = inspect.getsource(TrainConfig.launch)

    assert source.count("deadline_monotonic=d1a_pre_spawn_binding_deadline") == 2
    assert source.index("d1a_pre_spawn_binding_deadline =") < source.index(
        "_persist_and_verify_d1a_initial_run_record("
    )
    assert source.index("_persist_and_verify_d1a_initial_run_record(") < source.index(
        "_persist_and_verify_d1a_pre_spawn_app_binding("
    )


def test_d1a_pre_spawn_binding_rejects_nonempty_identity_conflict() -> None:
    class Record:
        training_run_id = "d1a-matrix-aaaaaaaaaaaa"
        modal_app_id = "ap-exact"
        modal_app_url = "https://modal.com/apps/exact"
        function_call_id = ""

        def save(self) -> None:
            return None

        @classmethod
        def from_id(cls, _run_id: str) -> Record:
            value = cls()
            value.modal_app_id = "ap-conflict"
            return value

    with pytest.raises(RuntimeError, match="authoritative pre-spawn app binding changed"):
        _persist_and_verify_d1a_pre_spawn_app_binding(
            Record(),
            polls=1,
            poll_seconds=0,
        )


def test_committed_overlay_contract_hashes_complete_declared_source_tree(
    tmp_path,
) -> None:
    source = tmp_path / "train"
    source.mkdir()
    runtime = source / "runtime.py"
    runtime.write_text("VALUE = 1\n")

    original = _image_overlay_contract(
        _overlay,
        [str(source)],
        required=True,
    )
    runtime.write_text("VALUE = 2\n")
    changed = _image_overlay_contract(
        _overlay,
        [str(source)],
        required=True,
    )

    assert original is not None
    assert changed is not None
    assert original["source_roots"][0]["sha256"] != changed["source_roots"][0]["sha256"]
    assert str(tmp_path) not in repr(original)


def test_overlay_source_receipt_is_independent_of_parent_checkout_path(
    tmp_path,
) -> None:
    first = tmp_path / "checkout-a" / "train"
    second = tmp_path / "checkout-b" / "train"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "runtime.py").write_text("VALUE = 1\n")
    (second / "runtime.py").write_text("VALUE = 1\n")

    first_receipt = _image_overlay_contract(
        _overlay,
        [str(first)],
        required=True,
    )
    second_receipt = _image_overlay_contract(
        _overlay,
        [str(second)],
        required=True,
    )

    assert first_receipt == second_receipt


def test_committed_overlay_requires_declared_source_inputs() -> None:
    with pytest.raises(ValueError, match="image_overlay_source_roots"):
        _image_overlay_contract(_overlay, [], required=True)


def test_repeated_unmerged_app_build_keeps_image_overlay_contract(
    monkeypatch,
    tmp_path,
) -> None:
    """An image preflight must not consume the paid launch's overlay."""

    source = tmp_path / "overlay-source"
    source.mkdir()
    (source / "runtime.py").write_text("VALUE = 1\n")
    recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=4,
        rollout_max_response_len=256,
        rollout_temperature=1.0,
        save_interval=1,
        image_overlay=_overlay,
        image_overlay_source_roots=[str(source)],
    )
    config = TrainConfig(
        model=Qwen3_4B(),
        dataset=HuggingFaceDataset(
            hf_repo="openai/gsm8k",
            input_column="question",
            output_column="answer",
        ),
        recipe=recipe,
        merge_model_recipe=False,
    )
    built_recipes = []

    def consume_overlay(**kwargs):
        resolved = kwargs["slime"]
        _image_overlay_contract(
            resolved.image_overlay,
            list(resolved.image_overlay_source_roots),
            required=False,
        )
        built_recipes.append(resolved)
        object.__setattr__(resolved, "image_overlay", None)
        return object()

    monkeypatch.setattr(
        "modal_training_gym.common.train.build_slime_app",
        consume_overlay,
    )

    config._build_app()  # image-only preflight
    config._build_app()  # launch app creation

    assert len(built_recipes) == 2
    assert built_recipes[0] is not built_recipes[1]
    assert all(resolved is not recipe for resolved in built_recipes)
    assert recipe.image_overlay is _overlay
    assert recipe.image_overlay_source_roots == [str(source)]


def test_committed_mode_rejects_dataset_rebuilds() -> None:
    recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=4,
        rollout_max_response_len=256,
        rollout_temperature=1.0,
        save_interval=1,
        attempt_mode="committed",
    )

    class _Dataset:
        always_prepare = True

    with pytest.raises(ValueError, match="always_prepare=False"):
        _validate_committed_dataset_inputs(recipe, _Dataset())  # type: ignore[arg-type]


def test_committed_attempt_mode_is_present_in_reporting_config() -> None:
    recipe = SlimeRecipe(
        gpu_type="H100",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=1,
        rollout_batch_size=4,
        rollout_max_response_len=256,
        rollout_temperature=1.0,
        save_interval=1,
        attempt_mode="committed",
    )

    assert _serialize_slime_params(recipe)["attempt_mode"] == "committed"


def test_train_function_timeout_is_recipe_controlled_and_removed() -> None:
    kwargs = {"timeout": 10_200, "ephemeral_disk": 123}

    assert _pop_train_function_timeout(kwargs) == 10_200
    assert kwargs == {"ephemeral_disk": 123}


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "10200", None])
def test_train_function_timeout_rejects_invalid_values(value: object) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        _pop_train_function_timeout({"timeout": value})


def test_train_function_timeout_preserves_extended_horizon_default() -> None:
    assert _pop_train_function_timeout({}) == 48 * 60 * 60


def test_d1a_remote_record_joins_late_function_call_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRIFT_ASYNC_RL_D1_MATRIX", "1")

    class Record:
        latest: Record

        def __init__(self, *, function_call_id: str) -> None:
            self.training_run_id = "d1a-matrix-aaaaaaaaaaaa"
            self.modal_app_id = "ap-exact"
            self.function_call_id = function_call_id

        @classmethod
        async def from_id(cls, _run_id: str, *, is_async: bool) -> Record:
            assert is_async is True
            return cls.latest

    remote = Record(function_call_id="")
    Record.latest = remote
    stale = Record(function_call_id="")

    async def publish_binding() -> None:
        await asyncio.sleep(0)
        remote.function_call_id = "fc-exact"

    async def exercise() -> Record:
        publisher = asyncio.create_task(publish_binding())
        result = await _require_d1a_function_call_binding(
            stale,
            polls=3,
            poll_seconds=0,
        )
        await publisher
        return result

    result = asyncio.run(exercise())
    assert result is stale
    assert stale.function_call_id == "fc-exact"


def test_d1a_remote_record_retries_transient_metadata_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRIFT_ASYNC_RL_D1_MATRIX", "1")

    class Record:
        load_calls = 0

        def __init__(self, *, function_call_id: str) -> None:
            self.training_run_id = "d1a-matrix-aaaaaaaaaaaa"
            self.modal_app_id = "ap-exact"
            self.function_call_id = function_call_id

        @classmethod
        async def from_id(cls, _run_id: str, *, is_async: bool) -> Record:
            assert is_async is True
            cls.load_calls += 1
            if cls.load_calls == 1:
                raise RuntimeError("transient metadata load")
            return cls(function_call_id="fc-exact")

    stale = Record(function_call_id="")
    result = asyncio.run(
        _require_d1a_function_call_binding(
            stale,
            polls=2,
            poll_seconds=0,
        )
    )
    assert result is stale
    assert result.function_call_id == "fc-exact"


def test_d1a_remote_record_rejects_missing_function_call_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRIFT_ASYNC_RL_D1_MATRIX", "1")

    class Record:
        training_run_id = "d1a-matrix-aaaaaaaaaaaa"
        modal_app_id = "ap-exact"
        function_call_id = ""

        @classmethod
        async def from_id(cls, _run_id: str, *, is_async: bool) -> Record:
            assert is_async is True
            return cls()

    with pytest.raises(RuntimeError, match="not durably joined"):
        asyncio.run(
            _require_d1a_function_call_binding(
                Record(),
                polls=1,
                poll_seconds=0,
            )
        )


def test_d1a_terminal_success_retries_metadata_only_and_reads_back() -> None:
    attempt_id = "a" * 32
    run_id = "d1a-matrix-aaaaaaaaaaaa"

    class Record:
        save_calls = 0
        include_journal_contract = True

        def __init__(self) -> None:
            self.training_run_id = run_id
            self.modal_app_id = "ap-exact"
            self.function_call_id = "fc-exact"
            self.status = type("Status", (), {"value": "completed"})()
            self.metadata = {
                "attempt_mode": "legacy",
                "event_journal_enabled": True,
                "event_journal_contract": "d1a_legacy_single_attempt_v1",
                "attempt_count": 1,
                "active_attempt_id": attempt_id,
                "logical_save_root": f"/checkpoints/{run_id}",
                "active_attempt_root": f"/checkpoints/{run_id}",
                "attempts": [
                    {
                        "attempt": 1,
                        "attempt_id": attempt_id,
                        "attempt_root": f"/checkpoints/{run_id}",
                        "status": "completed",
                    }
                ],
            }
            if not type(self).include_journal_contract:
                self.metadata.pop("event_journal_enabled")
                self.metadata.pop("event_journal_contract")

        async def save(self, *, is_async: bool) -> None:
            assert is_async is True
            type(self).save_calls += 1
            if type(self).save_calls == 1:
                raise RuntimeError("transient metadata write")

        @classmethod
        async def from_id(cls, observed_run_id: str, *, is_async: bool) -> Record:
            assert observed_run_id == run_id
            assert is_async is True
            return cls()

    result = asyncio.run(
        _persist_and_verify_d1a_terminal_success(
            Record(),
            expected_attempt_id=attempt_id,
            attempts=2,
            retry_seconds=0,
        )
    )
    assert result.metadata["attempts"][0]["status"] == "completed"
    assert Record.save_calls == 2

    Record.include_journal_contract = False
    with pytest.raises(RuntimeError, match="not durably persisted"):
        asyncio.run(
            _persist_and_verify_d1a_terminal_success(
                Record(),
                expected_attempt_id=attempt_id,
                attempts=1,
                retry_seconds=0,
            )
        )


def test_d1a_terminal_success_rejects_mismatched_authoritative_attempt() -> None:
    attempt_id = "a" * 32
    run_id = "d1a-matrix-aaaaaaaaaaaa"

    class Record:
        training_run_id = run_id
        modal_app_id = "ap-exact"
        function_call_id = "fc-exact"
        status = type("Status", (), {"value": "completed"})()
        metadata: dict = {}

        async def save(self, *, is_async: bool) -> None:
            assert is_async is True

        @classmethod
        async def from_id(cls, _run_id: str, *, is_async: bool) -> Record:
            assert is_async is True
            value = cls()
            value.metadata = {
                "attempt_mode": "legacy",
                "event_journal_enabled": True,
                "event_journal_contract": "d1a_legacy_single_attempt_v1",
                "attempt_count": 1,
                "active_attempt_id": "b" * 32,
                "logical_save_root": f"/checkpoints/{run_id}",
                "active_attempt_root": f"/checkpoints/{run_id}",
                "attempts": [],
            }
            return value

    with pytest.raises(RuntimeError, match="not durably persisted"):
        asyncio.run(
            _persist_and_verify_d1a_terminal_success(
                Record(),
                expected_attempt_id=attempt_id,
                attempts=1,
                retry_seconds=0,
            )
        )


def test_d1a_terminal_binding_restores_stale_blank_cache_fields() -> None:
    authoritative = type(
        "Record",
        (),
        {
            "modal_app_id": "ap-exact",
            "modal_app_url": "https://modal.com/apps/ap-exact",
            "function_call_id": "fc-exact",
        },
    )()
    stale = type(
        "Record",
        (),
        {"modal_app_id": "", "modal_app_url": "", "function_call_id": ""},
    )()

    assert _restore_d1a_terminal_binding(authoritative, stale) is stale
    assert stale.modal_app_id == "ap-exact"
    assert stale.modal_app_url == "https://modal.com/apps/ap-exact"
    assert stale.function_call_id == "fc-exact"


def test_d1a_terminal_binding_rejects_conflicting_nonempty_cache() -> None:
    authoritative = type(
        "Record",
        (),
        {"modal_app_id": "ap-exact", "modal_app_url": "", "function_call_id": "fc-exact"},
    )()
    stale = type(
        "Record",
        (),
        {"modal_app_id": "ap-other", "modal_app_url": "", "function_call_id": ""},
    )()

    with pytest.raises(RuntimeError, match="conflicts"):
        _restore_d1a_terminal_binding(authoritative, stale)
