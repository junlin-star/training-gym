import asyncio
import inspect
import json
import os
import signal
import sys
import types
from types import SimpleNamespace

import pytest

import modal_training_gym.frameworks.slime.megagem_stage_c_rollout as stage_c_rollout
from modal_training_gym import (
    MegaGem_Qwen3_4B_SFT,
    MegaGem_Qwen3_4B_StageC_Recipe,
    MegaGemStageCDataset,
)
from modal_training_gym.common.train import _merge_recipe
from modal_training_gym.frameworks.slime.modal_helpers.utils import build_train_cmd
from modal_training_gym.train_recipes.slime_recipe.megagem_qwen3_4b_stage_c import (
    megagem_stage_c_summary,
)
from modal_training_gym.frameworks.slime.megagem_stage_c_rollout import (
    MEGAGEM_STAGE_C_ADVANTAGE_PATH,
    MEGAGEM_STAGE_C_ROLLOUT_LOG_PATH,
    MEGAGEM_STAGE_C_REWARD_PATH,
    MEGAGEM_STAGE_C_ROLLOUT_CONTRACT,
    megagem_precomputed_advantages,
    megagem_precomputed_reward,
    megagem_stage_c_rollout,
)
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


def test_megagem_sft_model_keeps_private_checkpoint_identity() -> None:
    model = MegaGem_Qwen3_4B_SFT()

    assert model.model_name == "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2"
    assert model.model_name != "Qwen/Qwen3-4B"
    assert model.architecture.num_layers == 36
    assert model.architecture.hidden_size == 2560
    assert model.architecture.num_attention_heads == 32
    assert model.architecture.num_query_groups == 8
    assert model.architecture.rotary_base == 5000000


def test_get_base_recipe_selects_stage_c_for_megagem_sft() -> None:
    recipe = SlimeRecipe.get_base_recipe(MegaGem_Qwen3_4B_SFT())

    assert isinstance(recipe, MegaGem_Qwen3_4B_StageC_Recipe)


def test_stage_c_uses_one_train_node_and_four_rollout_nodes() -> None:
    recipe = MegaGem_Qwen3_4B_StageC_Recipe()
    summary = megagem_stage_c_summary(recipe)

    assert summary["total_gpus"] == 40
    assert summary["total_nodes"] == 5
    assert summary["actor_nodes"] == 1
    assert summary["actor_gpus"] == 8
    assert summary["actor_data_parallel"] == 8
    assert summary["rollout_gpus"] == 32
    assert summary["rollout_engines"] == 32
    assert summary["rollout_num_gpus_per_engine"] == 1
    assert summary["tp"] == 1
    assert summary["pp"] == 1
    assert summary["cp"] == 1


def test_stage_c_large_rollout_shape_matches_32_rollout_engines() -> None:
    recipe = MegaGem_Qwen3_4B_StageC_Recipe()
    summary = megagem_stage_c_summary(recipe)

    assert summary["rollout_batch_size"] == 128
    assert summary["n_samples_per_prompt"] == 16
    assert summary["games_per_rollout"] == 2048
    assert summary["global_batch_size"] == 2048
    assert summary["updates_per_rollout"] == 1
    assert recipe.rollout_max_response_len == 1024
    assert recipe.eval_max_response_len == 1024
    assert recipe.sglang_max_running_requests == 1024


def test_stage_c_stability_defaults_anchor_drift() -> None:
    recipe = MegaGem_Qwen3_4B_StageC_Recipe()
    summary = megagem_stage_c_summary(recipe)

    assert recipe.lr == 6e-6
    assert recipe.min_lr == 6e-7
    assert recipe.kl_loss_coef == 0.05
    assert recipe.rollout_temperature == 0.85
    assert recipe.rollout_top_p == 1.0
    assert summary["lr"] == 6e-6
    assert summary["min_lr"] == 6e-7
    assert summary["kl_loss_coef"] == 0.05
    assert summary["rollout_temperature"] == 0.85
    assert summary["rollout_top_p"] == 1.0


def test_stage_c_rejects_top_p_without_faithful_candidate_sets() -> None:
    with pytest.raises(ValueError, match="rollout_top_p=1.0"):
        MegaGem_Qwen3_4B_StageC_Recipe(rollout_top_p=0.95)

    with pytest.raises(RuntimeError, match="rollout_top_p=1.0"):
        stage_c_rollout._validate_rollout_probability_contract(
            SimpleNamespace(rollout_top_p=1.0), {"top_p": 0.95}
        )

    stage_c_rollout._validate_rollout_probability_contract(
        SimpleNamespace(rollout_top_p=1.0), {"temperature": 0.85, "top_k": 0}
    )
    stage_c_rollout._validate_rollout_probability_contract(
        SimpleNamespace(rollout_top_p=1.0), {"min_p": "0", "top_k": "-1"}
    )

    with pytest.raises(RuntimeError, match="non-numeric rollout top_p"):
        stage_c_rollout._validate_rollout_probability_contract(
            SimpleNamespace(rollout_top_p="not-a-float"), {"top_p": 1.0}
        )

    with pytest.raises(RuntimeError, match="top_k"):
        stage_c_rollout._validate_rollout_probability_contract(
            SimpleNamespace(rollout_top_p=1.0), {"top_k": 40}
        )
    with pytest.raises(RuntimeError, match="non-integer top_k"):
        stage_c_rollout._validate_rollout_probability_contract(
            SimpleNamespace(rollout_top_p=1.0), {"top_k": "bad"}
        )

    with pytest.raises(RuntimeError, match="min_p"):
        stage_c_rollout._validate_rollout_probability_contract(
            SimpleNamespace(rollout_top_p=1.0), {"min_p": 0.05}
        )
    with pytest.raises(RuntimeError, match="non-numeric min_p"):
        stage_c_rollout._validate_rollout_probability_contract(
            SimpleNamespace(rollout_top_p=1.0), {"min_p": "bad"}
        )


def test_stage_c_uses_current_phase3_reward_env_defaults() -> None:
    recipe = MegaGem_Qwen3_4B_StageC_Recipe()

    assert recipe.app_tags == {"megagem": "stage-c"}
    assert recipe.environment["PHASE3_REWARD_WIN_BONUS"] == ""
    assert recipe.environment["PHASE3_SHAPING_LAMBDA"] == ""
    assert recipe.environment["PHASE3_ILLEGAL_PENALTY"] == ""
    assert recipe.environment["PHASE3_CORRECTION_SCALE"] == ""
    assert recipe.environment["PHASE3_TERMINAL_CORRECTION"] == "1"
    assert recipe.environment["PHASE3_TERMINAL_SHAPE"] == "tanh"
    assert recipe.environment["PHASE3_REWARD_SCALE"] == "19.6"
    assert recipe.environment["PHASE3_REVEAL_SHAPING_WEIGHT"] == "0.0"


def test_stage_c_dataset_materializes_seed_group_labels(tmp_path) -> None:
    dataset = MegaGemStageCDataset(
        num_prompts=3,
        seed_start=42,
        k=16,
        rows_per_group=16,
        seed_stride=3,
    )
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"

    dataset.prepare(str(train_path), {"eval": str(eval_path)})
    dataset.validate_prepared(str(train_path))
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]

    assert len(rows) == 3
    assert rows[0]["prompt"].startswith("[megagem-stage-c]")
    label = json.loads(rows[0]["label"])
    assert label == {
        "k": 16,
        "num_players": 3,
        "opponent_actor_id": "current_self",
        "rows_per_group": 16,
        "seed": 42,
        "seed_stride": 3,
        "trainable_seat": 0,
        "value_chart": "A",
    }
    assert [json.loads(row["label"])["trainable_seat"] for row in rows] == [0, 1, 2]
    assert eval_path.exists()


def test_stage_c_dataset_can_pin_one_trainable_seat() -> None:
    dataset = MegaGemStageCDataset(
        num_prompts=4,
        seed_start=42,
        trainable_seat=1,
        rotate_seats=False,
    )

    seats = [json.loads(row["label"])["trainable_seat"] for row in dataset.load()]

    assert seats == [1, 1, 1, 1]


def test_stage_c_rolls_k_group_games_concurrently() -> None:
    source = inspect.getsource(stage_c_rollout._roll_group_rows)

    assert "_collect_k_successes" in source
    assert "asyncio.create_task" in inspect.getsource(stage_c_rollout._collect_k_successes)


def test_stage_c_sglang_sampling_params_drop_chat_template_kwargs() -> None:
    params = stage_c_rollout._sampling_params(
        {"temperature": 0.7},
        {
            "top_p": 0.9,
            "max_completion_tokens": 123,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )

    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.9
    assert params["max_new_tokens"] == 123
    assert "chat_template_kwargs" not in params
    assert "extra_body" not in params


def test_stage_c_custom_rows_do_not_fake_slime_probability_fields() -> None:
    sample = SimpleNamespace()

    stage_c_rollout._fill_rollout_compat_fields(sample, [10, 20, 30])

    assert not hasattr(sample, "rollout_log_probs")
    assert not hasattr(sample, "rollout_top_p_token_ids")
    assert not hasattr(sample, "rollout_top_p_token_offsets")
    assert not hasattr(sample, "rollout_top_p_log_probs")
    assert not hasattr(sample, "rollout_top_p_probs")


def test_stage_c_signal_shim_noops_signal_setup_in_worker_threads() -> None:
    original_signal = signal.signal
    original_alarm = getattr(signal, "alarm", None)
    original_setitimer = getattr(signal, "setitimer", None)
    stage_c_rollout._SIGNAL_SHIM_INSTALLED = False

    try:
        stage_c_rollout._install_non_main_thread_signal_shim(force=True)

        previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        assert previous == signal.SIG_DFL
        assert signal.alarm(10) == 0
        assert signal.setitimer(signal.ITIMER_REAL, 1.0) == (0.0, 0.0)
    finally:
        signal.signal = original_signal
        if original_alarm is not None:
            signal.alarm = original_alarm
        if original_setitimer is not None:
            signal.setitimer = original_setitimer
        stage_c_rollout._SIGNAL_SHIM_INSTALLED = False


def test_stage_c_balanced_select_refills_short_valid_row_groups() -> None:
    rows = [
        {"group_key": "g", "completion": f"c{i}", "stage_c": {"row_slot": i}}
        for i in range(3)
    ]

    selected = stage_c_rollout._balanced_select(rows, 5, seed=0)

    assert len(selected) == 5
    assert {row["completion"] for row in selected} == {"c0", "c1", "c2"}
    selected[0]["stage_c"]["row_slot"] = 999
    assert rows[0]["stage_c"]["row_slot"] == 0


def test_stage_c_recipe_exposes_kgroup_resilience_defaults() -> None:
    recipe = MegaGem_Qwen3_4B_StageC_Recipe()
    summary = megagem_stage_c_summary(recipe)

    assert recipe.extra_config["megagem_extra_games_per_group"] == 4
    assert recipe.extra_config["megagem_min_success_games"] == 12
    assert recipe.extra_config["megagem_game_timeout_s"] == 600
    assert recipe.extra_config["megagem_fail_open_groups"] is False
    assert recipe.extra_config["megagem_length_penalty_target_tokens"] == 512
    assert recipe.extra_config["megagem_length_penalty_per_100_tokens"] == 0.01
    assert recipe.extra_config["megagem_length_penalty_cap"] == 0.20
    assert (
        recipe.extra_config["training_gym_custom_rollout_log_function_path"]
        == MEGAGEM_STAGE_C_ROLLOUT_LOG_PATH
    )
    assert summary["megagem_extra_games_per_group"] == 4
    assert summary["megagem_min_success_games"] == 12
    assert summary["megagem_game_timeout_s"] == 600
    assert summary["megagem_fail_open_groups"] is False
    assert summary["megagem_length_penalty_target_tokens"] == 512
    assert summary["megagem_length_penalty_per_100_tokens"] == 0.01
    assert summary["megagem_length_penalty_cap"] == 0.20
    assert summary["custom_rollout_log_function_path"] == MEGAGEM_STAGE_C_ROLLOUT_LOG_PATH


def test_stage_c_game_timeout_can_be_configured(monkeypatch) -> None:
    args = SimpleNamespace(megagem_game_timeout_s=123.5)

    assert stage_c_rollout._game_timeout_s(args) == 123.5

    monkeypatch.setenv("MEGAGEM_STAGE_C_GAME_TIMEOUT_S", "77")
    assert stage_c_rollout._game_timeout_s(args) == 77.0

    monkeypatch.setenv("MEGAGEM_STAGE_C_GAME_TIMEOUT_S", "bad")
    assert stage_c_rollout._game_timeout_s(args) == 600.0


def test_stage_c_label_falls_back_to_synthetic_prompt() -> None:
    sample = SimpleNamespace(
        prompt="[megagem-stage-c] seed=42 chart=B seat=2 k=8 stride=5 rows=7"
    )

    assert stage_c_rollout._json_label(sample) == {
        "seed": 42,
        "seed_stride": 5,
        "value_chart": "B",
        "trainable_seat": 2,
        "num_players": 3,
        "k": 8,
        "rows_per_group": 7,
        "opponent_actor_id": "current_self",
    }


def _install_fake_megagem_row_modules(monkeypatch) -> None:
    monkeypatch.setenv("MEGAGEM_BENCH_ROOT", "/home/ec2-user/MegagemBench")
    fake_harness = types.ModuleType("_toy_grpo_harness")

    def flatten_training_rows(games, reward_cfg=None, *, trainable_seat=0):
        del reward_cfg
        rows = []
        for game in games:
            rows.append(
                {
                    "prompt": f"prompt-{game['k_index']}",
                    "completion": f"completion-{game['k_index']}",
                    "precomputed_reward": 0.0,
                    "precomputed_advantage": float(game["k_index"]),
                    "group_key": f"seat-{trainable_seat}",
                    "player_id": trainable_seat,
                }
            )
        return rows

    fake_harness.flatten_training_rows = flatten_training_rows
    fake_export = types.ModuleType("src.rl.export")
    fake_export.contract_check = lambda rows: len(rows)
    monkeypatch.setitem(sys.modules, "_toy_grpo_harness", fake_harness)
    monkeypatch.setitem(sys.modules, "src.rl.export", fake_export)


def test_stage_c_roll_group_replaces_failed_games(monkeypatch) -> None:
    stage_c_rollout._reset_stage_c_runtime_state()
    _install_fake_megagem_row_modules(monkeypatch)
    seen: list[int] = []

    async def fake_roll_schema_game(args, *, seed, label, k_index, sampling_params):
        del args, seed, label, sampling_params
        seen.append(k_index)
        if k_index == 0:
            raise RuntimeError("synthetic failed game")
        return {"k_index": k_index}

    monkeypatch.setattr(stage_c_rollout, "_roll_schema_game", fake_roll_schema_game)
    label = {
        "seed": 42,
        "seed_stride": 128,
        "value_chart": "A",
        "trainable_seat": 0,
        "num_players": 3,
        "k": 4,
        "rows_per_group": 4,
    }
    args = SimpleNamespace(
        megagem_extra_games_per_group=1,
        megagem_min_success_games=3,
        megagem_max_parallel_games=4,
    )

    rows = asyncio.run(stage_c_rollout._roll_group_rows(args, label, 0, {}))

    assert sorted(seen) == [0, 1, 2, 3, 4]
    assert len(rows) == 4
    assert {r["stage_c"]["k_success"] for r in rows} == {4}
    assert {r["stage_c"]["k_failures"] for r in rows} == {1}
    stage_c_rollout._reset_stage_c_runtime_state()


def test_stage_c_roll_group_fails_when_too_few_games_survive(monkeypatch) -> None:
    stage_c_rollout._reset_stage_c_runtime_state()
    _install_fake_megagem_row_modules(monkeypatch)

    async def fake_roll_schema_game(args, *, seed, label, k_index, sampling_params):
        del args, seed, label, k_index, sampling_params
        raise RuntimeError("synthetic outage")

    monkeypatch.setattr(stage_c_rollout, "_roll_schema_game", fake_roll_schema_game)
    label = {
        "seed": 42,
        "seed_stride": 128,
        "value_chart": "A",
        "trainable_seat": 0,
        "num_players": 3,
        "k": 4,
        "rows_per_group": 4,
    }
    args = SimpleNamespace(
        megagem_extra_games_per_group=1,
        megagem_min_success_games=3,
        megagem_max_parallel_games=4,
    )

    try:
        asyncio.run(stage_c_rollout._roll_group_rows(args, label, 0, {}))
    except RuntimeError as exc:
        assert "too few successful games" in str(exc)
        assert "successes=0/4" in str(exc)
    else:
        raise AssertionError("expected too-few-successful-games failure")
    stage_c_rollout._reset_stage_c_runtime_state()


def test_stage_c_assigns_slots_without_trusting_sample_indices(monkeypatch) -> None:
    stage_c_rollout._reset_stage_c_runtime_state()
    calls: list[int] = []
    label = {
        "seed": 42,
        "seed_stride": 128,
        "value_chart": "A",
        "trainable_seat": 0,
        "num_players": 3,
        "k": 4,
        "rows_per_group": 4,
    }

    async def fake_roll_group_rows(args, label, generation, sampling_params):
        del args, label, sampling_params
        calls.append(generation)
        return [
            {
                "prompt": f"prompt-{i}",
                "completion": f"completion-{i}",
                "precomputed_reward": 0.0,
                "precomputed_advantage": float(i),
                "group_key": "g",
            }
            for i in range(4)
        ]

    monkeypatch.setattr(stage_c_rollout, "_roll_group_rows", fake_roll_group_rows)

    async def run_once():
        slots = []
        for _ in range(4):
            sample = SimpleNamespace(
                label=json.dumps(label),
                sample_index=999,
                rollout_idx=999,
                metadata={"sample_index": 999},
            )
            rows, slot, generation = await stage_c_rollout._rows_for_sample(
                SimpleNamespace(), sample, label, {}
            )
            slots.append((slot, generation, rows[slot]["completion"]))
        return slots

    assert asyncio.run(run_once()) == [
        (0, 0, "completion-0"),
        (1, 0, "completion-1"),
        (2, 0, "completion-2"),
        (3, 0, "completion-3"),
    ]
    assert calls == [0]
    stage_c_rollout._reset_stage_c_runtime_state()


def test_stage_c_failed_group_task_can_fail_open_with_zero_advantage_rows(monkeypatch) -> None:
    stage_c_rollout._reset_stage_c_runtime_state()
    label = {
        "seed": 42,
        "seed_stride": 128,
        "value_chart": "A",
        "trainable_seat": 0,
        "num_players": 3,
        "k": 4,
        "rows_per_group": 4,
    }

    async def fail_roll_group_rows(args, label, generation, sampling_params):
        del args, label, generation, sampling_params
        raise ValueError("boom")

    monkeypatch.setattr(stage_c_rollout, "_roll_group_rows", fail_roll_group_rows)

    async def run_failure():
        sample = SimpleNamespace(label=json.dumps(label))
        rows, slot, generation = await stage_c_rollout._rows_for_sample(
            SimpleNamespace(megagem_fail_open_groups=True), sample, label, {}
        )
        return rows, slot, generation

    rows, slot, generation = asyncio.run(run_failure())
    assert slot == 0
    assert generation == 0
    assert len(rows) == 4
    assert all(r["precomputed_advantage"] == 0.0 for r in rows)
    assert all(r["stage_c"]["fallback"] is True for r in rows)
    state = stage_c_rollout._GROUP_STATES[stage_c_rollout._group_key(label)]
    assert state.task is None
    assert state.rows is None
    assert state.assigned_slots == set()
    stage_c_rollout._reset_stage_c_runtime_state()


def test_stage_c_failed_group_task_can_fail_closed(monkeypatch) -> None:
    stage_c_rollout._reset_stage_c_runtime_state()
    label = {
        "seed": 42,
        "seed_stride": 128,
        "value_chart": "A",
        "trainable_seat": 0,
        "num_players": 3,
        "k": 4,
        "rows_per_group": 4,
    }

    async def fail_roll_group_rows(args, label, generation, sampling_params):
        del args, label, generation, sampling_params
        raise ValueError("boom")

    monkeypatch.setattr(stage_c_rollout, "_roll_group_rows", fail_roll_group_rows)

    async def run_failure():
        sample = SimpleNamespace(label=json.dumps(label))
        try:
            await stage_c_rollout._rows_for_sample(
                SimpleNamespace(), sample, label, {}
            )
        except RuntimeError as exc:
            assert "group task failed" in str(exc)
            assert "cause_chain=ValueError: boom" in str(exc)
        else:
            raise AssertionError("expected group task failure")

    asyncio.run(run_failure())
    stage_c_rollout._reset_stage_c_runtime_state()


def test_merge_preserves_stage_c_recipe_for_sft_model() -> None:
    model = MegaGem_Qwen3_4B_SFT()
    user_recipe = MegaGem_Qwen3_4B_StageC_Recipe(num_rollout=5)
    merged = _merge_recipe(SlimeRecipe.get_base_recipe(model), user_recipe)

    assert isinstance(merged, MegaGem_Qwen3_4B_StageC_Recipe)
    assert merged.num_rollout == 5
    assert merged.rollout_num_gpus == 32


def test_train_command_uses_async_slime_private_sft_and_megagem_hooks() -> None:
    model = MegaGem_Qwen3_4B_SFT()
    recipe = MegaGem_Qwen3_4B_StageC_Recipe()
    dataset = MegaGemStageCDataset()
    cmd = build_train_cmd(recipe, "/root/slime", model=model, dataset=dataset)

    assert "train_async.py" in cmd
    assert "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2" in cmd
    assert "Qwen/Qwen3-4B" not in cmd
    assert "--rollout-num-gpus 32" in cmd
    assert "--rollout-num-gpus-per-engine 1" in cmd
    assert "--rollout-top-p 1.0" in cmd
    assert "--rollout-temperature 0.85" in cmd
    assert "frameworks.slime.megagem_stage_c_rollout.megagem_stage_c_rollout" in cmd
    assert "custom_generate_function_path" in cmd
    assert "--rollout-function-path" not in cmd
    assert "train_recipes.slime_recipe.megagem_stage_c_rollout" not in cmd
    assert "megagem_rollout_not_migrated" not in cmd
    assert "--custom-config-path" in cmd


def test_stage_c_rollout_and_advantage_contract_paths_are_live() -> None:
    recipe = MegaGem_Qwen3_4B_StageC_Recipe()

    assert recipe.custom_generate_function is megagem_stage_c_rollout
    assert recipe.rollout_function is None
    assert recipe.app_tags == {"megagem": "stage-c"}
    assert (
        recipe.extra_config["custom_generate_function_path"]
        == (
            "modal_training_gym.frameworks.slime.megagem_stage_c_rollout."
            "megagem_stage_c_rollout"
        )
    )
    assert recipe.extra_config["custom_rm_path"] == MEGAGEM_STAGE_C_REWARD_PATH
    assert (
        recipe.extra_config["custom_advantage_function_path"]
        == MEGAGEM_STAGE_C_ADVANTAGE_PATH
    )
    assert recipe.extra_config["megagem_max_parallel_games"] == 256
    assert recipe.extra_config["megagem_fail_open_groups"] is False
    assert callable(megagem_precomputed_reward)
    assert callable(megagem_precomputed_advantages)


def test_summary_is_json_serializable() -> None:
    payload = megagem_stage_c_summary(MegaGem_Qwen3_4B_StageC_Recipe())

    decoded = json.loads(json.dumps(payload))
    assert decoded["rollout_contract"] == MEGAGEM_STAGE_C_ROLLOUT_CONTRACT
    assert decoded["custom_rm_path"] == MEGAGEM_STAGE_C_REWARD_PATH
    assert decoded["custom_advantage_function_path"] == MEGAGEM_STAGE_C_ADVANTAGE_PATH
    assert decoded["megagem_max_parallel_games"] == 256
    assert decoded["megagem_fail_open_groups"] is False
    assert decoded["lr"] == 6e-6
    assert decoded["kl_loss_coef"] == 0.05
    assert decoded["custom_rollout_log_function_path"] == MEGAGEM_STAGE_C_ROLLOUT_LOG_PATH


def test_image_overlay_source_path_exists_for_devbox() -> None:
    # The overlay itself is exercised only at Modal image-build time, but the
    # default checkout path should be valid on the devbox where Stage C launches.
    source = os.environ.get("MEGAGEM_BENCH_SOURCE", "/home/ec2-user/MegagemBench")
    assert os.path.isdir(source)
