from __future__ import annotations

import os

from modal_training_gym.common.launcher_utils import redact_runtime_env
from modal_training_gym.common.wandb import (
    WandbConfig,
    build_wandb_runtime_env,
    install_wandb_api_key_in_process,
)
from modal_training_gym.frameworks.miles.modal_helpers.utils import (
    build_train_cmd as build_miles_train_cmd,
)
from modal_training_gym.frameworks.slime.modal_helpers.utils import (
    build_train_cmd as build_slime_train_cmd,
)
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


_RAW_SECRET = "test-only-wandb-secret-value"
_SLIME_KW = {
    "gpu_type": "H100",
    "colocate": True,
    "tensor_model_parallel_size": 1,
    "sequence_parallel": False,
    "rollout_num_gpus_per_engine": 1,
    "num_rollout": 1,
    "rollout_batch_size": 4,
    "rollout_max_response_len": 256,
    "rollout_temperature": 1.0,
    "save_interval": 1,
}


def test_slime_wandb_secret_is_ambient_not_ray_job_metadata(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    wandb = WandbConfig(project="secret-safe", key=_RAW_SECRET)
    recipe = SlimeRecipe(**_SLIME_KW, wandb=wandb)

    command = build_slime_train_cmd(recipe, "/root/slime")
    assert install_wandb_api_key_in_process(recipe.wandb)
    submitted_env = {
        "env_vars": build_wandb_runtime_env(
            recipe.wandb,
            run_id="attempt-1",
            entity="research",
        )
    }
    logged = f"Command: {command}, runtime_env: {redact_runtime_env(submitted_env)}"

    assert "--wandb-key" not in command
    assert _RAW_SECRET not in command
    assert "WANDB_API_KEY" not in submitted_env["env_vars"]
    assert os.environ["WANDB_API_KEY"] == _RAW_SECRET
    assert _RAW_SECRET not in logged
    assert recipe.wandb is wandb
    assert recipe.wandb.key == _RAW_SECRET


def test_miles_wandb_secret_is_ambient_not_ray_job_metadata(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    wandb = WandbConfig(project="secret-safe", key=_RAW_SECRET)
    recipe = MilesConfig(wandb=wandb)

    command = build_miles_train_cmd(recipe, "/root/miles")
    assert install_wandb_api_key_in_process(recipe.wandb)
    submitted_env = {"env_vars": build_wandb_runtime_env(recipe.wandb)}
    logged = f"Command: {command}, runtime_env: {redact_runtime_env(submitted_env)}"

    assert "--wandb-key" not in command
    assert _RAW_SECRET not in command
    assert "WANDB_API_KEY" not in submitted_env["env_vars"]
    assert os.environ["WANDB_API_KEY"] == _RAW_SECRET
    assert _RAW_SECRET not in logged
    assert recipe.wandb is wandb
    assert recipe.wandb.key == _RAW_SECRET


def test_modal_secret_overrides_config_without_mutating_recipe(monkeypatch) -> None:
    env_secret = "test-only-modal-secret-value"
    config_secret = "test-only-config-fallback"
    monkeypatch.setenv("WANDB_API_KEY", env_secret)
    wandb = WandbConfig(project="secret-safe", key=config_secret)
    recipe = SlimeRecipe(**_SLIME_KW, wandb=wandb)

    command = build_slime_train_cmd(recipe, "/root/slime")
    assert install_wandb_api_key_in_process(recipe.wandb)
    submitted = build_wandb_runtime_env(recipe.wandb)
    logged = repr(redact_runtime_env({"env_vars": submitted}))

    assert "WANDB_API_KEY" not in submitted
    assert os.environ["WANDB_API_KEY"] == env_secret
    assert env_secret not in command
    assert config_secret not in command
    assert env_secret not in logged
    assert recipe.wandb.key == config_secret


def test_runtime_env_diagnostics_redact_status_and_user_secrets() -> None:
    raw = {
        "env_vars": {
            "TRAINING_GYM_FRAMEWORK_STATUS_TOKEN": "status-token-value",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-value",
            "NORMAL_SETTING": "visible",
            "STOP_TOKEN_ID": "151645",
        }
    }

    redacted = redact_runtime_env(raw)

    assert redacted["env_vars"]["TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"] == "[redacted]"
    assert redacted["env_vars"]["AWS_SECRET_ACCESS_KEY"] == "[redacted]"
    assert redacted["env_vars"]["NORMAL_SETTING"] == "visible"
    assert redacted["env_vars"]["STOP_TOKEN_ID"] == "151645"
    assert raw["env_vars"]["AWS_SECRET_ACCESS_KEY"] == "aws-secret-value"
