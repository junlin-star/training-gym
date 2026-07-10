"""Local checks for DeploymentConfig(unauthenticated=...)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from modal_training_gym.common.deployment import DeploymentConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models.base import ModelConfig
from modal_training_gym.deploy_recipes.sglang_recipe.serve_sglang import (
    build_sglang_serve_app,
)
from modal_training_gym.deploy_recipes.vllm_recipe import VllmRecipe
from modal_training_gym.deploy_recipes.vllm_recipe.serve_vllm import (
    build_vllm_serve_app,
)


def test_sglang_builder_accepts_unauthenticated() -> None:
    assert "unauthenticated" in inspect.signature(build_sglang_serve_app).parameters


def test_vllm_builder_rejects_unauthenticated_param() -> None:
    assert "unauthenticated" not in inspect.signature(build_vllm_serve_app).parameters


def test_default_unauthenticated_is_false() -> None:
    cfg = DeploymentConfig(model=ModelConfig(model_name="test/model"))
    assert cfg.unauthenticated is False


def test_sglang_serve_forwards_unauthenticated() -> None:
    captured: dict = {}

    fake_app = MagicMock()
    fake_app.app_id = "ap-test"
    fake_endpoint = MagicMock()
    fake_endpoint.get_url = MagicMock(return_value="https://example.modal.run")
    fake_app.SGLangEndpoint = fake_endpoint

    def _capture_build(**kwargs):
        captured.update(kwargs)
        return fake_app

    cfg = DeploymentConfig(
        model=ModelConfig(model_name="test/model"),
        unauthenticated=True,
    )
    with (
        patch(
            "modal_training_gym.deploy_recipes.sglang_recipe.serve_sglang.build_sglang_serve_app",
            side_effect=_capture_build,
        ),
        patch(
            "modal_training_gym.common.deployment._run_coro",
            return_value="https://example.modal.run",
        ),
        patch(
            "modal_training_gym.common.deployment.ModelDeployment.save",
            return_value=None,
        ),
    ):
        cfg.serve()

    assert captured.get("unauthenticated") is True


def test_vllm_serve_rejects_unauthenticated() -> None:
    cfg = DeploymentConfig(
        model=ModelConfig(model_name="test/model"),
        recipe=VllmRecipe(),
        unauthenticated=True,
    )
    with pytest.raises(TrainingGymConfigError, match="only supported for SGLang"):
        cfg.serve()
