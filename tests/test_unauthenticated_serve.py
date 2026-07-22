"""Local checks for DeploymentConfig(unauthenticated=...)."""

from __future__ import annotations

import inspect
import warnings
from unittest.mock import MagicMock, patch

import pytest
from modal_training_gym.common.deployment import DeploymentConfig, ModelDeployment
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


def test_default_unauthenticated_is_true() -> None:
    cfg = DeploymentConfig(model=ModelConfig(model_name="test/model"))
    assert cfg.unauthenticated is True


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


def _serve_vllm(cfg: DeploymentConfig) -> tuple[object, MagicMock]:
    fake_app = MagicMock()
    fake_app.app_id = "ap-test"
    fake_server = MagicMock()
    fake_server.get_url = MagicMock(return_value="https://example.modal.run")
    fake_app.Server = fake_server
    with (
        patch(
            "modal_training_gym.deploy_recipes.vllm_recipe.serve_vllm.build_vllm_serve_app",
            return_value=fake_app,
        ) as mock_build,
        patch(
            "modal_training_gym.common.deployment._run_coro",
            return_value="https://example.modal.run",
        ),
        patch(
            "modal_training_gym.common.deployment.ModelDeployment.save",
            return_value=None,
        ),
        patch(
            "modal_training_gym.common.deployment.modal_app_dashboard_url",
            return_value="https://modal.com/apps/ap-test",
        ),
    ):
        deployment = cfg.serve()
    return deployment, mock_build


def test_vllm_serve_ignores_unauthenticated_true() -> None:
    cfg = DeploymentConfig(
        model=ModelConfig(model_name="test/model"),
        recipe=VllmRecipe(),
        unauthenticated=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deployment, mock_build = _serve_vllm(cfg)
    assert deployment.url == "https://example.modal.run"
    mock_build.assert_called_once()
    assert "unauthenticated" not in mock_build.call_args.kwargs
    assert not [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "unauthenticated=False" in str(w.message)
    ]


def test_vllm_serve_ignores_default_unauthenticated() -> None:
    cfg = DeploymentConfig(
        model=ModelConfig(model_name="test/model"),
        recipe=VllmRecipe(),
    )
    assert cfg.unauthenticated is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deployment, mock_build = _serve_vllm(cfg)
    assert deployment.url == "https://example.modal.run"
    mock_build.assert_called_once()
    assert "unauthenticated" not in mock_build.call_args.kwargs
    assert not [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "unauthenticated=False" in str(w.message)
    ]


def test_vllm_serve_warns_on_unauthenticated_false() -> None:
    cfg = DeploymentConfig(
        model=ModelConfig(model_name="test/model"),
        recipe=VllmRecipe(),
        unauthenticated=False,
    )
    with pytest.warns(UserWarning, match="unauthenticated=False"):
        deployment, mock_build = _serve_vllm(cfg)
    assert deployment.url == "https://example.modal.run"
    mock_build.assert_called_once()
    assert "unauthenticated" not in mock_build.call_args.kwargs


def test_from_config_missing_unauthenticated_defaults_true() -> None:
    md = ModelDeployment.model_validate(
        {
            "deployment_id": "dep-1",
            "url": "https://example.modal.run",
            "deployment_config": {
                "model": {"model_name": "test/model"},
                "app_name": "test-serve",
                "served_model_name": "model",
            },
        }
    )
    assert md.deployment_config.unauthenticated is True
