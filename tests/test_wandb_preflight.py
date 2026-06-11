"""The slime launcher's W&B pre-flight: turn a recurring mid-training wandb
CommError into an early, actionable failure (missing key, or no write access).
"""

import sys
import types

import pytest

from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.frameworks.slime.launcher import _preflight_wandb

_CFG = WandbConfig(project="qwen3-asr-rl", modal_wandb_secret_name="wandb-secret")


def _stub_wandb(monkeypatch, **attrs):
    """Stand in for the lazily-imported ``wandb`` module with a fake exposing *attrs*.

    ``_preflight_wandb`` does ``import wandb`` inside the function, so preloading a
    fake into ``sys.modules`` intercepts it — no real library, network, or login.
    """
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(**attrs))


def test_preflight_raises_clear_error_without_key(monkeypatch):
    """No WANDB_API_KEY → a clear error naming the missing var, before any GPU work."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        _preflight_wandb(_CFG)


def test_preflight_wraps_access_failure(monkeypatch):
    """Key present but W&B rejects the write → the raw wandb error is
    re-raised as a RuntimeError that names the project."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    def login_without_write_access(**_):
        raise Exception("user does not have models write access")

    _stub_wandb(monkeypatch, login=login_without_write_access)
    with pytest.raises(RuntimeError, match="W&B pre-flight failed.*qwen3-asr-rl"):
        _preflight_wandb(_CFG)
