"""Tests for orphaned deployment reconciliation."""

from __future__ import annotations

from modal_training_gym.common.deployment import DeploymentStatus
from modal_training_gym.common.deployment_reconciler import (
    reconcile_decision,
    reconcile_orphan_deployments,
)
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_put


def _deployment(
    *,
    status: str = DeploymentStatus.RUNNING.value,
    modal_app_id: str = "ap-123",
) -> dict[str, object]:
    return {
        "deployment_id": "deployment-1",
        "modal_app_id": modal_app_id,
        "status": status,
    }


def test_stopped_deployment_is_ignored():
    decision = reconcile_decision(
        _deployment(status=DeploymentStatus.STOPPED.value),
        app_live=False,
    )

    assert decision.should_terminalize is False


def test_dead_modal_app_is_terminalized():
    decision = reconcile_decision(
        _deployment(),
        app_live=False,
        modal_app_state=99,
    )

    assert decision.should_terminalize is True
    assert decision.reason == "stale_modal_app_terminated"
    assert decision.modal_app_state == 99


def test_live_modal_app_is_ignored():
    decision = reconcile_decision(_deployment(), app_live=True)

    assert decision.should_terminalize is False


def test_unknown_modal_app_state_is_ignored():
    decision = reconcile_decision(_deployment(), app_live=None)

    assert decision.should_terminalize is False


def test_reconcile_orphan_deployments_updates_status(fake_volume):
    deployment = _deployment()
    vol_put(MetadataStore.DEPLOYMENTS, "deployment-1", deployment)

    results = reconcile_orphan_deployments(
        check_app_live=lambda _app_id: False,
        get_lifecycle_state=lambda _app_id: 99,
    )

    assert len(results) == 1
    assert results[0].deployment_id == "deployment-1"
    assert results[0].reason == "stale_modal_app_terminated"
    assert results[0].previous_status == DeploymentStatus.RUNNING.value
    assert (
        vol_get(MetadataStore.DEPLOYMENTS, "deployment-1")["status"]
        == DeploymentStatus.STOPPED.value
    )
