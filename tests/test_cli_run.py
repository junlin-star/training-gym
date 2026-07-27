from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from modal_training_gym import cli as cli_module
from modal_training_gym.cli import run as run_module


class FakeDashboardClient:
    payload: object = []
    requests: list[tuple[str, dict[str, object]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_json(self, path, *, params):
        self.requests.append((path, params))
        return self.payload


@pytest.fixture(autouse=True)
def fake_dashboard(monkeypatch):
    FakeDashboardClient.payload = []
    FakeDashboardClient.requests = []
    monkeypatch.setattr(run_module, "DashboardClient", FakeDashboardClient)


def _summary(**overrides):
    value = {
        "training_run_id": "run-1",
        "run_id": "run-1",
        "status": "failed",
        "display_status": "failed",
        "display_stage": "Training",
        "framework_status": "training",
        "model": "org/model",
        "dataset": "org/data",
        "recipe": "slime",
        "group_id": "nightly",
        "created_at": 100,
        "updated_at": 200,
    }
    value.update(overrides)
    return value


def test_run_list_help_derives_filter_flags_from_schema():
    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "list", "--help"])

    assert result.exit_code == 0
    for flag in (
        "--status",
        "--model",
        "--dataset",
        "--recipe",
        "--group",
    ):
        assert flag in result.stdout
    assert "--since" in result.stdout
    assert "--limit" in result.stdout
    assert "-j, --json" in result.stdout
    assert "--group GROUP" in result.stdout


@pytest.mark.parametrize(
    ("flag", "value", "backend_field"),
    [
        ("--status", "failed", "display_status"),
        ("--model", "org/model", "model"),
        ("--dataset", "org/data", "dataset"),
        ("--recipe", "slime", "recipe"),
        ("--group", "nightly", "group_id"),
    ],
)
def test_run_list_forwards_each_filter_flag(flag, value, backend_field):
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "list", flag, value, "-j"],
    )

    assert result.exit_code == 0
    path, params = FakeDashboardClient.requests[0]
    assert path == "/api/runs"
    assert params[backend_field] == value


def test_run_list_forwards_filters_and_prints_configured_fields_as_json():
    FakeDashboardClient.payload = [_summary()]

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "list",
            "--status",
            "failed",
            "--group",
            "nightly",
            "--since",
            "2026-07-23T12:00:00Z",
            "--limit",
            "3",
            "-j",
        ],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.requests == [
        (
            "/api/runs",
            {
                "display_status": "failed",
                "model": None,
                "dataset": None,
                "recipe": None,
                "group_id": "nightly",
                "since": 1784808000,
                "limit": 3,
            },
        )
    ]
    assert json.loads(result.stdout) == [
        {
            "run_id": "run-1",
            "status": "failed",
            "stage": "Training",
            "model": "org/model",
            "dataset": "org/data",
            "recipe": "slime",
            "group": "nightly",
            "created_at": "1970-01-01T00:01:40Z",
            "last_updated_at": "1970-01-01T00:03:20Z",
        }
    ]


def test_run_list_rejects_invalid_since_before_request():
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "list", "--since", "yesterday-ish"],
    )

    assert result.exit_code == 2
    assert "epoch seconds, ISO 8601, or a relative time" in result.stderr
    assert FakeDashboardClient.requests == []


def test_run_list_renders_schema_columns():
    FakeDashboardClient.payload = [_summary()]

    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "list"])

    assert result.exit_code == 0
    for heading in (
        "Run",
        "Status",
        "Stage",
        "Model",
        "Dataset",
        "Recipe",
        "Group",
        "Created",
        "Last updated",
    ):
        assert heading in result.stdout
