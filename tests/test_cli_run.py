from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from modal_training_gym import cli as cli_module
from modal_training_gym.cli import run as run_module


class FakeDashboardClient:
    payload: object = []
    payloads: dict[str, object] = {}
    streams: dict[str, list[tuple[str, str]]] = {}
    not_found_paths: set[str] = set()
    requests: list[tuple[str, dict[str, object]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_json(self, path, *, params=None, not_found_error=None):
        self.requests.append((path, params))
        if path in self.not_found_paths and not_found_error is not None:
            raise not_found_error
        return self.payloads.get(path, self.payload)

    def iter_event_stream(self, path, *, params=None, not_found_error=None):
        self.requests.append((path, params))
        if path in self.not_found_paths and not_found_error is not None:
            raise not_found_error
        yield from self.streams.get(path, [])


@pytest.fixture(autouse=True)
def fake_dashboard(monkeypatch):
    FakeDashboardClient.payload = []
    FakeDashboardClient.payloads = {}
    FakeDashboardClient.streams = {}
    FakeDashboardClient.not_found_paths = set()
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


def test_run_list_cli_field_names_are_unique():
    fields = run_module.run_list_field_metadata()
    output_names = [run_module.CLI_FIELD_NAMES.get(name, name) for name in fields]
    option_names = [
        run_module.CLI_FIELD_NAMES.get(name, name).replace("_", "-")
        for name, metadata in fields.items()
        if metadata.get("filterable")
    ]

    assert len(output_names) == len(set(output_names))
    assert len(option_names) == len(set(option_names))


def test_run_get_help_documents_flags_and_examples():
    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "get", "--help"])

    assert result.exit_code == 0
    assert "Show status and top-level metadata for a single run." in result.stdout
    assert "RUN_ID" in result.stdout
    assert "--verbose" in result.stdout
    assert "-j, --json" in result.stdout
    assert "Examples:" in result.stdout


def test_run_get_prints_top_level_status():
    FakeDashboardClient.payload = _summary(
        display_status="pending",
        display_stage="Generating rollouts",
        framework_progress={
            "current": 3,
            "total": 10,
            "unit": "step",
        },
        latest_rollout={
            "rollout_id": 2,
            "mean": 0.625,
            "total": 16,
            "created_at": 190,
        },
    )

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "get", "run-1"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.requests == [("/api/runs/run-1", None)]
    assert "PENDING" in result.stdout
    assert "Generating rollouts" in result.stdout
    assert "3 / 10 step" in result.stdout
    assert "0.625" in result.stdout
    assert "Field" not in result.stdout
    assert "Value" not in result.stdout


def test_run_get_verbose_json_includes_reward_history_and_rollouts():
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(
            display_status="pending",
            framework_progress={"current": 4, "total": 10, "unit": "step"},
            latest_rollout={
                "rollout_id": 2,
                "mean": 0.75,
                "total": 8,
                "created_at": 200,
            },
        ),
        "/api/runs/run-1/rollouts": [
            {
                "training_run_id": "run-1",
                "rollout_id": 1,
                "created_at": 150,
                "total": 8,
                "mean": 0.5,
                "rollout_time": 12.5,
            },
            {
                "training_run_id": "run-1",
                "rollout_id": 2,
                "created_at": 200,
                "total": 8,
                "mean": 0.75,
            },
        ],
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "get", "run-1", "--verbose", "-j"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.requests == [
        ("/api/runs/run-1", None),
        ("/api/runs/run-1/rollouts", None),
    ]
    payload = json.loads(result.stdout)
    assert payload["current_step"] == 4
    assert payload["total_steps"] == 10
    assert payload["current_reward"] == 0.75
    assert payload["reward_over_time"] == [
        {
            "rollout_id": 1,
            "reward": 0.5,
            "created_at": "1970-01-01T00:02:30Z",
        },
        {
            "rollout_id": 2,
            "reward": 0.75,
            "created_at": "1970-01-01T00:03:20Z",
        },
    ]
    assert (
        payload["rollouts"] == FakeDashboardClient.payloads["/api/runs/run-1/rollouts"]
    )


def test_run_get_missing_run_returns_not_found_without_fetching_rollouts():
    FakeDashboardClient.not_found_paths = {"/api/runs/missing"}

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "get", "missing", "--verbose"],
    )

    assert result.exit_code == 3
    assert "Training run 'missing' was not found." in result.stderr
    assert "training-gym run list" in result.stderr
    assert FakeDashboardClient.requests == [("/api/runs/missing", None)]


def test_run_params_prints_raw_recipe_as_json():
    params = {
        "gpu_type": "H200",
        "num_rollout": 4,
        "sequence_parallel": False,
        "eval_prompt_data": ["eval", "/data/eval.jsonl"],
        "sglang_config": {"attention_backend": "flashinfer"},
        "load": "",
        "critic_num_nodes": None,
    }
    FakeDashboardClient.payload = _summary(
        framework="slime",
        config={"recipe": params},
    )

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "params", "run-1", "-j"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.requests == [("/api/runs/run-1", None)]
    assert json.loads(result.stdout) == params


def test_run_params_supports_non_slime_frameworks():
    params = {"gpu_type": "H100", "num_nodes": 2}
    FakeDashboardClient.payload = _summary(
        framework="miles",
        config={"recipe": params},
    )

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "params", "run-1", "-j"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == params
    assert FakeDashboardClient.requests == [("/api/runs/run-1", None)]


def test_run_logs_help_documents_modes_and_filters():
    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "logs", "--help"])

    assert result.exit_code == 0
    assert "Show logs for a training run." in result.stdout
    for flag in ("--follow", "--since", "--until", "--tail", "--search", "--json"):
        assert flag in result.stdout
    assert "RUN_ID" in result.stdout
    assert "training-gym run logs brave-falcon-3fa8 --follow" in result.stdout


def test_run_logs_fetches_recent_filtered_entries():
    FakeDashboardClient.payload = {
        "logs": [
            {"task_id": "task-1", "line": "first\n", "fd": 1, "ts": 100.0},
            {"task_id": "task-1", "line": "second\n", "fd": 1, "ts": 101.0},
        ],
        "has_more": False,
        "next_until": None,
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "logs",
            "run-1",
            "--since",
            "30m",
            "--until",
            "2026-07-28T12:00:00Z",
            "--tail",
            "25",
            "--search",
            "worker",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == "first\nsecond\n"
    assert FakeDashboardClient.requests == [
        (
            "/api/runs/run-1/logs",
            {
                "since": "30m",
                "until": "2026-07-28T12:00:00Z",
                "max_lines": 25,
                "search": "worker",
            },
        )
    ]


def test_run_logs_json_preserves_entries_and_pagination():
    logs = [{"task_id": "task-1", "line": "hello", "fd": 1, "ts": 100.0}]
    FakeDashboardClient.payload = {
        "logs": logs,
        "has_more": True,
        "next_until": 99.5,
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "-j"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "logs": logs,
        "has_more": True,
        "next_until": 99.5,
    }
    assert "older logs are available with --until 99.5" in result.stderr
    assert FakeDashboardClient.requests[0][1]["max_lines"] == 100


def test_run_logs_warns_when_human_output_is_truncated():
    FakeDashboardClient.payload = {
        "logs": [{"task_id": "task-1", "line": "hello"}],
        "has_more": True,
        "next_until": None,
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1"],
    )

    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert "showing the newest 1 logs; older logs are available" in result.stderr


def test_run_logs_follow_streams_lines_until_done():
    FakeDashboardClient.streams["/api/runs/run-1/logs/stream"] = [
        ("message", '{"task_id":"task-1","line":"hello\\n","ts":100}'),
        ("dropped", '{"dropped":2}'),
        ("reconnect", '{"reason":"temporary error"}'),
        ("message", '{"task_id":"task-1","line":"world\\n","ts":101}'),
        ("done", "{}"),
    ]

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "--follow", "--search", "worker"],
    )

    assert result.exit_code == 0
    assert result.stdout == "hello\nworld\n"
    assert "[dropped 2 log lines]" in result.stderr
    assert "[reconnecting log stream]" in result.stderr
    assert FakeDashboardClient.requests == [
        ("/api/runs/run-1/logs/stream", {"search": "worker"})
    ]


def test_run_logs_follow_json_uses_newline_delimited_events():
    FakeDashboardClient.streams["/api/runs/run-1/logs/stream"] = [
        ("message", '{"task_id":"task-1","line":"first","ts":100}'),
        ("message", '{"task_id":"task-1","line":"second","ts":101}'),
        ("message", '{"task_id":"task-1","line":"third","ts":102}'),
        ("done", "{}"),
    ]

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "--follow", "-j"],
    )

    assert result.exit_code == 0
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {"task_id": "task-1", "line": "first", "ts": 100},
        {"task_id": "task-1", "line": "second", "ts": 101},
        {"task_id": "task-1", "line": "third", "ts": 102},
    ]


def test_run_logs_rejects_historical_bounds_with_follow():
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "--follow", "--since", "30m"],
    )

    assert result.exit_code == 2
    assert "apply only when fetching logs without --follow" in result.stderr
    assert FakeDashboardClient.requests == []


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
