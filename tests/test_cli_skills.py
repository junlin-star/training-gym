from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

from click.testing import CliRunner

from modal_training_gym import cli as cli_module
from modal_training_gym.cli.skills import (
    SKILL_NAME,
    _bundled_skill_path,
)


def _contents(path: Path) -> dict[Path, bytes]:
    return {
        file.relative_to(path): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


def _claude_link(project_root: Path) -> Path:
    return project_root / ".claude" / "skills" / SKILL_NAME


def _expected_claude_target(project_root: Path) -> Path:
    link = _claude_link(project_root)
    destination = project_root / ".agents" / "skills" / SKILL_NAME
    return Path(os.path.relpath(destination, start=link.parent))


def test_wheel_contains_bundled_skill(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=project_root,
        check=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    packaged_prefix = f"modal_training_gym/_skills/{SKILL_NAME}/"
    with zipfile.ZipFile(wheel) as archive:
        packaged_contents = {
            Path(name.removeprefix(packaged_prefix)): archive.read(name)
            for name in archive.namelist()
            if name.startswith(packaged_prefix) and not name.endswith("/")
        }

    source = project_root / "skills" / SKILL_NAME
    assert packaged_contents == _contents(source)


def test_skills_install_copies_bundled_skill_to_git_root(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = CliRunner().invoke(cli_module.entrypoint_cli, ["skills", "install"])

    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    assert result.exit_code == 0
    assert str(destination) in result.stdout
    assert _contents(destination) == _contents(_bundled_skill_path())
    assert _claude_link(tmp_path).is_symlink()
    assert _claude_link(tmp_path).readlink() == _expected_claude_target(tmp_path)
    assert _claude_link(tmp_path).resolve() == destination
    assert f"Installed {SKILL_NAME}" in result.stdout
    assert "Linked Claude skill" in result.stdout


def test_skills_install_is_idempotent(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    first = runner.invoke(cli_module.entrypoint_cli, ["skills", "install"])
    second = runner.invoke(cli_module.entrypoint_cli, ["skills", "install"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert f"{SKILL_NAME} is already installed" in second.stdout
    assert "Claude skill already linked" in second.stdout
    assert _claude_link(tmp_path).readlink() == _expected_claude_target(tmp_path)


def test_skills_install_keeps_equivalent_absolute_claude_link(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    assert (
        runner.invoke(cli_module.entrypoint_cli, ["skills", "install"]).exit_code == 0
    )
    link = _claude_link(tmp_path)
    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    link.unlink()
    link.symlink_to(destination, target_is_directory=True)

    result = runner.invoke(cli_module.entrypoint_cli, ["skills", "install"])

    assert result.exit_code == 0
    assert "Claude skill already linked" in result.stdout
    assert link.readlink() == destination


def test_skills_install_preserves_modified_skill_without_force(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    destination = tmp_path / ".agents" / "skills" / SKILL_NAME

    assert (
        runner.invoke(cli_module.entrypoint_cli, ["skills", "install"]).exit_code == 0
    )
    (destination / "SKILL.md").write_text("customized\n")

    result = runner.invoke(cli_module.entrypoint_cli, ["skills", "install"])

    assert result.exit_code == 1
    assert "already exists" in result.stderr
    assert (destination / "SKILL.md").read_text() == "customized\n"


def test_skills_install_force_replaces_modified_skill(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    destination = tmp_path / ".agents" / "skills" / SKILL_NAME

    assert (
        runner.invoke(cli_module.entrypoint_cli, ["skills", "install"]).exit_code == 0
    )
    (destination / "SKILL.md").write_text("customized\n")

    result = runner.invoke(
        cli_module.entrypoint_cli,
        ["skills", "install", "--force"],
    )

    assert result.exit_code == 0
    assert _contents(destination) == _contents(_bundled_skill_path())


def test_skills_install_preserves_existing_claude_skill_without_force(
    monkeypatch, tmp_path
):
    (tmp_path / ".git").mkdir()
    existing = _claude_link(tmp_path)
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text("customized\n")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli_module.entrypoint_cli, ["skills", "install"])

    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    assert result.exit_code == 0
    assert "Skipped Claude skill link" in result.stderr
    assert (existing / "SKILL.md").read_text() == "customized\n"
    assert _contents(destination) == _contents(_bundled_skill_path())


def test_skills_install_skips_wrong_claude_link_without_force(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    existing = _claude_link(tmp_path)
    existing.parent.mkdir(parents=True)
    existing.symlink_to(Path("somewhere-else"), target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli_module.entrypoint_cli, ["skills", "install"])

    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    assert result.exit_code == 0
    assert "Skipped Claude skill link" in result.stderr
    assert existing.readlink() == Path("somewhere-else")
    assert _contents(destination) == _contents(_bundled_skill_path())


def test_skills_install_force_replaces_existing_claude_skill(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    existing = _claude_link(tmp_path)
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text("customized\n")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["skills", "install", "--force"],
    )

    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    assert result.exit_code == 0
    assert existing.is_symlink()
    assert existing.readlink() == _expected_claude_target(tmp_path)
    assert existing.resolve() == destination


def test_force_skips_symlinked_claude_parent_but_installs_canonical_skill(
    monkeypatch, tmp_path
):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".claude").mkdir()
    real_skill = tmp_path / "skills" / SKILL_NAME
    real_skill.mkdir(parents=True)
    (real_skill / "SKILL.md").write_text("customized\n")
    (tmp_path / ".claude" / "skills").symlink_to(
        Path("..") / "skills",
        target_is_directory=True,
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["skills", "install", "--force"],
    )

    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    assert result.exit_code == 0
    assert _contents(destination) == _contents(_bundled_skill_path())
    assert "Skipped Claude skill link" in result.stderr
    assert (real_skill / "SKILL.md").read_text() == "customized\n"
    assert (tmp_path / ".claude" / "skills").is_symlink()


def test_force_skips_symlinked_claude_directory(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    real_claude_directory = tmp_path / "shared-claude"
    real_skill = real_claude_directory / "skills" / SKILL_NAME
    real_skill.mkdir(parents=True)
    (real_skill / "SKILL.md").write_text("customized\n")
    (tmp_path / ".claude").symlink_to(
        real_claude_directory,
        target_is_directory=True,
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["skills", "install", "--force"],
    )

    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    assert result.exit_code == 0
    assert _contents(destination) == _contents(_bundled_skill_path())
    assert "Skipped Claude skill link" in result.stderr
    assert (real_skill / "SKILL.md").read_text() == "customized\n"
    assert (tmp_path / ".claude").is_symlink()


def test_skills_install_accepts_parent_link_to_canonical_skills(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "skills").symlink_to(
        Path("..") / ".agents" / "skills",
        target_is_directory=True,
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli_module.entrypoint_cli, ["skills", "install"])

    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    assert result.exit_code == 0
    assert _contents(destination) == _contents(_bundled_skill_path())
    assert "Claude skills already linked through" in result.stdout
    assert "Skipped Claude skill link" not in result.stderr


def test_skills_install_keeps_canonical_copy_when_claude_linking_fails(
    monkeypatch, tmp_path
):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    original_symlink_to = Path.symlink_to

    def fail_claude_link(
        path: Path,
        target: Path,
        target_is_directory: bool = False,
    ) -> None:
        if path.name == SKILL_NAME:
            raise OSError("simulated link failure")
        original_symlink_to(
            path,
            target,
            target_is_directory=target_is_directory,
        )

    monkeypatch.setattr(Path, "symlink_to", fail_claude_link)

    result = CliRunner().invoke(cli_module.entrypoint_cli, ["skills", "install"])

    destination = tmp_path / ".agents" / "skills" / SKILL_NAME
    assert result.exit_code == 0
    assert _contents(destination) == _contents(_bundled_skill_path())
    assert "Skipped Claude skill link" in result.stderr
    assert "simulated link failure" in result.stderr


def test_skills_install_restores_existing_skill_when_replace_fails(
    monkeypatch, tmp_path
):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    destination = tmp_path / ".agents" / "skills" / SKILL_NAME

    assert (
        runner.invoke(cli_module.entrypoint_cli, ["skills", "install"]).exit_code == 0
    )
    (destination / "SKILL.md").write_text("customized\n")

    original_rename = Path.rename

    def fail_staged_install(path: Path, target: Path) -> Path:
        if (
            path.name == SKILL_NAME
            and path.parent.name.startswith(f".{SKILL_NAME}-")
            and target == destination
        ):
            raise OSError("simulated install failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staged_install)

    result = runner.invoke(
        cli_module.entrypoint_cli,
        ["skills", "install", "--force"],
    )

    assert result.exit_code == 1
    assert "simulated install failure" in result.stderr
    assert (destination / "SKILL.md").read_text() == "customized\n"


def test_skills_install_accepts_explicit_non_git_project(tmp_path):
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["skills", "install", "--project-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert (tmp_path / ".agents" / "skills" / SKILL_NAME / "SKILL.md").is_file()
    assert _claude_link(tmp_path).resolve() == (
        tmp_path / ".agents" / "skills" / SKILL_NAME
    )


def test_skills_install_requires_git_repo_without_project_dir(monkeypatch, tmp_path):
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: False if path.name == ".git" else original_exists(path),
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli_module.entrypoint_cli, ["skills", "install"])

    assert result.exit_code == 2
    assert "Could not find a Git repository" in result.stderr
