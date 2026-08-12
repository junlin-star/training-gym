"""Install Training Gym agent skills into a project."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import click

from .commands import _TrainingGymGroup
from .errors import CLIError


SKILL_NAME = "agent-driven-training"
SKILLS_DIRECTORY = Path(".agents") / "skills"
CLAUDE_SKILLS_DIRECTORY = Path(".claude") / "skills"


def _bundled_skill_path() -> Path:
    """Locate the bundled skill in an installed wheel or source checkout."""
    package_root = Path(__file__).resolve().parent.parent
    candidates = (
        package_root / "_skills" / SKILL_NAME,
        package_root.parent / "skills" / SKILL_NAME,
    )
    for candidate in candidates:
        if (candidate / "SKILL.md").is_file():
            return candidate
    raise CLIError(
        f"The bundled {SKILL_NAME!r} skill is unavailable.",
        error="skill_not_bundled",
        hint="Reinstall modal-training-gym and try again.",
    )


def _find_project_root(start: Path) -> Path:
    """Return the nearest Git repository containing ``start``."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise click.UsageError(
        "Could not find a Git repository. Run this command inside a repository "
        "or pass --project-dir."
    )


def _symlinked_claude_link_parent(project_root: Path) -> Path | None:
    """Return the first symlink in the Claude skill parent hierarchy."""
    claude_directory = project_root / ".claude"
    claude_skills_directory = project_root / CLAUDE_SKILLS_DIRECTORY
    for candidate in (claude_directory, claude_skills_directory):
        if candidate.is_symlink():
            return candidate
    return None


def _directory_contents(path: Path) -> dict[Path, bytes]:
    """Return file contents keyed by paths relative to ``path``."""
    return {
        file.relative_to(path): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


def _install_claude_link(
    link: Path,
    destination: Path,
    *,
    replace_existing: bool,
) -> None:
    """Link Claude's skill directory to the canonical skill."""
    staging_root: Path | None = None
    backup: Path | None = None
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{SKILL_NAME}-",
                dir=link.parent,
            )
        )
        staged_link = staging_root / SKILL_NAME
        target = Path(os.path.relpath(destination, start=link.parent))
        staged_link.symlink_to(target, target_is_directory=True)

        if replace_existing:
            backup = staging_root / "previous"
            link.rename(backup)
        try:
            staged_link.rename(link)
        except OSError:
            if backup is not None:
                backup.rename(link)
            raise
    except OSError as exc:
        raise CLIError(
            f"Could not link {SKILL_NAME} at {link}: {exc}",
            error="skill_install_failed",
        ) from exc
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)


def _install_canonical_skill(
    source: Path,
    destination: Path,
    *,
    force: bool,
) -> bool:
    """Install the canonical skill and return whether it was already installed."""
    replace_existing = False
    if destination.is_symlink():
        if not force:
            raise CLIError(
                f"{destination} is a symbolic link.",
                error="skill_destination_exists",
                hint="Move it aside or rerun with --force.",
            )
        replace_existing = True
    elif destination.exists() and not destination.is_dir():
        if not force:
            raise CLIError(
                f"{destination} exists and is not a directory.",
                error="skill_destination_exists",
                hint="Move it aside or rerun with --force.",
            )
        replace_existing = True
    elif destination.is_dir():
        if _directory_contents(source) == _directory_contents(destination):
            return True
        if not force:
            raise CLIError(
                f"{SKILL_NAME} already exists at {destination}.",
                error="skill_destination_exists",
                hint="Rerun with --force to replace it.",
            )
        replace_existing = True

    staging_root: Path | None = None
    backup: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{SKILL_NAME}-",
                dir=destination.parent,
            )
        )
        staged_skill = staging_root / SKILL_NAME
        shutil.copytree(source, staged_skill)

        if replace_existing:
            backup = staging_root / "previous"
            destination.rename(backup)
        try:
            staged_skill.rename(destination)
        except OSError:
            if backup is not None:
                backup.rename(destination)
            raise
    except OSError as exc:
        raise CLIError(
            f"Could not install {SKILL_NAME} at {destination}: {exc}",
            error="skill_install_failed",
        ) from exc
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
    return False


def _ensure_claude_compatibility(
    project_root: Path,
    destination: Path,
    *,
    force: bool,
) -> None:
    """Expose the canonical skill to Claude when its paths are safe to manage."""
    skills_directory = project_root / CLAUDE_SKILLS_DIRECTORY
    link = skills_directory / SKILL_NAME
    symlinked_parent = _symlinked_claude_link_parent(project_root)

    if symlinked_parent is not None:
        try:
            parent_exposes_canonical = (
                skills_directory.resolve() == destination.parent.resolve()
            )
        except (OSError, RuntimeError):
            parent_exposes_canonical = False
        if parent_exposes_canonical:
            click.echo(f"Claude skills already linked through {skills_directory}")
        else:
            click.echo(
                "Skipped Claude skill link because "
                f"{symlinked_parent} is a symbolic link.",
                err=True,
            )
        return

    link_is_symlink = link.is_symlink()
    link_exists = link_is_symlink or link.exists()
    link_points_to_canonical = False
    if link_is_symlink:
        try:
            link_points_to_canonical = link.resolve() == destination.resolve()
        except (OSError, RuntimeError):
            pass
    if link_points_to_canonical:
        click.echo(f"Claude skill already linked at {link}")
        return
    if link_exists and not force:
        click.echo(
            f"Skipped Claude skill link because {link} already exists; "
            "rerun with --force to replace it.",
            err=True,
        )
        return

    try:
        _install_claude_link(
            link,
            destination,
            replace_existing=link_exists,
        )
    except CLIError as exc:
        click.echo(f"Skipped Claude skill link: {exc.format_message()}", err=True)
        return
    click.echo(f"Linked Claude skill at {link}")


def install_skills(*, project_dir: Path | None, force: bool) -> Path:
    """Install the bundled skill and return its destination."""
    project_root = (
        project_dir.expanduser().resolve()
        if project_dir is not None
        else _find_project_root(Path.cwd())
    )
    source = _bundled_skill_path()
    destination = project_root / SKILLS_DIRECTORY / SKILL_NAME
    skill_installed = _install_canonical_skill(source, destination, force=force)

    if skill_installed:
        click.echo(f"{SKILL_NAME} is already installed at {destination}")
    else:
        click.echo(f"Installed {SKILL_NAME} at {destination}")
    _ensure_claude_compatibility(project_root, destination, force=force)
    return destination


@click.group("skills", cls=_TrainingGymGroup)
def skills_group() -> None:
    """Manage Training Gym agent skills."""


@skills_group.command("install")
@click.option(
    "--project-dir",
    type=click.Path(
        path_type=Path,
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    default=None,
    metavar="DIR",
    help="Project root. Defaults to the nearest Git repository.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing canonical skill or manageable Claude child path.",
)
def install_command(*, project_dir: Path | None, force: bool) -> None:
    """Install agent-driven-training with optional Claude compatibility."""
    install_skills(project_dir=project_dir, force=force)
