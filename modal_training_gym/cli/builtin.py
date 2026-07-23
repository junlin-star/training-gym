"""Click wrappers for the CLI's existing top-level commands."""

from __future__ import annotations

import click

from .commands import TrainingGymCommand


@click.command("setup", cls=TrainingGymCommand)
def setup_command() -> None:
    """Deploy the training-gym dashboard to Modal."""
    from .setup import setup
    setup()


@click.command("open", cls=TrainingGymCommand)
def open_command() -> None:
    """Open the deployed dashboard in your browser."""
    from .setup import open_dashboard
    open_dashboard()


@click.command("set-proxy-auth", cls=TrainingGymCommand)
def set_proxy_auth_command() -> None:
    """Set/replace the Modal proxy-auth tokens (MODAL_KEY / MODAL_SECRET)"""
    from .setup import set_proxy_auth
    set_proxy_auth()


@click.command("set-password", cls=TrainingGymCommand)
@click.option(
    "--password",
    default=None,
    metavar="PASSWORD",
    help="Password to set (prompted securely if omitted; empty disables auth).",
)
def set_password_command(password: str | None) -> None:
    """Set or clear the dashboard password (Basic Auth) and redeploy."""
    from .setup import set_password
    set_password(password=password)


@click.command("cleanup", cls=TrainingGymCommand)
@click.option(
    "--older-than-days",
    type=int,
    default=7,
    show_default=True,
    metavar="DAYS",
    help="Delete failed or cancelled runs older than this many days.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without deleting.",
)
def cleanup_command(older_than_days: int, dry_run: bool) -> None:
    """Delete metadata for old failed or cancelled runs."""
    from .cleanup import cleanup
    cleanup(older_than_days=older_than_days, dry_run=dry_run)
