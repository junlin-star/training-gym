"""Typed CLI failures with process exit codes."""

from __future__ import annotations

from enum import IntEnum

import click


class ExitCode(IntEnum):
    SUCCESS = 0
    UNEXPECTED = 1
    USAGE = 2
    NOT_FOUND = 3
    AUTH = 4
    DASHBOARD = 5
    INTERRUPTED = 130


class CLIError(click.ClickException):
    """Base class for expected errors that are safe to show to users."""

    exit_code = ExitCode.UNEXPECTED


class CLIUsageError(click.UsageError):
    exit_code = ExitCode.USAGE


class ResourceNotFoundError(CLIError):
    exit_code = ExitCode.NOT_FOUND


class AuthenticationError(CLIError):
    exit_code = ExitCode.AUTH


class DashboardError(CLIError):
    exit_code = ExitCode.DASHBOARD


class DashboardConfigurationError(DashboardError):
    """Local dashboard configuration is missing or invalid."""


class DashboardNetworkError(DashboardError):
    """The dashboard could not be reached."""


class DashboardTimeoutError(DashboardError):
    """The dashboard request exceeded its timeout."""


class DashboardServerError(DashboardError):
    """The dashboard returned a server-side failure."""


class MalformedResponseError(DashboardError):
    """The dashboard response was not valid JSON."""
