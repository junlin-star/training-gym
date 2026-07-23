"""Shared HTTP client for the deployed training-gym dashboard."""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

from modal_training_gym.common.config import get_dashboard_url

from .errors import (
    AuthenticationError,
    DashboardConfigurationError,
    DashboardError,
    DashboardNetworkError,
    DashboardServerError,
    DashboardTimeoutError,
    MalformedResponseError,
    ResourceNotFoundError,
)


DASHBOARD_PASSWORD_ENV = "TRAINING_GYM_DASHBOARD_PASSWORD"
DEFAULT_TIMEOUT_SECONDS = 10.0
QueryParams = Mapping[str, str | int | float | bool | None]


class DashboardClient:
    """Transport shared by dashboard-backed commands."""

    def __init__(
        self,
        *,
        password: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        configured_url = (get_dashboard_url() or "").strip()
        if not configured_url:
            raise DashboardConfigurationError(
                "Dashboard URL is not configured; run `training-gym setup` first."
            )

        parsed = urlsplit(configured_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise DashboardConfigurationError(
                "Configured dashboard URL must use HTTP or HTTPS."
            )

        dashboard_password = (
            os.environ.get(DASHBOARD_PASSWORD_ENV, "") if password is None else password
        )
        auth = (
            httpx.BasicAuth("training-gym", dashboard_password)
            if dashboard_password
            else None
        )
        self._client = httpx.Client(
            base_url=configured_url.rstrip("/") + "/",
            auth=auth,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def get_json(
        self,
        path: str,
        *,
        params: QueryParams | None = None,
    ) -> Any:
        """GET a dashboard-relative path and decode its JSON response."""
        parsed_path = urlsplit(path)
        if parsed_path.scheme or parsed_path.netloc:
            raise DashboardError("Dashboard request path must be relative.")

        query = (
            {key: value for key, value in params.items() if value is not None}
            if params
            else None
        )
        try:
            response = self._client.get(path.lstrip("/"), params=query)
        except httpx.TimeoutException as exc:
            raise DashboardTimeoutError("Dashboard request timed out.") from exc
        except httpx.RequestError as exc:
            raise DashboardNetworkError("Could not connect to the dashboard.") from exc

        self._raise_for_status(response.status_code)
        try:
            return response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise MalformedResponseError("Dashboard returned malformed JSON.") from exc

    @staticmethod
    def _raise_for_status(status_code: int) -> None:
        if status_code in {401, 403}:
            raise AuthenticationError("Dashboard authentication was rejected.")
        if status_code == 404:
            raise ResourceNotFoundError("Dashboard resource was not found.")
        if status_code >= 500:
            raise DashboardServerError(f"Dashboard returned HTTP {status_code}.")
        if status_code >= 400:
            raise DashboardError(f"Dashboard returned HTTP {status_code}.")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
