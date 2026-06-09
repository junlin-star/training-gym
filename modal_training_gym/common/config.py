"""User-local config persisted at ``~/.training-gym.toml``.

Populated by ``training-gym setup``; read by the slime launcher (and any other
caller) to look up where to POST phase reports and other client-side defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


CONFIG_PATH = Path.home() / ".training-gym.toml"


def load_config() -> dict[str, Any]:
    """Return the parsed ``~/.training-gym.toml``, or ``{}`` if missing."""
    if not CONFIG_PATH.is_file():
        return {}
    try:
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def save_dashboard_url(url: str) -> None:
    """Persist the deployed dashboard URL under ``[dashboard].url``."""
    config = load_config()
    dashboard = config.get("dashboard")
    if not isinstance(dashboard, dict):
        dashboard = {}
    dashboard["url"] = url
    config["dashboard"] = dashboard
    CONFIG_PATH.write_text(_render(config))


def get_dashboard_url() -> str | None:
    """Return the saved dashboard base URL, or ``None``."""
    dashboard = load_config().get("dashboard")
    if isinstance(dashboard, dict):
        url = dashboard.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def get_framework_status_url() -> str | None:
    """Return the saved framework-status endpoint URL, or ``None``."""
    base = get_dashboard_url()
    if not base:
        return None
    return base.rstrip("/") + "/api/framework-status"


def _render(config: dict[str, Any]) -> str:
    """Minimal TOML writer for the shapes we persist (string-valued tables)."""
    lines: list[str] = []
    for section, entries in config.items():
        if not isinstance(entries, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in entries.items():
            lines.append(f"{key} = {_format_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'
