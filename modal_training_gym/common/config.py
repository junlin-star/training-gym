"""User-local config persisted at ``~/.training-gym.toml``.

Populated by ``training-gym setup``; read by the slime launcher (and any other
caller) to look up where to POST phase reports and other client-side defaults.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


CONFIG_PATH = Path.home() / ".training-gym.toml"
MODAL_CONFIG_PATH = Path(
    os.environ.get("MODAL_CONFIG_PATH") or os.path.expanduser("~/.modal.toml")
)


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


# ── Modal credential resolution ──────────────────────────────────────────


def read_modal_toml_creds() -> tuple[str, str, str]:
    """Resolve ``(token_id, token_secret, profile_name)`` from ``~/.modal.toml``.

    Follows Modal's own profile-selection rules: the ``MODAL_PROFILE`` env
    var wins if set; otherwise the profile flagged ``active = true``;
    otherwise the ``[default]`` profile; otherwise the first profile in the
    file. Returns empty strings if no credentials can be found.
    """
    if not MODAL_CONFIG_PATH.is_file():
        return "", "", ""

    try:
        with MODAL_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return "", "", ""

    profiles = {
        name: section for name, section in data.items() if isinstance(section, dict)
    }
    if not profiles:
        return "", "", ""

    candidate_names: list[str] = []
    env_profile = os.environ.get("MODAL_PROFILE", "").strip()
    if env_profile:
        candidate_names.append(env_profile)
    candidate_names.extend(
        name for name, sec in profiles.items() if sec.get("active") is True
    )
    if "default" in profiles:
        candidate_names.append("default")
    candidate_names.extend(profiles.keys())

    seen: set[str] = set()
    for name in candidate_names:
        if name in seen or name not in profiles:
            continue
        seen.add(name)
        section = profiles[name]
        token_id = str(section.get("token_id") or "").strip()
        token_secret = str(section.get("token_secret") or "").strip()
        if token_id and token_secret:
            return token_id, token_secret, name
    return "", "", ""


def resolve_modal_creds() -> tuple[str, str, str]:
    """Resolve Modal credentials with a source label for logging.

    Order: ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` env vars → active
    profile in ``~/.modal.toml``.
    """
    env_id = (os.environ.get("MODAL_TOKEN_ID") or "").strip()
    env_secret = (os.environ.get("MODAL_TOKEN_SECRET") or "").strip()
    if env_id and env_secret:
        return env_id, env_secret, "environment"

    toml_id, toml_secret, profile_name = read_modal_toml_creds()
    if toml_id and toml_secret:
        return toml_id, toml_secret, f"~/.modal.toml profile [{profile_name}]"
    return "", "", ""
