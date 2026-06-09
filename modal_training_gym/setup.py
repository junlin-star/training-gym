"""Deploy the training-gym dashboard to Modal.

Usage (Python):
    import modal_training_gym
    modal_training_gym.setup()

Usage (CLI):
    training-gym setup

After deploying, the FastAPI endpoint URL is persisted to
``~/.training-gym.toml`` so other clients (e.g. the slime launcher) can find
the dashboard without a hardcoded fallback.
"""

from __future__ import annotations


def setup() -> str:
    """Deploy the training-gym dashboard, persist its URL, and return it."""
    import modal

    from modal_training_gym._dashboard import app, fastapi_app
    from modal_training_gym.common.config import CONFIG_PATH, save_dashboard_url

    with modal.enable_output():
        app.deploy()

    web_url = fastapi_app.get_web_url()
    save_dashboard_url(web_url)
    print(f"\nDashboard deployed: {web_url}")
    print(f"Saved dashboard URL to {CONFIG_PATH}")
    return web_url
