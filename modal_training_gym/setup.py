"""Deploy the training-gym dashboard to Modal.

Usage (Python):
    import modal_training_gym
    modal_training_gym.setup()

Usage (CLI):
    training-gym setup

What this does:
1. Provisions a ``_training-gym-modal-creds`` Modal Secret containing
   ``MODAL_TOKEN_ID`` + ``MODAL_TOKEN_SECRET``. The deployed dashboard uses
   these credentials to stream logs from *other* Modal apps (the user's
   training runs) into the dashboard UI — container-default creds aren't
   scoped to read cross-app logs, so a workspace token is required.

   The underscore prefix marks the secret as auto-managed (hidden from the
   normal Secrets list in the UI). Credentials are auto-resolved from
   ``MODAL_TOKEN_*`` env vars or the active profile in ``~/.modal.toml``.

2. Deploys the dashboard's ASGI app.
3. Persists the FastAPI web URL to ``~/.training-gym.toml`` so other
   clients (e.g. the slime launcher) can find the dashboard.
"""

from __future__ import annotations

from modal_training_gym._dashboard import (
    MODAL_CREDS_SECRET_NAME,
    ensure_creds_secret,
)


def setup() -> str:
    """Deploy the training-gym dashboard, persist its URL, and return it."""
    import modal

    from modal_training_gym._dashboard import app, fastapi_app
    from modal_training_gym.common.config import CONFIG_PATH, save_dashboard_url

    if not ensure_creds_secret(interactive=True):
        print(
            f"WARNING: continuing without the {MODAL_CREDS_SECRET_NAME!r} "
            "Modal Secret — the dashboard will not be able to stream Modal "
            "app logs."
        )

    with modal.enable_output():
        app.deploy()

    web_url = fastapi_app.get_web_url()
    save_dashboard_url(web_url)
    print(f"\nDashboard deployed: {web_url}")
    print(f"Saved dashboard URL to {CONFIG_PATH}")
    return web_url
