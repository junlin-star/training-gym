"""Modal app serving the Starlight documentation site.

Serves pre-built static files. Run the build locally before deploying:

    cd docs-next && npm install && npm run build

Deploy:

    uv run modal deploy docs-next/docs_next_app.py

Local development:

    cd docs-next && npm run dev
"""

from __future__ import annotations

from pathlib import Path

import modal

DOCS_DIR = Path(__file__).resolve().parent
DIST_DIR = DOCS_DIR / "dist"
REMOTE_DIST = "/assets/dist"


if not modal.is_local():
    pass
elif not DIST_DIR.exists() or not any(DIST_DIR.iterdir()):
    import subprocess
    import sys

    REPO_ROOT = DOCS_DIR.parent
    subprocess.check_call(
        [sys.executable, "scripts/generate_all.py"], cwd=str(REPO_ROOT)
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.118.0")
    .add_local_dir(DIST_DIR, remote_path=REMOTE_DIST, copy=True)
)

app = modal.App("training-gym-docs", image=image)


@app.function(min_containers=1)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(custom_domains=["gym.modal.dev"])
def serve():
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.staticfiles import StaticFiles

    web = FastAPI()
    web.add_middleware(GZipMiddleware, minimum_size=500)

    @web.middleware("http")
    async def cache_control(request: Request, call_next):
        response: Response = await call_next(request)
        path = request.url.path
        if path.startswith("/_astro/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.endswith(".html") or "text/html" in response.headers.get(
            "content-type", ""
        ):
            response.headers["Cache-Control"] = (
                "public, max-age=3600, stale-while-revalidate=86400"
            )
        return response

    web.mount("/", StaticFiles(directory=REMOTE_DIST, html=True), name="static")

    return web
