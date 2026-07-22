"""Fetch pinned Slime source snapshots used by build-time patch golden tests.

The source files exist only inside the pinned Slime image, so this maintenance
utility uses Modal to read that image. Normal pytest collection never imports or
runs this module. It refreshes the committed ``*.input`` fixtures; CI checks the
resulting Git diff to detect image-source drift.
"""

from pathlib import Path

import modal

from modal_training_gym.frameworks.slime.launcher import SLIME_IMAGE

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA_DIR = REPO_ROOT / "tests" / "testdata"
SLIME_SOURCE_PATHS = {
    "train.py": "/root/slime/train.py",
    "train_async.py": "/root/slime/train_async.py",
    "slime_actor/megatron_actor.py": "/root/slime/slime/backends/megatron_utils/actor.py",
    "slime_actor/train_actor.py": "/root/slime/slime/ray/train_actor.py",
}

app = modal.App("fetch-slime-snapshots")
image = modal.Image.from_registry(SLIME_IMAGE).entrypoint([])


@app.function(image=image, serialized=True)
def read_sources() -> dict[str, str]:
    return {name: Path(path).read_text() for name, path in SLIME_SOURCE_PATHS.items()}


@app.local_entrypoint()
def main() -> None:
    TESTDATA_DIR.mkdir(exist_ok=True)
    for name, source in read_sources.remote().items():
        path = TESTDATA_DIR / f"{name}.input"
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
