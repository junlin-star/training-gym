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
}
MEGATRON_MODEL_PATH = "/root/slime/slime/backends/megatron_utils/model.py"
MEGATRON_HOOK_SEQUENCES = {
    "log_prob": (
        "# Turn on evaluation mode which disables dropout.",
        "custom_before_log_prob_hook(args, model, store_prefix)",
        "forward_backward_func = get_forward_backward_func()",
        "forward_data_store += forward_backward_func(",
    ),
    "train_step": (
        "# Set grad to zero.",
        "optimizer.zero_grad()",
        "custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)",
        "# Forward pass.",
        "losses_reduced = forward_backward_func(",
    ),
}

app = modal.App("fetch-slime-snapshots")
image = modal.Image.from_registry(SLIME_IMAGE).entrypoint([])


@app.function(image=image, serialized=True)
def read_sources() -> dict[str, str]:
    sources = {
        name: Path(path).read_text() for name, path in SLIME_SOURCE_PATHS.items()
    }
    model_source = Path(MEGATRON_MODEL_PATH).read_text()
    hook_snapshot: list[str] = []
    for name, anchors in MEGATRON_HOOK_SEQUENCES.items():
        positions = [model_source.index(anchor) for anchor in anchors]
        if positions != sorted(positions):
            raise RuntimeError(f"Unexpected {name} hook ordering in pinned Slime")
        hook_snapshot.extend((f"[{name}]", *anchors))
    sources["slime_actor/megatron_model_hooks.txt"] = "\n".join(hook_snapshot) + "\n"
    return sources


@app.local_entrypoint()
def main() -> None:
    TESTDATA_DIR.mkdir(exist_ok=True)
    for name, source in read_sources.remote().items():
        path = TESTDATA_DIR / f"{name}.input"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
