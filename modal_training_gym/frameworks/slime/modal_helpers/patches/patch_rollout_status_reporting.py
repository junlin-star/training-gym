"""Patch slime train entrypoints to report rollout-engine startup status."""

from __future__ import annotations

import re
from pathlib import Path


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping rollout-status patch")
        return

    src = path.read_text()
    marker = "PATCHED_TRAINING_GYM_ROLLOUT_STATUS"
    if marker in src:
        print(f"{path.name} already patched for rollout status reporting")
        return

    pattern = re.compile(
        r"^(?P<indent>[ \t]*)rollout_manager, num_rollout_per_epoch = create_rollout_manager\(args, pgs\[\"rollout\"\]\)",
        re.M,
    )
    def _replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}# {marker}: rollout engine startup state\n"
            f"{indent}__import__('sys').path.insert(0, '/root')\n"
            f"{indent}__import__('modal_training_gym.frameworks.slime.phase_reporting', fromlist=['report_rollout_initializing']).report_rollout_initializing(args)\n"
            f"{indent}rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs[\"rollout\"])"
        )

    src, count = pattern.subn(_replacement, src, count=1)
    if count != 1:
        print(f"WARNING: Could not patch {path.name} for rollout status reporting")
        return

    path.write_text(src)
    print(f"Patched {path.name} with rollout status reporting")


_patch_file(Path("/root/slime/train.py"))
_patch_file(Path("/root/slime/train_async.py"))
