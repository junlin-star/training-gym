from pathlib import Path


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping async timing flush patch")
        return

    source = path.read_text()
    flush = (
        "        from modal_training_gym.frameworks.slime.phase_reporting import (\n"
        "            flush_async_timing_events,\n"
        "        )\n"
        "        flush_async_timing_events()\n"
    )
    if flush in source:
        return
    target = "        return result\n\n    def train_critic("
    if target not in source:
        print(f"WARNING: Could not patch {path} with async timing flush")
        return
    path.write_text(source.replace(target, flush + target, 1))


if __name__ == "__main__":
    _patch_file(Path("/root/slime/slime/backends/megatron_utils/actor.py"))
