from pathlib import Path


MARKER = "PATCHED_TRAINING_GYM_TRAINING_TIMING_COMPLETION"


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping training timing completion patch")
        return

    source = path.read_text()
    if MARKER in source:
        return

    import_target = "from slime.utils.types import RolloutBatch\n"
    return_target = "        return result\n"
    if import_target not in source or source.count(return_target) != 1:
        print(f"WARNING: Could not patch {path} for training timing completion")
        return

    reporter_import = (
        f"# {MARKER}\n"
        "from modal_training_gym.frameworks.slime.phase_reporting import (\n"
        "    report_training_role_finished as _tg_report_training_role_finished,\n"
        ")\n"
    )
    completed_training = (
        "        _tg_report_training_role_finished(self.args, rollout_id, self.role)\n"
        "        return result\n"
    )
    source = source.replace(import_target, import_target + reporter_import, 1)
    source = source.replace(return_target, completed_training, 1)
    path.write_text(source)


if __name__ == "__main__":
    _patch_file(Path("/root/slime/slime/backends/megatron_utils/actor.py"))
