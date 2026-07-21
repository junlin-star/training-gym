from pathlib import Path


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping critic value timing patch")
        return

    source = path.read_text()
    reporter = (
        "from modal_training_gym.frameworks.slime.phase_reporting import (\n"
        "    report_step_event as _tg_report_step_event,\n"
        ")\n"
    )
    timed_value_inference = (
        "        _tg_record_critic_value_timing = "
        "self.args.async_mode and dist.get_rank() == 0\n"
    )
    if timed_value_inference in source:
        return

    import_target = "from slime.utils.types import RolloutBatch\n"
    value_inference_target = (
        "        rollout_data.update(forward_only(get_values, self.args, self.model, "
        "data_iterator, num_microbatches))\n"
    )
    if import_target not in source or value_inference_target not in source:
        print(f"WARNING: Could not patch {path} with critic value timing")
        return

    timed_value_inference += (
        "        if _tg_record_critic_value_timing:\n"
        "            _tg_report_step_event(\n"
        '                "value_inference",\n'
        "                self.args,\n"
        "                rollout_id,\n"
        '                "phase_start",\n'
        '                timeline_lane="training",\n'
        '                parent_phase="training",\n'
        '                display_name="Critic value inference",\n'
        "            )\n"
        f"{value_inference_target}"
        "        if _tg_record_critic_value_timing:\n"
        "            _tg_report_step_event(\n"
        '                "value_inference",\n'
        "                self.args,\n"
        "                rollout_id,\n"
        '                "phase_finish",\n'
        '                timeline_lane="training",\n'
        '                parent_phase="training",\n'
        '                display_name="Critic value inference",\n'
        "            )\n"
    )
    source = source.replace(import_target, import_target + reporter, 1)
    source = source.replace(value_inference_target, timed_value_inference, 1)
    path.write_text(source)


if __name__ == "__main__":
    _patch_file(Path("/root/slime/slime/backends/megatron_utils/actor.py"))
