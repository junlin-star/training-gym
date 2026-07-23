from pathlib import Path


MARKER = "PATCHED_TRAINING_GYM_TRAINING_SUBSTEP_TIMING"


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping training substep timing patch")
        return

    source = path.read_text()
    if MARKER in source:
        return

    import_target = "from slime.utils.types import RolloutBatch\n"
    train_target = "    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):\n"
    wake_target = (
        "        if self.args.offload_train:\n"
        "            self.wake_up()\n\n"
        '        with timer("data_preprocess"):\n'
    )
    preprocess_finish_target = (
        "            rollout_data = self._get_rollout_data(rollout_data_ref)\n\n"
        '        if self.role == "critic":\n'
    )
    offload_target = (
        "        if self.args.offload_train:\n"
        "            del rollout_data\n"
        "            self.sleep()\n\n"
        "        return result\n"
    )
    value_target = (
        "        rollout_data.update(forward_only(get_values, self.args, self.model, "
        "data_iterator, num_microbatches))\n"
    )
    reference_update_target = (
        '            with timer("ref_model_update"):\n'
        "                if is_megatron_main_rank():\n"
        '                    logger.info(f"Updating ref model at rollout_id {rollout_id}")\n'
        '                self.weights_backuper.backup("ref")\n'
    )
    targets = {
        "reporter import": import_target,
        "training entrypoint": train_target,
        "training-model wake": wake_target,
        "data preprocessing finish": preprocess_finish_target,
        "training-model offload": offload_target,
        "critic value inference": value_target,
        "reference-model update": reference_update_target,
    }
    missing = [name for name, target in targets.items() if target not in source]
    if missing:
        print(f"WARNING: Could not patch {path} for: {', '.join(missing)}")
        return

    reporter = (
        f"# {MARKER}\n"
        "from modal_training_gym.frameworks.slime.phase_reporting import (\n"
        "    flush_async_timing_events as _tg_flush_timings,\n"
        "    flush_async_timing_queue_before_reraise as _tg_flush_timings_on_error,\n"
        "    record_async_phase_interval as _tg_record_interval,\n"
        ")\n"
    )
    wake = (
        "        if self.args.offload_train:\n"
        "            with _tg_record_interval(\n"
        '                "training_model_wake", self.args, rollout_id,\n'
        '                timeline_lane="coordination", parent_phase="training",\n'
        '                display_name="Load training model",\n'
        "            ):\n"
        "                self.wake_up()\n\n"
        "        with _tg_record_interval(\n"
        '            "data_preprocess", self.args, rollout_id,\n'
        '            timeline_lane="training", parent_phase="training",\n'
        '            display_name="Load & transfer training batch",\n'
        "        ):\n"
        '            with timer("data_preprocess"):\n'
    )
    preprocess_finish = (
        "                rollout_data = self._get_rollout_data(rollout_data_ref)\n\n"
        '        if self.role == "critic":\n'
    )
    offload = (
        "        if self.args.offload_train:\n"
        "            del rollout_data\n"
        "            with _tg_record_interval(\n"
        '                "training_model_offload", self.args, rollout_id,\n'
        '                timeline_lane="coordination", parent_phase="training",\n'
        '                display_name="Offload training model",\n'
        "            ):\n"
        "                self.sleep()\n\n"
        "        if rollout_id == self.args.num_rollout - 1:\n"
        "            _tg_flush_timings()\n"
        "        return result\n"
    )
    value_inference = (
        "        with _tg_record_interval(\n"
        '            "value_inference", self.args, rollout_id,\n'
        '            timeline_lane="training", parent_phase="training",\n'
        '            display_name="Critic value inference",\n'
        "        ):\n"
        "            rollout_data.update(\n"
        "                forward_only(\n"
        "                    get_values, self.args, self.model, data_iterator, num_microbatches\n"
        "                )\n"
        "            )\n"
    )
    reference_update = (
        "            with _tg_record_interval(\n"
        '                "reference_model_update", self.args, rollout_id,\n'
        '                timeline_lane="training", parent_phase="training",\n'
        '                display_name="Update reference model",\n'
        "            ):\n"
        '                with timer("ref_model_update"):\n'
        "                    if is_megatron_main_rank():\n"
        '                        logger.info(f"Updating ref model at rollout_id {rollout_id}")\n'
        '                    self.weights_backuper.backup("ref")\n'
    )

    source = source.replace(import_target, import_target + reporter, 1)
    source = source.replace(
        train_target,
        "    @_tg_flush_timings_on_error\n"
        "    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):\n"
        "        self.args.training_gym_role = self.role\n",
        1,
    )
    source = source.replace(wake_target, wake, 1)
    source = source.replace(preprocess_finish_target, preprocess_finish, 1)
    source = source.replace(offload_target, offload, 1)
    source = source.replace(value_target, value_inference, 1)
    source = source.replace(reference_update_target, reference_update, 1)
    path.write_text(source)


if __name__ == "__main__":
    _patch_file(Path("/root/slime/slime/backends/megatron_utils/actor.py"))
