"""Lazily constructed text-only slime data source."""

from __future__ import annotations


def _build_text_data_source():
    from pathlib import Path

    from slime.rollout.data_source import RolloutDataSourceWithBuffer, pop_first
    from slime.utils.data import Dataset
    from slime.utils.misc import load_function
    from slime.utils.processing_utils import load_tokenizer

    class TextOnlyRolloutDataSourceWithBuffer(RolloutDataSourceWithBuffer):
        """Buffered source that deliberately skips an HF processor."""

        def __init__(self, args) -> None:
            self.args = args
            self.epoch_id = 0
            self.sample_group_index = 0
            self.sample_index = 0
            self.sample_offset = 0
            self.metadata = {}

            if args.rollout_global_dataset and args.prompt_data is not None:
                tokenizer = load_tokenizer(
                    args.hf_checkpoint,
                    trust_remote_code=True,
                )
                if (details_dir := args.dump_details) is not None:
                    tokenizer.save_pretrained(Path(details_dir) / "tokenizer")
                self.dataset = Dataset(
                    args.prompt_data,
                    tokenizer=tokenizer,
                    processor=None,
                    max_length=args.rollout_max_prompt_len,
                    prompt_key=args.input_key,
                    multimodal_keys=None,
                    label_key=args.label_key,
                    metadata_key=args.metadata_key,
                    tool_key=args.tool_key,
                    apply_chat_template=args.apply_chat_template,
                    apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                    seed=args.rollout_seed,
                )
                if args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
            else:
                self.dataset = None

            self.buffer = []
            self.buffer_filter = (
                pop_first
                if args.buffer_filter_path is None
                else load_function(args.buffer_filter_path)
            )

    return TextOnlyRolloutDataSourceWithBuffer


def __getattr__(name: str):
    if name != "TextOnlyRolloutDataSourceWithBuffer":
        raise AttributeError(name)
    value = _build_text_data_source()
    globals()[name] = value
    return value
