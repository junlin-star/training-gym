"""Smoke-test launch script for Qwen3.6-35B-A3B miles recipe."""

import modal

from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym import (
    HuggingFaceDataset,
    Qwen3_6_35B,
    Qwen3_6_35B_A3B_Recipe,
    TrainConfig,
)


app = modal.App()


class MathDataset(HuggingFaceDataset):
    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = ""
    output_column = ""
    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True


def build_training_config() -> TrainConfig:
    return TrainConfig(
        model=Qwen3_6_35B(),
        dataset=MathDataset(n_rows=10),
        recipe=Qwen3_6_35B_A3B_Recipe(),
    )


training_run = build_training_config()
training_app = training_run._build_app()


@app.local_entrypoint()
def main() -> None:
    with modal.enable_output():
        with training_app.run():
            modal_app_id = training_app.app_id or ""
            result = training_app.train.remote(
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_dashboard_url(modal_app_id),
            )
            print(f"Train result: {result}")
