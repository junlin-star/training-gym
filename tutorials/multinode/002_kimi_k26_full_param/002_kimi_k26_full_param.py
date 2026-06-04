# Kimi multinode smoke example for Training Gym.

import modal

from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym import (
    HuggingFaceDataset,
    Kimi_K2_6,
    Kimi_K2_6_FullParam_Recipe,
    TrainConfig,
)


tutorial_cli_app = modal.App()


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
        model=Kimi_K2_6(),
        dataset=MathDataset(n_rows=10),
        recipe=Kimi_K2_6_FullParam_Recipe(),
    )


training_run = build_training_config()
app = training_run._build_app()


@tutorial_cli_app.local_entrypoint()
def main() -> None:
    with modal.enable_output():
        with app.run():
            modal_app_id = app.app_id or ""
            function_call = app.train.spawn(
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_dashboard_url(modal_app_id),
            )
            print(f"Spawned train function call: {function_call.object_id}")


if __name__ == "__main__":
    print(type(training_run.model).__name__)
    print(type(training_run.recipe).__name__)
    print(sorted(app.registered_functions))
    print(training_run.recipe.cli_args(dataset=training_run.dataset, model=training_run.model)[:20])
