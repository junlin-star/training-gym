# Kimi K2.5 multinode smoke example for Training Gym.

from modal_training_gym import (
    HuggingFaceDataset,
    Kimi_K2_5,
    Kimi_K2_5_Recipe,
    TrainConfig,
)


class MathDataset(HuggingFaceDataset):
    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = "prompt"
    output_column = "label"
    output_format = "jsonl"
    apply_chat_template = False


def build_training_config() -> TrainConfig:
    return TrainConfig(
        model=Kimi_K2_5(),
        dataset=MathDataset(n_rows=10),
        recipe=Kimi_K2_5_Recipe(),
    )


training_run = build_training_config()
app = training_run._build_app()


if __name__ == "__main__":
    print(type(training_run.model).__name__)
    print(type(training_run.recipe).__name__)
    print(sorted(app.registered_functions))
    print(training_run.recipe.cli_args(dataset=training_run.dataset, model=training_run.model)[:20])
