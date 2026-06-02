"""Tutorial source for `000_rl_basics_miles` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`miles`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Qwen3-4B GSM8K math reasoning with Miles — train with built-in math reward",
    "difficulty": "Beginner",
    "order": 11,
    "api_classes": [
        "Qwen3_4B",
        "TrainConfig",
        "MilesConfig",
    ],
}


from tutorial_generator import code, markdown, py_only, notebook_only, shell


@markdown
def _intro():
    """
    # RL basics with Miles: GSM8K math reasoning

    This tutorial trains Qwen3-4B on the GSM8K math-reasoning
    benchmark using [Miles](https://github.com/radix-ai/miles) on Modal.

    Miles provides a built-in `rm_type="math"` reward model that
    automatically extracts the final numeric answer from each
    response (the `#### <number>` pattern) and compares it to the
    ground-truth label — no custom reward function needed.

    **What you'll do:**

    1. Define a GSM8K dataset.
    2. Configure a Miles GRPO training run.
    3. Build the Modal app and inspect the registered functions.
    """


@py_only
@markdown
def run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run modal run tutorials/rl/000_rl_basics_miles/000_rl_basics_miles.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        HuggingFaceDataset,
        MilesConfig,
        Qwen3_4B,
        TrainConfig,
    )


@markdown
def _dataset_intro():
    """
    ## Define the GSM8K dataset

    GSM8K is a dataset of grade-school math word problems. Each row
    has a `question` column and an `answer` column. The answer ends
    with `#### <number>` — the ground-truth numeric result.

    We use `HuggingFaceDataset` to point at the dataset, map the
    columns, and set up chat-template formatting.
    """


@code
def _define_dataset():
    class GSM8KDataset(HuggingFaceDataset):
        hf_repo = "openai/gsm8k"
        hf_split = "train"
        input_column = "question"
        output_column = "answer"
        output_format = "parquet"
        apply_chat_template = True
        system_prompt = (
            "You are a helpful math tutor. Solve the problem step by step, "
            "then give the final numeric answer after ####."
        )
        prompt_template = "{input}"
        always_prepare = True

    train_dataset = GSM8KDataset(n_rows=100)


@markdown
def _config_intro():
    """
    ## Configure the Miles training run

    `MilesConfig` mirrors `SlimeRecipe` — it accepts the same GRPO
    algorithm knobs but launches Miles under the hood.

    The key setting here is `rm_type="math"` — this tells Miles to
    use its built-in math verifier that extracts `#### <number>`
    from model responses and compares against the label. No custom
    reward function is needed.
    """


@code
def _define_training_run():
    base_model = Qwen3_4B()

    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=MilesConfig(
            rm_type="math",

            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,

            num_rollout=10,
            rollout_batch_size=16,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,

            save_interval=5,
            apply_chat_template_kwargs={"enable_thinking": False},
        ),
    )


@markdown
def _build_app_intro():
    """
    ## Build and inspect the Modal app

    `TrainConfig._build_app()` wires up the download, dataset
    preparation, and training functions as a Modal app. We print the
    registered functions to verify everything is configured correctly.
    """


@code
def _build_app():
    app = training_run._build_app()


@py_only
@code
def _inspect():
    if __name__ == "__main__":
        print(type(training_run.recipe).__name__)
        print(sorted(app.registered_functions))
        print(
            training_run.recipe.cli_args(
                dataset=training_run.dataset, model=training_run.model
            )[:12]
        )
