"""Tutorial source for `001_kimi_k26` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`miles`",
    "cluster_shape": "16 x 8xH200",
    "summary": "Kimi K2.6 LoRA GRPO training on 128 GPUs with DAPO-Math-17k",
    "difficulty": "Advanced",
    "order": 20,
    "api_classes": [
        "Kimi_K2_6",
        "Kimi_K2_6_LoRA_Recipe",
        "TrainConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Multi-node Kimi K2.6 LoRA training

    This tutorial runs LoRA GRPO on
    [Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6), an
    updated Mixture-of-Experts model from Moonshot AI, using
    [miles](https://github.com/volcengine/verl) across **16 nodes
    (128 H200 GPUs)**.

    The `Kimi_K2_6_LoRA_Recipe` preset configures INT4-quantized
    training weights with BF16 reference weights, colocated actor/critic,
    and DeepScaler math reward verification.
    """


@py_only
@markdown
def _run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run tutorials/multinode/001_kimi_k26/001_kimi_k26.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.common.modal_urls import modal_app_dashboard_url
    from modal_training_gym import (
        HuggingFaceDataset,
        Kimi_K2_6,
        Kimi_K2_6_LoRA_Recipe,
        TrainConfig,
    )


@markdown
def _dataset_intro():
    """
    ## Dataset

    We use [DAPO-Math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k),
    a collection of math competition problems with verifiable answers.
    The `deepscaler` reward model checks whether the model's response
    matches the reference answer.
    """


@code
def _define_dataset():
    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_column = ""
        output_column = ""
        input_key = "prompt"
        label_key = "label"
        output_format = "jsonl"
        apply_chat_template = True
        always_prepare = True


@markdown
def _train_intro():
    """
    ## Build and launch training

    Build the training config, construct the Modal app, and spawn
    the training function as a detached call.
    """


@code
def _build_and_run():
    def build_training_config() -> TrainConfig:
        return TrainConfig(
            model=Kimi_K2_6(),
            dataset=MathDataset(n_rows=10),
            recipe=Kimi_K2_6_LoRA_Recipe(),
        )

    training_run = build_training_config()
    app = training_run._build_app()

    with modal.enable_output():
        with app.run():
            modal_app_id = app.app_id or ""
            function_call = app.train.spawn(
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_dashboard_url(modal_app_id),
            )
            print(f"Spawned train function call: {function_call.object_id}")
