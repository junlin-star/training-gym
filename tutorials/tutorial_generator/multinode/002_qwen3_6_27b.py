"""Tutorial source for `002_qwen3_6_27b` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "2 x 8xH100",
    "summary": "Qwen3.6-27B full-weight GRPO training with slime on 16 GPUs",
    "difficulty": "Advanced",
    "order": 30,
    "api_classes": [
        "Qwen3_6_27B",
        "Qwen3_6_27b_Recipe",
        "Qwen3_6_27b_SglangRecipe",
        "DeploymentConfig",
        "TrainConfig",
        "TrainResult",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Multi-node Qwen3.6-27B full-weight training

    This tutorial runs full-weight GRPO on
    [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), a dense
    27B-parameter hybrid-attention model, using
    [slime](https://github.com/THUDM/slime) across **2 nodes (16 H100 GPUs)**.

    Qwen3.6-27B is a hybrid model: most layers use Gated DeltaNet
    linear attention, with periodic full Gated Attention layers. The
    `Qwen3_6_27B` model preset carries the architecture and Megatron
    model type needed for slime, so Training Gym automatically performs
    the HF-to-Megatron checkpoint pre-conversion before training starts.

    The `Qwen3_6_27b_Recipe` defaults are tuned for a single 8×H100
    node. In this guide, we override `actor_num_nodes=2` so the same
    TP4×PP2 model-parallel layout runs with an additional data-parallel
    replica across the second node.
    """


@py_only
@markdown
def _run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run python tutorials/multinode/002_qwen3_6_27b/002_qwen3_6_27b.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        DeploymentConfig,
        HuggingFaceDataset,
        Qwen3_6_27B,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.deploy_recipes.sglang_recipe import Qwen3_6_27b_SglangRecipe
    from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_27b_Recipe


@markdown
def _dataset_intro():
    """
    ## Dataset

    We use [DAPO-Math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k),
    a collection of math problems with verifiable answers. The `deepscaler`
    reward model built into slime extracts the final numerical answer and
    compares it to the reference label.

    The tutorial uses a small subset so you can validate the training path
    before launching a longer run.
    """


@code
def _define_dataset():
    class DAPOMath(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_key = "prompt"
        label_key = "label"
        output_format = "jsonl"
        apply_chat_template = True
        always_prepare = True

    dataset = DAPOMath(n_rows=120)


@markdown
def _model_intro():
    """
    ## Model and recipe

    `Qwen3_6_27B()` points at `Qwen/Qwen3.6-27B` on HuggingFace and
    includes the model architecture used by slime's Megatron integration.

    `Qwen3_6_27b_Recipe()` supplies the dense 27B slime defaults:
    TP=4, PP=2, sequence parallelism, colocated SGLang rollout, and
    CPU-offloaded optimizer states. Setting `actor_num_nodes=2` makes
    this a 16-GPU multinode run while keeping the validated per-replica
    model-parallel shape.
    """


@code
def _define_model():
    model = Qwen3_6_27B()

    recipe = Qwen3_6_27b_Recipe(
        actor_num_nodes=2,
        rm_type="deepscaler",
        n_samples_per_prompt=4,
        global_batch_size=128,
        sglang_max_running_requests=512,
        train_function_kwargs={"ephemeral_disk": 2_097_152},
    )

    print(f"Model: {model.model_name}")
    print(f"Nodes: {recipe.actor_num_nodes}, GPUs/node: {recipe.actor_num_gpus_per_node}")
    print(
        f"Parallelism: TP={recipe.tensor_model_parallel_size}, "
        f"PP={recipe.pipeline_model_parallel_size}, "
        f"CP={recipe.context_parallel_size}"
    )
    print(f"Rollout engine GPUs: {recipe.rollout_num_gpus_per_engine}")
    print(f"Algorithm: {recipe.advantage_estimator}")


@markdown
def _train_intro():
    """
    ## Train

    `TrainConfig.train()` builds the Modal app, downloads the model
    weights, prepares the dataset, pre-converts the checkpoint to
    Megatron format, and launches slime on the multinode Ray cluster.

    The first run downloads the model into the shared HuggingFace cache
    volume and writes converted checkpoints to the training checkpoint
    volume. Subsequent runs reuse those volumes.
    """


@code
def _run_training():
    training_run = TrainConfig(
        model=model,
        dataset=dataset,
        recipe=recipe,
    )

    print(f"Training run: {training_run.training_run_id}")
    print(f"Total nodes: {recipe.total_nodes}")
    print("--- Starting training... ---")
    train_result = training_run.train()
    print("--- Training complete ---")


@markdown
def _serve_intro():
    """
    ## Serve the trained checkpoint

    After training, serve the checkpoint with SGLang for inference.
    `Qwen3_6_27b_SglangRecipe` defaults to 4×H100 with TP=4.
    """


@code
def _serve_checkpoint():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    deployment = DeploymentConfig(
        model=Qwen3_6_27B(),
        checkpoint=checkpoint,
        recipe=Qwen3_6_27b_SglangRecipe(),
        app_name="qwen3-6-27b-multinode-serve",
        served_model_name="qwen3-6-27b-multinode",
    ).serve()
    print(f"Deployed to {deployment.url}")


@notebook_only
@markdown
def _try_it():
    """
    Let's test the trained model with a math problem.
    """


@notebook_only
@code
def _try_generation():
    response = deployment.generate(
        "Let $p$ be a prime number. Find the number of integers $n$ "
        "with $1 \\le n \\le p^2$ such that $n^{p-1} \\equiv 1 \\pmod{p^2}.",
    )
    print(response)
