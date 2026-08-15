"""Tutorial source for `010_async_swe_rebench` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "3 x 8xH200",
    "summary": "Fully-async Qwen3.6-27B agent RL on SWE-Rebench V2",
    "difficulty": "Advanced",
    "order": 70,
    "api_classes": [
        "Qwen3_6_27B",
        "Qwen3_6_27b_Recipe",
        "SweEnvironment",
        "SweRebenchV2Dataset",
        "TrainConfig",
        "TrainingRun",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Fully-async SWE agent RL with Qwen3.6-27B

    This tutorial trains a coding agent on
    [SWE-Rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2).
    Each rollout runs stock
    [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) against a
    repository in a fresh Modal Sandbox. The model edits the repository, emits
    a patch, and a second clean sandbox runs the held-out tests.

    The rollout path is **token faithful**: model outputs are sent to SGLang as
    token IDs and the exact returned IDs, log probabilities, loss mask, and
    weight version are recorded for training. Tool observations are context
    (`loss_mask=0`); only model-generated tokens receive gradient.

    Training and rollout use separate nodes. A continuous fully-async pool
    overlaps both phases while bounding the generated-but-unconsumed pool, so
    policy lag cannot grow without limit.

    > This is an advanced multi-node tutorial. The default smoke configuration
    > uses 24 H200 GPUs for one rollout/train step. Multi-node access is required.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run from the repository root:

    ```bash
    uv run python tutorials/rl/010_async_swe_rebench/010_async_swe_rebench.py
    ```

    Set `FULL_RUN=1` to use the research-scale rollout topology and schedule.
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip when using a local editable checkout so local changes remain active.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    import os

    from modal_training_gym import (
        Qwen3_6_27B,
        Qwen3_6_27b_Recipe,
        SweEnvironment,
        SweRebenchV2Config,
        SweRebenchV2Dataset,
        TrainConfig,
    )


@markdown
def _overlay_intro():
    """
    ## Training Gym's native agent stack

    Training Gym provides the complete integration used below:

    - `SweRebenchV2Dataset` prepares the filtered Python/pytest task slice;
    - `SweEnvironment` runs bash in a task image and grades in a second clean
      sandbox;
    - the slime agent adapter runs stock mini-swe-agent while recording exact
      SGLang token IDs, logprobs, loss masks, and weight versions;
    - a Training Gym rollout plugin bounds fully-async staleness and performs
      dynamic sampling; and
    - message-list dataset rows stay compatible with slime's standard processor
      path while the agent retains control of chat-template rendering.

    The only added image dependency is mini-swe-agent itself. The tutorial uses
    Training Gym's pinned slime image without cloning or patching slime.
    """


@code
def _overlay():
    def agentic_swe_image(image):
        return image.uv_pip_install("mini-swe-agent==2.3.0")


@markdown
def _dataset_intro():
    """
    ## Prepare a reproducible mixed-outcome task slice

    GRPO needs variation within each prompt group. We use the public
    `prefilter_ids.json` produced by base-model rollouts and select a small
    Python/pytest subset from the immutable raw SWE-Rebench V2 revision.

    The prepared JSONL stores only the fields the rollout needs. Task images are
    referenced by `image_name`; the dataset does not copy repositories or test
    assets onto the Training Gym volume. The smoke run selects eight tasks;
    `FULL_RUN=1` uses the entire filtered set. Prompts are stored as a single
    user message so they remain compatible with slime's standard data source.
    """


@code
def _dataset():
    def build_dataset(*, full_run: bool = False) -> SweRebenchV2Dataset:
        return SweRebenchV2Dataset(
            config=SweRebenchV2Config(
                n_tasks=None if full_run else 8,
            ),
        )


@markdown
def _async_intro():
    """
    ## Configure fully-async training

    The smoke topology uses:

    - **2 actor nodes**: Qwen3.6-27B full-weight training with TP4 × CP2;
    - **1 rollout node**: four TP2 SGLang engines;
    - **4 prompts × 4 samples** per update; and
    - a pool bounded to two rollout batches.

    `dynamic_sampling_filter_path` removes all-equal-reward groups and keeps
    collecting until the trainer has a useful batch. Watch the unbiased
    `dynamic_sampling/raw_reward_all`, not only the selected batch reward.

    `FULL_RUN=1` expands to four rollout nodes, the research batch sizes, and
    the complete filtered task set. It is intentionally opt-in.
    """


@code
def _config():
    FULL_RUN = os.environ.get("FULL_RUN", "0") == "1"

    def build_training_config(*, full_run: bool = False) -> TrainConfig:
        rollout_nodes = 4 if full_run else 1
        rollout_batch_size = 32 if full_run else 4
        n_samples_per_prompt = 8 if full_run else 4
        num_rollout = 100 if full_run else 1
        max_staleness = 4 if full_run else 2

        recipe = Qwen3_6_27b_Recipe(
            # Non-colocated async topology.
            gpu_type="H200",
            async_mode=True,
            colocate=False,
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=8 * rollout_nodes,
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            decoder_last_pipeline_num_layers=None,
            context_parallel_size=2,
            conversion_tensor_model_parallel_size=4,
            # Convert with the recipe's tested TP4×PP2 layout. torch_dist
            # reshards it into the TP4×PP1 actor layout when training starts.
            conversion_pipeline_model_parallel_size=2,
            ref_load="/checkpoints/Qwen3.6-27B_torch_dist_tp4pp2",
            # Agent rollouts.
            rollout_function=(
                "modal_training_gym.frameworks.slime.bounded_async_rollout."
                "generate_rollout_fully_async"
            ),
            image_overlay=agentic_swe_image,
            capture_trace=True,
            trace_sample_limit=4,
            rm_type=None,
            num_rollout=num_rollout,
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            global_batch_size=rollout_batch_size * n_samples_per_prompt,
            rollout_num_gpus_per_engine=2,
            rollout_max_response_len=16384,
            rollout_temperature=1.0,
            max_tokens_per_gpu=32768,
            # The one-step smoke skips a large checkpoint; full runs save every
            # 20 updates for resume and separate evaluation.
            save_interval=20 if full_run else 1000,
            eval_interval=None,
            dynamic_sampling_filter_path=(
                "slime.rollout.filter_hub.dynamic_sampling_filters."
                "check_reward_nonzero_std"
            ),
            # Launcher and custom-generate settings.
            train_function_kwargs={"ephemeral_disk": 2_097_152},
            environment={
                "PYTHONPATH": "/root/Megatron-LM/:/root/slime:/root",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
                "SLIME_AGENT_SANDBOX_CPU": "2",
                "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
            },
            extra_config={
                "custom_generate_function_path": (
                    "modal_training_gym.frameworks.slime.swe_agent."
                    "generate.generate"
                ),
                "metadata_key": "metadata",
                "rollout_shuffle": True,
                "rollout_max_staleness": max_staleness,
                "rollout_max_context_len": 65536,
                "sglang_server_concurrency": 64 if full_run else 8,
                "sglang_tool_call_parser": "qwen3_coder",
                "sglang_reasoning_parser": "qwen3",
                "use_rollout_logprobs": True,
                "no_check_for_nan_in_loss_and_grad": True,
                "agentic_max_steps": 75 if full_run else 20,
                "agentic_episode_timeout": 1800 if full_run else 900,
                "agentic_exec_timeout": 120,
                "agentic_grade_timeout": 1800,
                "agentic_query_timeout": 600 if full_run else 300,
                "agentic_max_boot_retries": 3,
                "agentic_ramp_window": 30.0 if full_run else 10.0,
                "router_policy": "consistent_hashing",
            },
        )
        return TrainConfig(
            model=Qwen3_6_27B(),
            dataset=build_dataset(full_run=full_run),
            recipe=recipe,
        )

    training_config = build_training_config(full_run=FULL_RUN)


@markdown
def _launch_intro():
    """
    ## Launch detached training

    `launch()` returns after spawning the Modal app. The run survives the
    notebook or local process. Dataset preparation and checkpoint conversion are
    automatic on the first launch and cached for later runs.
    """


@code
def _launch():
    run = training_config.launch(prepare_inputs=True)
    print(f"Training run:  {run.training_run_id}")
    print(f"Modal app:     {run.modal_app_url}")
    print(f"Function call: {run.function_call_id}")


@markdown
def _observe():
    """
    ## Decide whether the run is healthy

    In addition to loss and reward, check:

    - the Training Gym conversation view for sampled rollout traces;
    - `rollout/raw_reward`: reward over the selected training batch;
    - `dynamic_sampling/raw_reward_all`: unbiased reward before filtering;
    - `dynamic_sampling/kept_frac`: whether useful mixed-outcome groups are
      becoming too rare;
    - standard rollout/train timing and loss metrics.

    A rising selected-batch reward with a flat
    `dynamic_sampling/raw_reward_all` is selection bias, not learning.
    """


@markdown
def _evaluation():
    """
    ## Evaluate separately

    Slime's fully-async collector does not support evaluation mode. Keep
    `eval_interval=None`, finish or checkpoint the training run, and evaluate
    checkpoints in a separate deployment/eval job. This prevents evaluation
    from serializing the continuous rollout pool.
    """


@markdown
def _reattach():
    """
    ## Reattach or stop later

    ```python
    from modal_training_gym import TrainingRun

    run = TrainingRun.from_id("<training_run_id>")
    train_result = run.result()

    # Stop the detached run:
    run.function_call.cancel(terminate_containers=True)
    ```
    """
