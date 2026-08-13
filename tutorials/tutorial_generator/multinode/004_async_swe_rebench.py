"""Tutorial source for `004_async_swe_rebench` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "3 x 8xH200",
    "summary": "Fully-async Qwen3.6-27B agent RL on SWE-Rebench V2",
    "difficulty": "Advanced",
    "order": 32,
    "api_classes": [
        "DatasetConfig",
        "Qwen3_6_27B",
        "Qwen3_6_27b_Recipe",
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
    uv run python tutorials/multinode/004_async_swe_rebench/004_async_swe_rebench.py
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
    import json
    import os
    from pathlib import Path

    from modal_training_gym import (
        DatasetConfig,
        Qwen3_6_27B,
        Qwen3_6_27b_Recipe,
        TrainConfig,
    )


@markdown
def _overlay_intro():
    """
    ## Pin the agentic slime overlay

    Training Gym pins its base slime image by digest. The agent loop and four
    focused slime changes live in a separate public slime commit:

    - the `agentic_rl` package;
    - a staleness-bounded fully-async collector with dynamic sampling;
    - full-group regeneration after a weight-update abort;
    - neutral handling of fully masked samples during GRPO normalization; and
    - correct text-only loading for Qwen3.6.

    The image overlay copies only `agentic_rl/` and applies
    `agentic_rl/training_gym.patch`. It does **not** replace `/root/slime`, so
    Training Gym's built-in compatibility patches remain intact.

    The commit is immutable. Override `SLIME_AGENTIC_REF` only when testing a
    newer compatible commit.
    """


@code
def _overlay():
    SLIME_OVERLAY_REPO = "https://github.com/modal-projects/slime.git"
    SLIME_OVERLAY_REF = os.environ.get(
        "SLIME_AGENTIC_REF",
        "a03d399266a49280c286d87160340b3a1816878e",
    )

    def agentic_slime_overlay(image):
        checkout = "/tmp/training-gym-agentic-slime"
        return image.run_commands(
            (
                f"rm -rf {checkout} && "
                f"git init {checkout} && "
                f"git -C {checkout} remote add origin {SLIME_OVERLAY_REPO} && "
                f"git -C {checkout} fetch --depth 1 origin {SLIME_OVERLAY_REF} && "
                f"git -C {checkout} checkout --detach FETCH_HEAD"
            ),
            f"cp -R {checkout}/agentic_rl /root/slime/agentic_rl",
            (
                f"cd /root/slime && "
                f"git apply --check {checkout}/agentic_rl/training_gym.patch && "
                f"git apply {checkout}/agentic_rl/training_gym.patch"
            ),
            "uv pip install --system mini-swe-agent==2.3.0",
            f"rm -rf {checkout}",
        )


@markdown
def _dataset_intro():
    """
    ## Prepare a reproducible mixed-outcome task slice

    GRPO needs variation within each prompt group. We use the public
    `prefilter_ids.json` produced by base-model rollouts and select a small
    Python/pytest subset from the immutable raw SWE-Rebench V2 revision.

    The prepared JSONL stores only the fields the rollout needs. Task images are
    referenced by `image_name`; the dataset does not copy repositories or test
    assets onto the Training Gym volume.
    """


@code
def _dataset():
    RAW_DATASET = "nebius/SWE-rebench-V2"
    RAW_DATASET_REVISION = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
    PREFILTER_DATASET = "junlin-modal/swe-rebench-v2"
    PREFILTER_REVISION = "f2cf9141b7fff2febb748d4859b9b5d0d2aacda1"
    SUPPORTED_PARSERS = {
        "parse_log_pytest",
        "parse_log_pytest_options",
        "parse_log_pytest_v2",
    }

    class SweRebenchV2TutorialDataset(DatasetConfig):
        input_key = "prompt"
        label_key = "label"
        output_format = "jsonl"
        apply_chat_template = False
        writes_eval_paths = False

        def __init__(self, n_tasks: int = 8):
            self.n_tasks = n_tasks
            super().__init__(dataset_id=f"swe-rebench-v2-agentic-{n_tasks}")

        def prepare(
            self,
            path: str,
            eval_paths: dict[str, str] | None = None,
        ) -> None:
            from datasets import load_dataset
            from huggingface_hub import hf_hub_download

            ids_path = hf_hub_download(
                PREFILTER_DATASET,
                "swe_rebench_v2/prefilter_ids.json",
                repo_type="dataset",
                revision=PREFILTER_REVISION,
            )
            mixed_outcome_ids = set(
                json.loads(Path(ids_path).read_text())["instance_ids"]
            )

            rows = []
            source = load_dataset(
                RAW_DATASET,
                split="train",
                revision=RAW_DATASET_REVISION,
                streaming=True,
            )
            for raw in source:
                install_config = raw.get("install_config") or {}
                if raw["instance_id"] not in mixed_outcome_ids:
                    continue
                if (raw.get("language") or "").lower() != "python":
                    continue
                if install_config.get("log_parser") not in SUPPORTED_PARSERS:
                    continue
                if not raw.get("FAIL_TO_PASS") or not raw.get("test_patch"):
                    continue

                metadata = {
                    "task_type": "swerebench",
                    "instance_id": raw["instance_id"],
                    "image_name": raw["image_name"],
                    "repo": raw["repo"],
                    "install_config": install_config,
                    "test_patch": raw["test_patch"],
                    "FAIL_TO_PASS": list(raw["FAIL_TO_PASS"]),
                    "PASS_TO_PASS": list(raw["PASS_TO_PASS"]),
                    "problem_statement": raw["problem_statement"],
                }
                rows.append(
                    {
                        "prompt": raw["problem_statement"],
                        "label": raw["instance_id"],
                        "metadata": metadata,
                    }
                )
                if len(rows) >= self.n_tasks:
                    break

            if len(rows) < self.n_tasks:
                raise RuntimeError(
                    f"Only found {len(rows)}/{self.n_tasks} compatible tasks"
                )

            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

    dataset = SweRebenchV2TutorialDataset(n_tasks=8)


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

    `FULL_RUN=1` expands to four rollout nodes and the research batch sizes.
    It is intentionally opt-in.
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
            conversion_pipeline_model_parallel_size=1,
            ref_load="/checkpoints/Qwen3.6-27B_torch_dist_tp4pp1",
            # Agent rollouts.
            rollout_function=(
                "slime.rollout.fully_async_rollout."
                "generate_rollout_fully_async"
            ),
            image_overlay=agentic_slime_overlay,
            rm_type=None,
            num_rollout=num_rollout,
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            global_batch_size=rollout_batch_size * n_samples_per_prompt,
            rollout_num_gpus_per_engine=2,
            rollout_max_response_len=8192,
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
                "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
                "SLIME_AGENT_SANDBOX_CPU": "2",
                "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
            },
            extra_config={
                "custom_generate_function_path": "agentic_rl.generate.generate",
                "training_gym_custom_rollout_log_function_path": (
                    "agentic_rl.metrics.log_rollout_data"
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
            dataset=dataset,
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
    run = training_config.launch()
    print(f"Training run:  {run.training_run_id}")
    print(f"Modal app:     {run.modal_app_url}")
    print(f"Function call: {run.function_call_id}")


@markdown
def _observe():
    """
    ## Decide whether the run is healthy

    In addition to loss and reward, check:

    - `dynamic_sampling/raw_reward_all`: unbiased reward before filtering;
    - `dynamic_sampling/kept_frac`: whether useful mixed-outcome groups are
      becoming too rare;
    - `async/version_lag/max`: maximum behavior-policy lag in the trained batch;
    - `agentic/removed_frac`: sandbox/image failures removed from gradient;
    - `agentic/exec_timeouts/mean`: hung commands cut off by the client deadline;
    - `agentic/truncated_frac`: episodes stopped by turn, context, or time limits.

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
