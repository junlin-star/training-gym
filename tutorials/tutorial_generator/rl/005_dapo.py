# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `005_dapo` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "DAPO on math",
    "difficulty": "Advanced",
    "order": 35,
    "api_classes": [
        "Qwen3_4B",
        "HuggingFaceDataset",
        "Sample",
        "SlimeRecipe",
        "TrainConfig",
        "endpoint_chat",
        "wait_for_server_url",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)

    This tutorial trains **Qwen3-4B** according to DAPO as presented in Yu et al.,
    2025 on the provided dataset `zhuzilin/dapo-math-17k`.

    DAPO presents four changes to the vanilla GRPO recipe aimed to improve long
    chain-of-thought RL.

    1. **Clip-Higher** addresses entropy collapse by using asymmetric clipping,
       with the upper clip loosened (`1 + ε_high`) so tokens with positive
       advantage can be reinforced more aggressively while the lower clip stays
       the same for stability.
    2. **Dynamic Sampling** over-samples prompts and drops batches with the same
       reward (no variance → no advantage → no gradient update) to create
       higher signal for each batch.
    3. **Token-level policy-gradient loss** calculates average loss over tokens
       instead of sequences, so long correct answers aren't down-weighted.
    4. **Overlong Reward Shaping** applies a soft length penalty so good
       reasoning is not confused with answers that are simply too long.

    DAPO also **removes the KL penalty** (`use_kl_loss=False`) because the long
    CoT reasoning model is expected to diverge significantly during training.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run locally (your machine drives the Modal GPU workers):

    ```
    cd training-gym
    uv sync
    uv run tutorials/rl/005_dapo/005_dapo.py
    ```

    To detach and watch it from the Modal dashboard instead:

    ```
    uv run modal run -d tutorials/rl/005_dapo/005_dapo.py
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
    import re
    from typing import Any

    import modal

    from modal_training_gym import (
        HuggingFaceDataset,
        Qwen3_4B,
        Sample,
        SlimeRecipe,
        TrainConfig,
        endpoint_chat,
        wait_for_server_url,
    )


@markdown
def _dataset_intro():
    """
    ## Dataset

    We use the same math dataset as the original paper: competition math problems
    where the model is asked to answer with `Answer: \\boxed{N}`.

    [Here's the link to `zhuzilin/dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)

    Each row contains a `prompt` field with the original chat message and a
    `label` containing the integer answer. Since the prompt is already stored as
    chat messages, we point `input_key` directly at `prompt` and let slime apply
    the model's chat template during training.

    For this tutorial, we train on 2,000 prompts and hold out 100 prompts for
    checking, using a row offset so the two splits never overlap.
    """


@code
def _dataset():
    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_key = "prompt"
        label_key = "label"
        output_format = "jsonl"
        apply_chat_template = True
        row_offset = 0
        always_prepare = True

        def load(self, split: str = "all") -> Any:
            from datasets import load_dataset

            ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
            start = min(self.row_offset, len(ds))
            stop = len(ds) if not self.n_rows else min(start + self.n_rows, len(ds))
            return ds.select(range(start, stop))


@code
def _make_datasets():
    train_dataset = MathDataset(n_rows=2_000)
    check_dataset = MathDataset(n_rows=100, row_offset=16_000)


@notebook_only
@code
def _dataset_peek():
    rows = check_dataset.load()
    for row in rows.select(range(2)):
        prompt = row["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]["content"] if prompt else ""
        print(prompt[:200])
        print(f"  label: {row['label']}")
        print()


@code
def _check_helpers():
    def _normalize_answer(answer: str) -> str:
        answer = str(answer).strip()
        answer = answer.split("=")[-1]
        for old, new in [("$", ""), ("\\$", ""), (",", ""), (" ", ""),
                          ("\\text{", ""), ("}", ""), ("\\boxed{", "")]:
            answer = answer.replace(old, new)
        return answer.strip()

    def _extract_answer(response: str) -> str:
        match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response)
        return match[-1].strip() if match else "[INVALID]"

    def _check_math(response: str, label: str) -> bool:
        pred = _normalize_answer(_extract_answer(response))
        gt = _normalize_answer(label)
        try:
            gt = str(int(float(gt)))
        except (ValueError, OverflowError):
            pass
        return pred == gt

    def score_math_example(base_url: str, model_id: str, example: dict) -> Sample:
        prompt = example.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0]["content"] if prompt else ""
        label = example.get("label", "")

        response = endpoint_chat(
            base_url,
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            proxy_auth=True,
        )

        correct = _check_math(response, label)
        pred = _normalize_answer(_extract_answer(response))

        return Sample(
            score=1.0 if correct else 0.0,
            prompt=prompt,
            response=response,
            metadata={"correct": correct, "pred": pred, "label": label},
        )

    def run_math_check(base_url: str, model_id: str) -> list[Sample]:
        return [
            score_math_example(base_url, model_id, example)
            for example in check_dataset.load()
        ]

    def mean_score(rows: list[Sample]) -> float:
        return sum(row.score for row in rows) / len(rows) if rows else float("nan")


@markdown
def _check_base_intro():
    """
    ## Baseline check

    Qwen3-4B is not in the managed Endpoint catalog, so this follows the ASR
    tutorial's pattern: Modal `@app.server` code launches SGLang, then a plain
    Python loop scores the held-out rows.

    Custom `@app.server` endpoints here use `unauthenticated=False`. Run
    `training-gym set-proxy-auth` or export `MODAL_KEY`/`MODAL_SECRET` before
    calling `wait_for_server_url` / `endpoint_chat` with `proxy_auth=True`.
    """


@code
def _check_base():
    base_model = Qwen3_4B()
    MODEL_ID = base_model.model_name
    SERVER_PORT = 8000
    SERVER_STARTUP_TIMEOUT = 20 * 60

    server_image = (
        modal.Image.from_registry("lmsysorg/sglang:v0.5.12")
        .entrypoint([])
        .run_commands("rm -rf /root/.cache/huggingface")
        .env({"HF_HUB_CACHE": "/root/.cache/huggingface"})
    )

    def serve_model(
        model_path: str,
        served_model_name: str,
        app_name: str,
        checkpoints_volume_name: str | None = None,
    ) -> str:
        app = modal.App(app_name)
        volumes = {
            "/root/.cache/huggingface": modal.Volume.from_name(
                "huggingface-cache", create_if_missing=True
            )
        }
        if checkpoints_volume_name:
            volumes["/checkpoints"] = modal.Volume.from_name(
                checkpoints_volume_name, create_if_missing=True
            )

        @app.server(
            image=server_image,
            gpu="H100",
            volumes=volumes,
            port=SERVER_PORT,
            startup_timeout=SERVER_STARTUP_TIMEOUT,
            scaledown_window=10 * 60,
            exit_grace_period=25,
            target_concurrency=4,
            unauthenticated=False,
            serialized=True,
        )
        class ModelServer:
            @modal.enter()
            def start(self):
                import subprocess as _sp
                import time as _time
                import urllib.error as _ue
                import urllib.request as _ur

                self.proc = _sp.Popen(
                    [
                        "python",
                        "-m",
                        "sglang.launch_server",
                        "--model-path",
                        model_path,
                        "--served-model-name",
                        served_model_name,
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(SERVER_PORT),
                        "--mem-fraction-static",
                        "0.80",
                        "--trust-remote-code",
                    ]
                )
                deadline = _time.monotonic() + SERVER_STARTUP_TIMEOUT
                health = f"http://127.0.0.1:{SERVER_PORT}/health"
                while True:
                    if self.proc.poll() is not None:
                        raise RuntimeError(
                            f"SGLang exited with code {self.proc.returncode} "
                            "before healthy"
                        )
                    try:
                        with _ur.urlopen(health, timeout=5) as response:
                            if response.status == 200:
                                return
                    except (_ue.URLError, TimeoutError, OSError):
                        pass
                    if _time.monotonic() >= deadline:
                        raise TimeoutError(f"SGLang not healthy at {health}")
                    _time.sleep(2)

            @modal.exit()
            def stop(self):
                proc = getattr(self, "proc", None)
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)

        with modal.enable_output():
            app.deploy()
        return wait_for_server_url(ModelServer, label="Qwen3-4B check server", proxy_auth=True)

    base_url = serve_model(MODEL_ID, MODEL_ID, "gym-qwen3-4b-dapo-check-base")
    print(f"Base model URL: {base_url}")

    print("--- Checking base model... ---")
    base_rows = run_math_check(base_url, MODEL_ID)
    base_mean = mean_score(base_rows)
    n_correct = sum(1 for row in base_rows if row.metadata.get("correct"))
    print(f"Base accuracy: {n_correct}/{len(base_rows)} ({base_mean:.1%})")


@notebook_only
@code
def _base_examples():
    for r in base_rows[:3]:
        status = "CORRECT" if r.metadata["correct"] else "WRONG"
        print(f"[{status}] label={r.metadata['label']}, pred={r.metadata['pred']}")
        print(f"  ...{r.response[-150:]}")
        print()


@markdown
def _rm_intro():
    """
    ## Reward function

    The correctness term comes from slime's DAPO math scorer. It gives +1 for
    correct answers and -1 for incorrect answers.

    DAPO also adds a soft penalty for overlong responses. For speed, we use a
    smaller response cap than the paper:

    ```text
    R_length = 0                                      if |y| <= L_max - L_cache
             = ((L_max - L_cache) - |y|) / L_cache   if L_max - L_cache < |y| <= L_max
             = -1                                     if |y| > L_max
    ```

    The goal is that the model can still get credit for correct reasoning while
    learning to finish within the token budget. We set `L_max` to the generation
    cap, 8192, and `L_cache` to 2048, matching the paper's 4:1 ratio
    (16384/4096) scaled for this tutorial.
    """

@code
def _reward_helpers():
    async def dapo_overlong_rm(args, sample, **kwargs) -> float:
        from slime.rollout.rm_hub.math_dapo_utils import (
            compute_score as compute_score_dapo,
        )

        payload = compute_score_dapo(sample.response, sample.label)
        base = float(payload["score"] if isinstance(payload, dict) else payload)

        L_max = args.rollout_max_response_len
        L_cache = 2048
        n = sample.response_length

        if n <= L_max - L_cache:
            length_penalty = 0.0
        elif n <= L_max:
            length_penalty = ((L_max - L_cache) - n) / L_cache
        else:
            length_penalty = -1.0

        return base + length_penalty


@markdown
def _train_intro():
    """
    ## Training

    The recipe below is slime's reference Qwen3-4B layout (TP=2, 8192-token
    responses, `max_tokens_per_gpu=9216`) with the DAPO modifications  on
    top of GRPO. We follow ([the paper's recipe](https://arxiv.org/abs/2503.14476)) for the most part, 
    but with some modifications for speed:

    Mentioned in the paper:
    - **Clip-Higher** (`eps_clip=0.2`, `eps_clip_high=0.28`)  
    - **No KL penalty** (`use_kl_loss=False`, `kl_coef=0.0`)  
    - **Token-level policy-gradient loss** (`calculate_per_token_loss=True`)  
    - **Dynamic sampling** (`over_sampling_batch_size=48` plus the zero-variance reward filter)

    Modified for speed:
    - **8 samples per prompt** (`n_samples_per_prompt=8`) 
    - **Overlong buffer of 2048 tokens** (`L_cache=2048`)
    - **8192-token response cap** (`L_max=8192`)

    This tutorial runs a short 15-rollout job to demonstrate the DAPO training setup.
    For a more meaningful accuracy gain, increase the rollout count.
    """


@code
def _train():
    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            rm_type="dapo",
            custom_rm_function=dapo_overlong_rm,
            gpu_type="H100",
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            rollout_num_gpus_per_engine=1,
            num_rollout=15,
            rollout_batch_size=16,
            n_samples_per_prompt=8,
            rollout_max_response_len=8192,
            rollout_temperature=1.0,
            rollout_shuffle=True,
            global_batch_size=32,
            lr=1e-6,
            lr_decay_style="constant",
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.98,
            optimizer="adam",
            advantage_estimator="grpo",
            use_kl_loss=False,
            kl_loss_type="low_var_kl",
            kl_loss_coef=0.0,
            entropy_coef=0.0,
            eps_clip=0.2,
            eps_clip_high=0.28,
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=9216,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            accumulate_allreduce_grads_in_fp32=True,
            attention_softmax_in_fp32=True,
            sglang_mem_fraction_static=0.75,
            save_interval=10,
            eval_interval=None,
            eval_top_p=1.0,
            eval_max_response_len=8192,
            n_samples_per_eval_prompt=8,
            apply_chat_template_kwargs='{"enable_thinking": true}',
            environment={
                "PYTHONPATH": "/root/Megatron-LM/:/root",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
            },
            over_sampling_batch_size=48,
            dynamic_sampling_filter_path=(
                "slime.rollout.filter_hub.dynamic_sampling_filters."
                "check_reward_nonzero_std"
            ),
            kl_coef=0.0,
            balance_data=True,
            calculate_per_token_loss=True,
            rollout_top_p=1.0,
        ),
    )
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _check_trained_intro():
    """
    ## Check the trained model

    Redeploy the custom SGLang server with the checkpoint Volume mounted and
    run the same scoring loop.
    """


@code
def _check_trained():
    trained_model = train_result.hf_model()
    print(f"Checkpoint: {trained_model.model_path}")

    trained_url = serve_model(
        trained_model.model_path,
        trained_model.model_name,
        "gym-qwen3-4b-dapo-check-trained",
        train_result.checkpoints_volume,
    )
    print(f"Trained model URL: {trained_url}")

    print("--- Checking trained model... ---")
    trained_rows = run_math_check(trained_url, trained_model.model_name)
    trained_mean = mean_score(trained_rows)
    n_correct = sum(1 for row in trained_rows if row.metadata.get("correct"))
    print(f"Trained accuracy: {n_correct}/{len(trained_rows)} ({trained_mean:.1%})")


@notebook_only
@code
def _trained_examples():
    for base_r, trained_r in zip(base_rows[:3], trained_rows[:3]):
        label = base_r.metadata["label"]
        b_status = "CORRECT" if base_r.metadata["correct"] else "WRONG"
        t_status = "CORRECT" if trained_r.metadata["correct"] else "WRONG"
        print(f"label={label}")
        print(f"  Base:    [{b_status}] pred={base_r.metadata['pred']}")
        print(f"  Trained: [{t_status}] pred={trained_r.metadata['pred']}")
        print()


@markdown
def _compare_intro():
    """
    ## Results

    Let's see if our model works better!
    """


@code
def _compare():
    base_correct = sum(1 for row in base_rows if row.metadata.get("correct"))
    trained_correct = sum(1 for row in trained_rows if row.metadata.get("correct"))
    total = len(base_rows)
    print(f"Base model:    {base_correct}/{total} ({base_mean:.1%})")
    print(f"Trained model: {trained_correct}/{total} ({trained_mean:.1%})")
    print(f"Delta:         {trained_mean - base_mean:+.1%}")
