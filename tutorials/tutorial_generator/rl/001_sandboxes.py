# pyright: reportUndefinedVariable=false
"""Tutorial source for `001_sandboxes` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Code RL with Harbor sandboxes",
    "difficulty": "Intermediate",
    "order": 20,
    "api_classes": [
        "HarborDataset",
        "endpoint_chat",
        "Qwen3_4B",
        "SlimeRecipe",
        "TrainConfig",
        "wait_for_server_url",
        "score_in_sandbox",
        "extract_code",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Code RL with Harbor hello-world + Modal sandboxes

    What if you have a task where you want to score model outputs by running them in an environment?

    This tutorial trains a model on the
    [hello-world](https://hub.harborframework.com/tasks/harbor/hello-world/latest)
    task from Harbor Hub, scoring solutions by spawning and executing them in Modal sandboxes.

    Workflow:
    1. Pull the hello-world task from Harbor Hub via `HarborDataset`.
    2. Serve the base model with a small Modal `@app.server`.
    3. Score model outputs yourself: `endpoint_chat` → `extract_code` → `score_in_sandbox`.
    4. Reuse the same `score_in_sandbox` helper as a SLIME `custom_rm_function`.
    5. Train and compare base vs. trained behavior with the same custom check.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run tutorials/rl/001_sandboxes/001_sandboxes.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main\n"
    "if importlib.util.find_spec('harbor') is None:\n"
    "    %uv pip install -q harbor"
)
def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym import (
        HarborDataset,
        Qwen3_4B,
        SlimeRecipe,
        TrainConfig,
        endpoint_chat,
        extract_code,
        score_in_sandbox,
        wait_for_server_url,
    )


@markdown
def _dataset_intro():
    """
    ## Load hello-world from Harbor Hub

    `HarborDataset` accepts a `dataset_name` to pull tasks from
    [Harbor Hub](https://hub.harborframework.com). Each task has:
    - `instruction.md` — the problem statement (prompt)
    - `task.toml` — metadata (difficulty, category)
    - `tests/` — verification tests (format varies by task)

    The hello-world task uses pytest-based verification rather than
    `*.in`/`*.out` file pairs, so we define stdin/stdout test cases
    inline and pass them to `score_in_sandbox`.

    A single dataset instance handles training and held-out checks.
    `prepare()` writes both splits to the volume,
    while `load()` returns all tasks for offline checking.
    """


@code
def _dataset():
    HELLO_WORLD_TESTS = [{"input": "", "expected_output": "Hello, world!\n"}]

    dataset = HarborDataset(
        dataset_name="harbor/hello-world",
        label_metadata_path="task.toml",
        train_repeats=20,
        always_prepare=True, # For the purpose of this tutorial, we want to prepare the dataset every time we run it, in case there is stale data from a previous run.
        system_prompt=(
            "You are an expert Python programmer. "
            "Solve the given problem by writing a complete Python program. "
            "Your program must print the answer to stdout using print(). "
            "Do not create or write any files. "
            "Put your solution in a ```python code fence."
        ),
    )


@notebook_only
@markdown
def _dataset_preview():
    """
    Let's take a quick look at part of the dataset as a pandas DataFrame.
    Each row includes the task prompt plus the parsed Harbor label metadata.
    """


@notebook_only
@code
def _dataset_preview_code():
    df = dataset.to_pandas()
    print(len(df))
    df.head(5)


@markdown
def _custom_check_intro():
    """
    ## Custom scoring loop

    Qwen3-4B is not in the managed Endpoint catalog. As in the ASR tutorial,
    ordinary Modal `@app.server` code launches SGLang; the check then runs
    `endpoint_chat` → `extract_code` → `score_in_sandbox`.

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

    base_url = serve_model(MODEL_ID, MODEL_ID, "gym-qwen3-4b-hello-world-check-base")
    print(f"Base model server: {base_url}")

    def run_custom_check(url: str) -> float:
        scores = []
        for example in dataset.load():
            messages = []
            if dataset.system_prompt:
                messages.append({"role": "system", "content": dataset.system_prompt})
            messages.append({"role": "user", "content": example["instruction"]})
            response = endpoint_chat(
                url,
                model=MODEL_ID,
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                proxy_auth=True,
            )
            code = extract_code(response, model=base_model)
            reward, meta = score_in_sandbox(code, test_cases=HELLO_WORLD_TESTS)
            print(f"score={reward:.4f} code={code!r}", flush=True)
            if meta.get("stderr"):
                print(f"  sandbox stderr={meta['stderr']!r}", flush=True)
            scores.append(reward)
        return sum(scores) / len(scores) if scores else float("nan")

    print("——— Running base model custom check... ———")
    base_mean = run_custom_check(base_url)
    print(f"Base mean reward: {base_mean:.4f}")
    print("——— Base model custom check complete ———")


@markdown
def _train_intro():
    """
    ## Train with SLIME and sandbox reward

    For training, we reuse the same `score_in_sandbox` and `extract_code`
    helpers from the custom check — wrapped in an async reward function for
    SLIME's `custom_rm_function`.

    `score_in_sandbox` enforces `sandbox_cpu`/`sandbox_memory` with a
    `"limit"` policy by default: rather than reserving that capacity up
    front, the values become burst ceilings, so Modal bills each sandbox
    by actual CPU-/RAM-second usage instead of the (usually idle)
    reservation. Pass `cpu_policy="ignore"` to let rollouts burst above
    the configured values, or `"reserve"` for the legacy fixed-reservation
    behavior.
    """


@code
def _train():
    async def sandbox_rm(args, sample, **kwargs) -> float:
        import asyncio

        code = extract_code(sample.response, model=base_model)
        reward, meta = await asyncio.to_thread(
            score_in_sandbox, code, test_cases=HELLO_WORLD_TESTS,
        )
        sample.metadata = {**(getattr(sample, "metadata", None) or {}), "sandbox": meta}
        return float(reward)

    training_run = TrainConfig(
        model=base_model,
        dataset=dataset,
        recipe=SlimeRecipe(
            custom_rm_function=sandbox_rm,

            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,

            num_rollout=10,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            rollout_max_response_len=2048,
            rollout_temperature=0.9,

            global_batch_size=8,
            eval_max_response_len=2048,
            n_samples_per_eval_prompt=8,
            max_tokens_per_gpu=4096,
            save_interval=10,
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system modal>=1.2.0",
            ),
        ),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _serve_trained_intro():
    """
    ## Serve and check the trained checkpoint

    Reuse the SGLang `@app.server` code with the checkpoint Volume mounted,
    then run the same custom check.
    """


@code
def _serve_trained():
    trained_model = train_result.hf_model()
    print(trained_model.model_path)

    trained_url = serve_model(
        trained_model.model_path,
        trained_model.model_name,
        "gym-qwen3-4b-hello-world-check-trained",
        train_result.checkpoints_volume,
    )
    print(f"Trained model server: {trained_url}")

    print("——— Running trained model custom check... ———")
    trained_mean = run_custom_check(trained_url)
    print(f"Trained mean reward: {trained_mean:.4f}")
    print(f"Base mean reward:    {base_mean:.4f}")
    print("——— Trained model custom check complete ———")
