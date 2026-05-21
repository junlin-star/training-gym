# pyright: reportUndefinedVariable=false
"""Tutorial source for `004_usaco` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "4 × 8×H100",
    "summary": "Hill-climb Qwen3.6-35B-A3B on USACO with sandbox-verified rewards",
    "difficulty": "Advanced",
    "order": 25,
    "api_classes": [
        "HarborDataset",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "ModelDeployment",
        "Qwen3_6_35B",
        "SlimeRecipe",
        "TrainConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Hill-climbing USACO with Qwen3.6-35B-A3B

    This tutorial trains **Qwen3.6-35B-A3B** (a 35B-parameter MoE model
    with ~3B active) on competitive programming problems from the
    [USACO dataset on Harbor Hub](https://hub.harborframework.com/datasets/usaco/usaco/latest).

    The loop:
    1. Pull 304 USACO tasks from Harbor Hub via `HarborDataset`.
       Each task ships with `instruction.md` and `tests/*.in` / `*.out` pairs.
    2. Score model outputs by executing generated code in Modal sandboxes,
       piping test inputs via stdin and comparing stdout.
    3. Feed that score back as a GRPO reward through SLIME.
    4. Compare base vs. trained pass rates.

    This is the same sandbox-verified reward pattern from
    [001 Sandboxes](../001_sandboxes/001_sandboxes), scaled up to a
    larger model and a real competitive-programming benchmark.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run modal run -d tutorials/rl/004_usaco/004_usaco.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main harbor")
def _install():
    pass


@code
def _imports():
    import json
    import re

    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        HarborDataset,
        ModelDeployment,
        Qwen3_6_35B,
        SlimeRecipe,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.deploy_recipes.sglang_recipe import Qwen3_6_35b_SglangRecipe


@markdown
def _dataset_intro():
    """
    ## Load USACO from Harbor Hub

    The [usaco/usaco](https://hub.harborframework.com/datasets/usaco/usaco/latest)
    dataset contains 304 competitive-programming tasks. Each task has:
    - `instruction.md` — the problem statement
    - `tests/` — verification test cases as `*.in` / `*.out` file pairs

    Setting `test_data_dir="tests"` tells `HarborDataset` to parse those
    pairs into a `test_cases` list on each row's label, which we later
    feed to the sandbox scorer.

    We use a subset for training and hold out 20 tasks for eval.
    """


@code
def _dataset():
    SYSTEM_PROMPT = (
        "You are an expert competitive programmer. "
        "Solve the given problem by writing a complete Python program. "
        "Your program must read from stdin and print the answer to stdout. "
        "Put your solution in a ```python code fence."
    )

    dataset = HarborDataset(
        dataset_name="usaco/usaco",
        test_data_dir="tests",
        train_size=100,
        eval_size=20,
        train_repeats=5,
        always_prepare=True,
        shuffle_tasks=True,
        shuffle_seed=42,
        system_prompt=SYSTEM_PROMPT,
    )


@notebook_only
@markdown
def _dataset_preview():
    """
    Let's take a quick look at the dataset. `to_pandas()` returns all
    tasks in order — the first `train_size` rows are the training split,
    the next `eval_size` rows are the eval split.
    """


@notebook_only
@code
def _dataset_preview_code():
    df = dataset.to_pandas()
    train_df = df[: dataset.train_size]
    eval_df = df[dataset.train_size : dataset.train_size + dataset.eval_size]
    print(f"{len(train_df)} train tasks, {len(eval_df)} eval tasks")

    print("\n— Train split —")
    display(train_df.head(3))

    print("\n— Eval split —")
    display(eval_df.head(3))


@markdown
def _sandbox_reward_intro():
    """
    ## Sandbox-backed scorer

    We extract Python code from the model's response, execute it in a
    Modal sandbox with each test input piped to stdin, and compare
    stdout against expected output. The reward is the fraction of
    test cases passed — a hard, verifiable signal.
    """


@code
def _sandbox_reward():
    _CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

    def extract_python_code(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if "<|im_start|>assistant" in normalized:
            normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
        if "</think>" in normalized:
            normalized = normalized.split("</think>", 1)[-1]
        normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
        if match := _CODE_FENCE_RE.search(normalized):
            return match.group(1).strip()
        return normalized

    def score_with_sandbox(
        response: str,
        *,
        test_cases: list[dict],
        timeout_sec: int = 60,
    ) -> tuple[float, dict]:
        import modal as _modal

        code = extract_python_code(response)
        if not test_cases:
            return 0.0, {"passed": 0, "total": 0}

        script = "\n".join([
            "import sys, io, json",
            f"candidate = {json.dumps(code)}",
            f"tests = {json.dumps(test_cases)}",
            "results = []",
            "for tc in tests:",
            "    sys.stdin = io.StringIO(tc['input'])",
            "    buf = io.StringIO()",
            "    old = sys.stdout",
            "    sys.stdout = buf",
            "    try:",
            "        exec(candidate, {}, {})",
            "        sys.stdout = old",
            "        results.append({'output': buf.getvalue(), 'ok': True})",
            "    except Exception as e:",
            "        sys.stdout = old",
            "        results.append({'output': '', 'ok': False})",
            "print(json.dumps(results))",
        ]) + "\n"

        try:
            app = _modal.App.lookup("training-gym-sandbox-rm", create_if_missing=True)
            sandbox = _modal.Sandbox.create(
                "python", "-c", script,
                app=app,
                image=_modal.Image.debian_slim(python_version="3.11"),
                timeout=timeout_sec,
                cpu=1.0,
                memory=1024,
            )
            stdout = sandbox.stdout.read()
            sandbox.stderr.read()
            sandbox.wait()
        except Exception:
            return 0.0, {"passed": 0, "total": len(test_cases)}

        try:
            results = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return 0.0, {"passed": 0, "total": len(test_cases)}

        passed = sum(
            1 for r, tc in zip(results, test_cases)
            if r.get("ok") and r.get("output", "").strip() == tc["expected_output"].strip()
        )
        return passed / len(test_cases), {"passed": passed, "total": len(test_cases)}

    async def usaco_rm(args, sample, **kwargs) -> float:
        import asyncio

        label = json.loads(sample.extra.get("label", "{}"))
        test_cases = label.get("test_cases", [])
        reward, meta = await asyncio.to_thread(
            score_with_sandbox, sample.response, test_cases=test_cases,
        )
        sample.metadata = {**(getattr(sample, "metadata", None) or {}), "usaco": meta}
        return float(reward)


@markdown
def _serve_eval_base_intro():
    """
    ## Serve and evaluate the base model

    First, let's see how the raw Qwen3.6-35B-A3B performs on USACO
    out of the box.
    """


@code
def _serve_eval_base():
    base_model = Qwen3_6_35B()
    base_deployment: ModelDeployment = DeploymentConfig(
        model=base_model,
        recipe=Qwen3_6_35b_SglangRecipe(),
    ).serve()
    print(f"Base model URL: {base_deployment.url}")

    def eval_fn(deployment: ModelDeployment, example: dict) -> EvalRowResult:
        prompt = example.get("instruction", "")
        test_cases = example.get("label", {}).get("test_cases", [])
        response = deployment.generate(
            prompt,
            ensure_ready=False,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        score, metadata = score_with_sandbox(response, test_cases=test_cases)
        return EvalRowResult(score=score, response=response, metadata=metadata)

    eval_config = EvalConfig(
        dataset=dataset,
        eval_fn=eval_fn,
    )
    print("Running base eval...")
    base_eval = eval_config.evaluate(base_deployment, debug=True)
    print(f"Base mean pass rate: {base_eval.mean:.4f}")


@markdown
def _train_intro():
    """
    ## Train with SLIME

    Now we hill-climb. This MoE model needs serious hardware —
    4 nodes × 8×H100 (32 GPUs) with TP4, PP2, CP4, and optimizer
    CPU offload, matching the official Slime recipe for Qwen3.5-27B.
    Key overrides:
    - **`custom_rm_function=usaco_rm`** — the sandbox reward defined above.
    - **`rollout_max_response_len=4096`** — competitive programming
      solutions need room to think.
    - **`image_overlay`** — install `modal` in the training image so
      the reward function can create sandboxes.
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3_6_35B(),
        dataset=dataset,
        recipe=SlimeRecipe(
            custom_rm_function=usaco_rm,

            gpu_type="H100",
            colocate=True,
            actor_num_nodes=4,
            actor_num_gpus_per_node=8,
            tensor_model_parallel_size=4,
            sequence_parallel=True,

            rollout_num_gpus_per_engine=2,
            num_rollout=1,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,
            sglang_mem_fraction_static=0.75,

            global_batch_size=64,
            lr=1e-6,
            max_tokens_per_gpu=8192,
            eval_max_response_len=4096,
            n_samples_per_eval_prompt=8,
            save_interval=20,
            eval_interval=20,

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
    ## Evaluate the trained checkpoint

    Let's serve the latest checkpoint and re-run the same eval.
    """


@code
def _serve_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    trained_deployment = DeploymentConfig(
        model=Qwen3_6_35B(),
        recipe=Qwen3_6_35b_SglangRecipe(),
        checkpoint=checkpoint,
        app_name="qwen3-6-35b-usaco-serve",
        served_model_name="qwen3-6-35b-usaco",
    ).serve()
    print(f"Trained model URL: {trained_deployment.url}")

    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    print(f"Trained mean pass rate: {trained_eval.mean:.4f}")
    print(f"Base mean pass rate:    {base_eval.mean:.4f}")
