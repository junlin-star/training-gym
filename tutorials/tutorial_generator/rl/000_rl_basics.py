"""Tutorial source for `000_rl_basics` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Haiku scoring with verifiable rewards",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
        "Qwen3_4B",
        "endpoint_chat",
        "wait_for_server_url",
        "TrainConfig",
        "SlimeRecipe",
        "TrainResult",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # RL basics: verifiable rewards, haiku edition

    This tutorial uses Qwen3-4B and haiku poems to introduce the
    **verifiable reward** pattern that underpins RL post-training:

    1. Serve the base model with a small Modal `@app.server`.
    2. Define a scoring function with a verifiable reward (syllable structure).
    3. Run that scorer yourself over a small check set in custom code.
    4. GRPO-train the model with [slime](https://github.com/THUDM/slime) using the reward function.
    5. Point the same server code at the trained checkpoint volume.
    6. Re-run the same custom scorer and compare.

    **Why haikus?** A haiku has two attributes you can score
    automatically — whether it follows the 5-7-5 syllable format
    (deterministic, cheap) and whether the poem is actually good. That split between
    *verifiable* and *subjective* rewards is exactly the landscape
    RL post-training operates in. This tutorial covers the
    verifiable half. In a later tutorial, we will cover the subjective half.
    """

@py_only
@markdown
def run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run tutorials/rl/000_rl_basics/000_rl_basics.py
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
    "if importlib.util.find_spec('nltk') is None:\n"
    "    %uv pip install -q nltk"
)
def _install():
    pass

@code
def _imports():
    import re

    import modal

    from modal_training_gym import (
        HuggingFaceDataset,
        Qwen3_4B,
        SlimeRecipe,
        TrainConfig,
        endpoint_chat,
        wait_for_server_url,
    )


@markdown
def _serve_base_intro():
    """
    ## Serve the base model

    So, how does Qwen3-4B currently fare at writing haikus? We can
    serve the base model and find out.

    Qwen3-4B is not in the managed Endpoint catalog, so this follows the ASR
    tutorial's pattern: ordinary Modal `@app.server` code launches SGLang and
    exposes its OpenAI-compatible `/v1` API.

    Custom `@app.server` endpoints here use `unauthenticated=False`. Run
    `training-gym set-proxy-auth` or export `MODAL_KEY`/`MODAL_SECRET` before
    calling `wait_for_server_url` / `endpoint_chat` with `proxy_auth=True`.
    """


@code
def _serve_base_model():
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

    base_url = serve_model(MODEL_ID, MODEL_ID, "gym-qwen3-4b-haiku-check-base")
    print(f"Base model server: {base_url}")

@notebook_only
@markdown
def _qualitative_check_of_base_model():
    """
    The server will take a moment to provision, but once it's ready, we can
    request it to write a haiku about a topic.
    """

@notebook_only
@code
def _qualitative_check_of_base_model_code():
    response = endpoint_chat(
        base_url,
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Write a haiku about cat."}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        proxy_auth=True,
    )
    print(response)


@markdown
def _scoring_intro():
    """
    Write post-train checks in custom code. A good check assigns
    a score to an outcome: binary or continuous, deterministic or subjective.

    In our case, we want our model to be good at writing haiku poems, so how do we score whether an LLM response is a good haiku?
    
    Well, a haiku must follow the 5-7-5 syllable format, so we can count syllables using NLTK's CMU Pronouncing Dictionary
    (with a regex fallback for words not in the dictionary)
    and score how close each line is to its target syllable count.

    We can give it score 0 if it doesn't follow the 5-7-5 syllable format, and 1 if it does. But that's not very informative.
    Instead, we can score it based on how close it is to the target syllable count for each line.
    """
@code
def _score_haiku():
    _cmudict_cache = {}

    def _get_cmudict() -> dict:
        if not _cmudict_cache:
            import nltk
            from nltk.corpus import cmudict
            nltk.download("cmudict", quiet=True)
            _cmudict_cache.update(cmudict.dict())
        return _cmudict_cache

    def _count_syllables(text: str) -> int:
        cmu = _get_cmudict()
        total = 0
        for word in re.findall(r"[a-zA-Z]+", text):
            phones = cmu.get(word.lower())
            if phones:
                total += sum(p[-1].isdigit() for p in phones[0])
            else:
                count = len(re.findall(r"[aeiouy]+", word.lower()))
                if word.lower().endswith("e") and count > 1:
                    count -= 1
                total += max(count, 1)
        return total

    def score_haiku(response: str) -> float:
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        if len(lines) != 3:
            return -10
        total_diff = sum(
            abs(_count_syllables(line) - target)
            for line, target in zip(lines, [5, 7, 5])
        )
        return -float(total_diff)

@notebook_only
@code
def _score_haiku_demo():
    response = endpoint_chat(
        base_url,
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Write a haiku about cat."}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        proxy_auth=True,
    )
    print(response)
    print(f"Score: {score_haiku(response)}")

@markdown
def _define_dataset():
    """
    Let's also define a Haiku dataset.
    Here, we use the statworx/haiku dataset from HuggingFace.
    Each row has a `keywords` topic and a reference `text` haiku.
    We can use this dataset to train our model.

    Datasets for training models can take many form factors, and huggingface dataset is just one of them.
    If you're curious about other options, check out the [DatasetConfig](https://gym.modal.dev/reference/core/datasetconfig/) documentation.
    """

@code
def _define_dataset_code():
    class HaikuDataset(HuggingFaceDataset):
        hf_repo = "statworx/haiku"
        input_column = "keywords"
        output_column = "text"
        output_format = "jsonl"
        apply_chat_template = True
        system_prompt = (
            "You are a haiku poet. Write a haiku about the given topic. "
            "Use the 5-7-5 syllable format across three lines."
        )
        prompt_template = "Write a haiku about {input}."
        always_prepare = True # For the purpose of this tutorial, we want to prepare the dataset every time we run it, in case there is stale data from a previous run.

    train_dataset = HaikuDataset(n_rows=10)
    check_dataset = HaikuDataset(n_rows=5)

@notebook_only
@markdown
def _check_dataset_head():
    """
    Let's take a look at the held-out set. Each row has a `keywords`
    topic and a reference `text` haiku.
    """

@notebook_only
@code
def _check_dataset_head_code():
    df = check_dataset.to_pandas()
    print(len(df))
    df.head(5)

@markdown
def _custom_scoring_intro():
    """
    ## Custom scoring loop

    Loop the dataset: call the server, score, aggregate. Keep the scorer as a
    plain function so training can reuse it as `custom_rm_function`.
    """


@code
def _check_base_model():
    def run_custom_check(url: str) -> float:
        scores = []
        for example in check_dataset.load():
            topic = example[check_dataset.input_column]
            messages = [
                {"role": "system", "content": check_dataset.system_prompt},
                {
                    "role": "user",
                    "content": check_dataset.prompt_template.format(input=topic),
                },
            ]
            response = endpoint_chat(
                url,
                model=MODEL_ID,
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                proxy_auth=True,
            )
            score = score_haiku(response)
            print(f"score={score:.1f} response={response!r}", flush=True)
            scores.append(score)
        return sum(scores) / len(scores) if scores else float("nan")

    print("——— Running base model custom check... ———")
    base_mean = run_custom_check(base_url)
    print(f"Average haiku score: {base_mean:.1f}")
    print("——— Base model custom check complete ———")

@markdown
def _train_intro():
    """
    ## Train with slime

    Now, let's actually train the model to write good haikus.
    Here, we use the slime framework (https://github.com/THUDM/slime) on Modal.

    All flags that are native to slime can be passed to the `TrainConfig` object.
    You can also add patches to slime using the `image_overlay` argument.
    """

@code
def _define_training_run():
    async def haiku_rm(args, sample, **kwargs) -> float:
        response = base_model.parse_response(sample.response)
        return score_haiku(response.content)
    
    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            custom_rm_function=haiku_rm,

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
            apply_chat_template_kwargs='{"enable_thinking": false}',

            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp nltk>=3.8.0",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )


@markdown
def _train_section():
    """
    ## Train

    `TrainConfig.train()` builds the Modal app, runs training, and
    returns a `TrainResult` with the run ID and checkpoint path.
    """


@code
def _invoke_train():
    print("——— Running training... ———")
    train_result = training_run.train()
    print("——— Training complete ———")


@markdown
def _trained_endpoint_intro():
    """
    ## Serve the trained checkpoint

    Reuse the SGLang `@app.server` code with the checkpoint Volume mounted.
    SGLang loads HuggingFace-format directories, while slime writes
    Megatron/torch_dist checkpoints, so `train_result.hf_model()` converts the
    newest checkpoint to `<name>_hf` on the same Volume first.
    """


@code
def _serve_trained():
    checkpoint = next(
        c for c in reversed(train_result.checkpoints()) if not c.name.endswith("_hf")
    )
    print(checkpoint.path)

    trained_model = train_result.hf_model()
    print(trained_model.model_path)

    trained_url = serve_model(
        trained_model.model_path,
        trained_model.model_name,
        "gym-qwen3-4b-haiku-check-trained",
        train_result.checkpoints_volume,
    )
    print(f"Trained model server: {trained_url}")


@code
def _check_trained():
    print("——— Running trained model custom check... ———")
    trained_mean = run_custom_check(trained_url)
    print(f"Trained haiku score: {trained_mean:.1f}")
    print("——— Trained model custom check complete ———")

@markdown
def _continue_to_train_off_of_a_checkpoint():
    """
    ## Train off of a checkpoint
    Hmm, looks like the trained model is not doing very well.
    Maybe it's because it only trained for 10 iterations.

    What happens if we train it for more?
    We want to train it off of the latest checkpoint, not from scratch.
    """

@code
def _continue_to_train_off_of_a_checkpoint_code():
    new_training_run = TrainConfig(
        model=Qwen3_4B(),
        dataset=train_dataset,
        checkpoint=checkpoint,
        recipe=SlimeRecipe(
            custom_rm_function=haiku_rm,

            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,

            num_rollout=20,
            rollout_batch_size=16,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,

            save_interval=10,
            apply_chat_template_kwargs='{"enable_thinking": false}',

            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp nltk>=3.8.0",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )
    print("——— Running new training... ———")
    new_train_result = new_training_run.train()
    print("——— New training complete ———")

@markdown
def _trained_check_off_of_a_checkpoint():
    """
    ## Score the continued checkpoint

    Redeploy the custom server with the new checkpoint and re-run the custom check.
    """

@code
def _trained_check_off_of_a_checkpoint_code():
    new_trained_model = new_train_result.hf_model()
    print(new_trained_model.model_path)

    new_url = serve_model(
        new_trained_model.model_path,
        new_trained_model.model_name,
        "gym-qwen3-4b-haiku-check-continued",
        new_train_result.checkpoints_volume,
    )
    print(f"Newly trained model server: {new_url}")

@markdown
def _trained_check_off_of_a_checkpoint_results():
    """
    ## Compare second-run results

    Now let's compare the results of the newly trained model and the base model.
    """

@code
def _trained_check_off_of_a_checkpoint_results_code():
    print("——— Running trained model custom check... ———")
    new_mean = run_custom_check(new_url)
    print(f"Trained model (new) haiku score: {new_mean:.1f}")
    print("——— Trained model (new) custom check complete ———")

@markdown
def _compare_results():
    """
    ## Compare all runs

    Now let's compare the results across all three checkpoints.
    """

@code
def _compare_results_code():
    print(f"Base model haiku score: {base_mean:.1f}")
    print(f"Trained model haiku score: {trained_mean:.1f}")
    print(f"Trained model (new) haiku score: {new_mean:.1f}")
