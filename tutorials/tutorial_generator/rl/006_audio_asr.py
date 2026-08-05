# pyright: reportUndefinedVariable=false
"""Tutorial source for `006_audio_asr` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 2×H100",
    "summary": "Audio GRPO for transcribing LibriSpeech",
    "difficulty": "Intermediate",
    "order": 39,
    "api_classes": [
        "Qwen3_ASR_1_7B",
        "Qwen3_ASR_1_7b_Recipe",
        "MultimodalDataset",
        "Sample",
        "TrainConfig",
        "TrainResult",
        "wait_for_server_url",
    ],
    "required_modal_secrets": [
        {"name": "wandb-secret", "key": "WANDB_API_KEY"},
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Audio GRPO on Qwen3-ASR-1.7B

    This tutorial demonstrates training [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)-- a speech
    recognizer — end-to-end with GRPO.

    The loop:

    1. Load LibriSpeech speech clips with a small `MultimodalDataset`
       (`modality="audio"`).
    2. slime serves Qwen3-ASR on SGLang's `/v1/audio/transcriptions` endpoint and
       the gym's audio-transcription rollout posts each clip, collecting the
       transcript.
    3. Your `word_error_rate_reward` scores each transcript as **−WER** (word error
       rate) against the reference text.
    4. That reward drives a GRPO update through slime/Megatron.

    The Training Gym takes care of the nitty gritty compatibility matching--
    you just pick `model=Qwen3_ASR_1_7B()` and `recipe=Qwen3_ASR_1_7b_Recipe(...)`,
    and bring the reward.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run --with soundfile --with jiwer --with datasets \\
      python tutorials/rl/006_audio_asr/006_audio_asr.py
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
    "if importlib.util.find_spec('soundfile') is None:\n"
    "    %uv pip install -q soundfile jiwer datasets"
)
def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym import (
        MultimodalDataset,
        Qwen3_ASR_1_7B,
        Qwen3_ASR_1_7b_Recipe,
        Sample,
        TrainConfig,
        wait_for_server_url,
    )
@markdown
def _dataset_intro():
    """
    ## Load LibriSpeech audio

    The dataset is the one piece of boilerplate worth seeing in full. It's a small
    `MultimodalDataset` (`modality="audio"`) that pulls a few clips from the
    standard LibriSpeech dummy set: each row is a prompt with an `<audio>`
    placeholder, the clip itself (base64 `data:audio/wav` URI), and the reference
    transcript as the label. As a `MultimodalDataset` it tells slime to forward the
    audio column to the rollout — the same passthrough images and video use.

    We re-encode every clip to WAV and keep `sample.prompt` a message list
    (`apply_chat_template=False`) so the audio data-URI survives into the rollout.
    """


@code
def _dataset():
    INSTRUCTION = (
        "<audio>\nTranscribe the speech to text. Respond with only the transcript."
    )

    class LibriSpeechASRDataset(MultimodalDataset):
        """LibriSpeech ASR rows (prompt + audio data-URI + transcript label)."""

        modality = "audio"
        hf_repo = "hf-internal-testing/librispeech_asr_dummy"
        hf_config = "clean"
        hf_split = "validation"
        n_rows = 8
        # Re-materialize each run so prompt changes take effect instead of being
        # shadowed by a stale jsonl on the data volume.
        always_prepare = True
        # Keep sample.prompt a conversation list (don't collapse to a templated
        # string) so the audio data-URI survives for the transcription rollout.
        apply_chat_template = False

        def __init__(self, **kwargs):
            super().__init__(rows=[], **kwargs)

        def _build_rows(self) -> list[dict]:
            import base64 as b64
            import io

            import soundfile as sf
            from datasets import Audio, load_dataset

            ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
            ds = ds.select(range(min(self.n_rows, len(ds))))
            # decode=False avoids the torchcodec dependency; decode with soundfile.
            ds = ds.cast_column("audio", Audio(decode=False))
            # Demo-scale: materializes every clip as an inline base64 row in memory.
            # Fine for a handful of clips; for large corpora stream / store by reference.
            rows = []
            for ex in ds:
                audio = ex["audio"]
                data = (
                    audio["bytes"]
                    if audio.get("bytes")
                    else open(audio["path"], "rb").read()
                )
                arr, sr = sf.read(io.BytesIO(data))
                buf = io.BytesIO()
                sf.write(buf, arr, sr, format="WAV")
                data_uri = "data:audio/wav;base64," + b64.b64encode(
                    buf.getvalue()
                ).decode("ascii")
                rows.append(
                    {
                        self.input_key: INSTRUCTION,
                        self.media_column: [data_uri],
                        self.label_key: ex["text"].lower().strip(),
                    }
                )
            return rows

        def load(self, split: str = "all") -> list[dict]:
            return self._build_rows()

        def prepare(self, path, eval_paths=None):
            rows = self._build_rows()
            self._write_jsonl(rows, path)
            if eval_paths:
                for eval_path in eval_paths.values():
                    self._write_jsonl(rows, eval_path)

    dataset = LibriSpeechASRDataset(n_rows=8)


@notebook_only
@markdown
def _dataset_preview():
    """
    Let's look at a row — text prompt, an audio data-URI, and the reference label.
    """


@notebook_only
@code
def _dataset_preview_code():
    row = dataset.load()[0]
    print("prompt:", row["prompt"])
    print("audio: ", row["audios"][0][:48], "...")
    print("label: ", row["label"])


@markdown
def _reward_intro():
    """
    ## Define the reward

    This is the one task-specific piece. slime calls the reward once per rollout
    sample with a `Sample` carrying `.response` (the transcript the model produced)
    and `.label` (the reference). We score it as **negative word error rate** so
    that lower WER → higher reward, and GRPO pushes the model toward more accurate
    transcripts. (`jiwer` is installed for you with the model.)

    Qwen3-ASR is already near-perfect on clean LibriSpeech, so the
    `Qwen3_ASR_1_7b_Recipe` defaults sample many transcripts per clip at
    temperature 1.0 — that's what gives the GRPO group enough within-group WER
    variance to produce a non-zero gradient.
    """


@code
def _reward():
    async def word_error_rate_reward(args, sample, **kwargs) -> float:
        import jiwer

        response = (getattr(sample, "response", "") or "").lower().strip()
        reference = (getattr(sample, "label", "") or "").lower().strip()
        if not reference:
            return 0.0
        return -float(jiwer.wer(reference, response))


@markdown
def _train_intro():
    """
    ## Train

    `Qwen3_ASR_1_7b_Recipe` carries the ASR-specific defaults — the transcription
    rollout, padded (bshd) batches, the lighter SGLang memory fraction, and the
    many-samples/high-temperature settings that surface reward variance — so the
    recipe you write only sets the reward. It defaults to a `H100:2` single node;
    pass `actor_num_gpus_per_node=8` (and a larger `num_rollout`) to use a full node.

    To log training curves to W&B, also pass `wandb=WandbConfig(project="…")` to the
    recipe — that needs a W&B account with write access, supplied via the
    `wandb-secret` Modal secret.

    `TrainConfig.train()` builds the Modal app, runs GRPO, and saves the trained
    model as a Megatron checkpoint, which `train_result.hf_model()` converts to
    HuggingFace format when the check needs to serve it.
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3_ASR_1_7B(),
        dataset=dataset,
        recipe=Qwen3_ASR_1_7b_Recipe(custom_rm_function=word_error_rate_reward),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _check_intro():
    """
    ## Check the trained model

    Qwen3-ASR only answers on `/v1/audio/transcriptions`, and managed
    [Modal Endpoints](https://modal.com/docs/guide/endpoints) serve the OpenAI
    *chat* API under `/v1` — nothing else. So the check serves the checkpoint
    itself: a small `@app.server` running SGLang against the checkpoints Volume,
    the same shape the on-policy-distillation tutorials use for their teachers.

    Serving reads HuggingFace-format weights, while slime writes
    Megatron/torch_dist, so `train_result.hf_model()` converts the newest
    checkpoint to `<name>_hf` on the same Volume before the server mounts it.

    The scoring function is the read-side twin of the reward. It scores word
    accuracy (`1 - WER`) and keeps the reference and WER in `Sample.metadata`.
    """


@code
def _check_fn():
    def transcribe_and_score(
        base_url: str,
        model_id: str,
        example: dict,
    ) -> Sample:
        import base64
        import io

        import jiwer
        import requests
        import soundfile as sf

        data_uri = example["audios"][0]
        reference = (example["label"] or "").lower().strip()
        b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
        arr, sr = sf.read(io.BytesIO(base64.b64decode(b64)))

        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        buf.seek(0)
        resp = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": ("clip.wav", buf, "audio/wav")},
            data={
                "model": model_id,
                "temperature": "0.0",
            },
            timeout=120,
            allow_redirects=False,
        )
        resp.raise_for_status()
        hypothesis = (resp.json().get("text") or "").lower().strip()
        wer = float(jiwer.wer(reference, hypothesis)) if reference else 0.0

        return Sample(
            score=max(0.0, 1.0 - wer),
            response=hypothesis,
            prompt=example["prompt"],
            metadata={
                "reference": reference,
                "metrics": {"wer": wer},
            },
        )


@code
def _serve_fn():
    ASR_APP_NAME = "gym-qwen3-asr-1-7b-check"
    ASR_PORT = 8000
    ASR_STARTUP_TIMEOUT = 20 * 60

    asr_image = (
        modal.Image.from_registry("lmsysorg/sglang:v0.5.12")
        .entrypoint([])
        .env({"HF_HUB_CACHE": "/root/hf-cache"})
    )

    def serve_asr_transcriptions(
        model_path: str,
        served_model_name: str,
        checkpoints_volume_name: str,
        checkpoints_mount_path: str = "/checkpoints",
    ) -> str:
        asr_app = modal.App(ASR_APP_NAME)
        port = ASR_PORT
        startup_timeout = ASR_STARTUP_TIMEOUT

        @asr_app.server(
            image=asr_image,
            gpu="H100",
            volumes={
                "/root/hf-cache": modal.Volume.from_name(
                    "huggingface-cache", create_if_missing=True
                ),
                checkpoints_mount_path: modal.Volume.from_name(
                    checkpoints_volume_name, create_if_missing=True
                ),
            },
            port=port,
            startup_timeout=startup_timeout,
            scaledown_window=10 * 60,
            exit_grace_period=25,
            target_concurrency=4,
            unauthenticated=True,
            serialized=True,
        )
        class AsrServer:
            @modal.enter()
            def start(self):
                import subprocess as _sp
                import time as _time
                import urllib.error as _ue
                import urllib.request as _ur

                cmd = [
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
                    str(port),
                    "--mem-fraction-static",
                    "0.80",
                    "--trust-remote-code",
                ]
                self.proc = _sp.Popen(cmd)
                deadline = _time.monotonic() + startup_timeout
                health = f"http://127.0.0.1:{port}/health"
                while True:
                    if self.proc.poll() is not None:
                        raise RuntimeError(
                            f"SGLang exited with code {self.proc.returncode} "
                            "before healthy"
                        )
                    try:
                        with _ur.urlopen(health, timeout=5) as resp:
                            if resp.status == 200:
                                return
                    except (_ue.URLError, TimeoutError, OSError):
                        pass
                    if _time.monotonic() >= deadline:
                        raise TimeoutError(f"Qwen3-ASR not healthy at {health}")
                    _time.sleep(2)

            @modal.exit()
            def stop(self):
                proc = getattr(self, "proc", None)
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)

        with modal.enable_output():
            asr_app.deploy()
        return wait_for_server_url(AsrServer, label="Qwen3-ASR check server")


@code
def _check():
    trained_model = train_result.hf_model()
    print(f"Checkpoint: {trained_model.model_path}")

    trained_url = serve_asr_transcriptions(
        model_path=trained_model.model_path,
        served_model_name=trained_model.model_name,
        checkpoints_volume_name=train_result.checkpoints_volume,
    )
    print(f"Serving trained model at {trained_url}")

    rows = [
        transcribe_and_score(trained_url, trained_model.model_name, example)
        for example in dataset.load()
    ]
    mean_wer = sum(row.metadata["metrics"]["wer"] for row in rows) / len(rows)
    mean_accuracy = sum(row.score for row in rows) / len(rows)
    print(
        f"Check: mean WER {mean_wer:.3f} "
        f"(accuracy {mean_accuracy:.3f}) over {len(rows)} clips"
    )
