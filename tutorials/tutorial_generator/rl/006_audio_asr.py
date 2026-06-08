# pyright: reportUndefinedVariable=false
"""Tutorial source for `006_audio_asr` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 2×H100",
    "summary": "Audio GRPO on Qwen3-ASR-1.7B — transcribe LibriSpeech, reward −WER",
    "difficulty": "Intermediate",
    "order": 40,
    "api_classes": [
        "Qwen3_ASR_1_7B",
        "Qwen3_ASR_1_7b_Recipe",
        "MultimodalDataset",
        "TrainConfig",
        "WandbConfig",
        "DeploymentConfig",
        "EvalConfig",
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
    Run with (the eval step decodes audio locally, so bring the audio libs):
    ```
    uv run --with soundfile --with librosa --with jiwer --with datasets \\
      python tutorials/rl/006_audio_asr/006_audio_asr.py
    ```
    """


@notebook_only
@shell(
    "%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main "
    "soundfile librosa jiwer datasets"
)
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        ModelDeployment,
        MultimodalDataset,
        Qwen3_ASR_1_7B,
        Qwen3_ASR_1_7b_Recipe,
        TrainConfig,
        WandbConfig,
        list_checkpoints,
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
    recipe you write only sets the reward and (optionally) W&B logging. It defaults
    to a `H100:2` single node; pass `actor_num_gpus_per_node=8` (and a larger
    `num_rollout`) to use a full node.

    `TrainConfig.train()` builds the Modal app, runs GRPO, and saves the trained
    model as a Megatron checkpoint (exported to HuggingFace on demand at deploy).
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3_ASR_1_7B(),
        dataset=dataset,
        recipe=Qwen3_ASR_1_7b_Recipe(
            custom_rm_function=word_error_rate_reward,
            wandb=WandbConfig(
                project="qwen3-asr-rl",
                group="gym-demo",
                exp_name="qwen3-asr-grpo-audio-demo",
            ),
        ),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _eval_intro():
    """
    ## Evaluate and watch it on the dashboard

    Evaluation is the same `DeploymentConfig` → `EvalConfig` flow every gym example
    uses. `DeploymentConfig.serve()` serves the trained checkpoint on SGLang
    (converting the Megatron checkpoint to HuggingFace first, audio tower included),
    and `EvalConfig.evaluate()` runs our `eval_fn` over the held-out clips.

    The eval_fn is the read-side twin of the reward: it POSTs each clip to the
    deployment's `/v1/audio/transcriptions` endpoint, scores word accuracy
    (`1 − WER`), and attaches a downsampled audio clip + reference + WER as
    `metadata`. The gym dashboard's **Evals** panel renders that metadata, so each
    clip shows up with an audio player next to its reference and score (run
    `training-gym setup` to get the dashboard URL).
    """


@code
def _eval_fn():
    def transcribe_and_score(
        deployment: ModelDeployment, example: dict
    ) -> EvalRowResult:
        import base64
        import io

        import jiwer
        import librosa
        import requests
        import soundfile as sf

        data_uri = example["audios"][0]
        reference = (example["label"] or "").lower().strip()
        b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
        arr, sr = sf.read(io.BytesIO(base64.b64decode(b64)))

        # Transcribe the full-resolution clip (WER must reflect the real audio).
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        buf.seek(0)
        resp = requests.post(
            f"{deployment.url}/v1/audio/transcriptions",
            files={"file": ("clip.wav", buf, "audio/wav")},
            data={
                "model": deployment.deployment_config.served_model_name,
                "temperature": "0.0",
            },
            timeout=120,
        )
        resp.raise_for_status()
        hypothesis = (resp.json().get("text") or "").lower().strip()
        wer = float(jiwer.wer(reference, hypothesis)) if reference else 0.0

        # Light, downsampled clip (8 kHz mono) for the dashboard audio player.
        small = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=8000)
        sbuf = io.BytesIO()
        sf.write(sbuf, small, 8000, format="WAV", subtype="PCM_16")
        audio_uri = "data:audio/wav;base64," + base64.b64encode(
            sbuf.getvalue()
        ).decode()

        # Score is word accuracy (1 − WER) in [0, 1] — higher is better, matching
        # the dashboard's score model; raw WER stays in metadata.
        return EvalRowResult(
            score=max(0.0, 1.0 - wer),
            response=hypothesis,
            prompt=example["prompt"],
            metadata={
                "audio": audio_uri,
                "reference": reference,
                "wer": wer,
                "hyp": hypothesis,
            },
        )


@code
def _eval():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    deployment = DeploymentConfig(
        model=Qwen3_ASR_1_7B(),
        checkpoint=checkpoint,
    ).serve()
    print(f"Serving trained model at {deployment.url}")

    eval_config = EvalConfig(dataset=dataset, eval_fn=transcribe_and_score)
    eval_result = eval_config.evaluate(deployment, debug=True)
    mean_wer = sum(r.metadata["wer"] for r in eval_result.rows) / len(eval_result.rows)
    print(
        f"Eval: mean WER {mean_wer:.3f} "
        f"(accuracy {eval_result.mean:.3f}) over {eval_result.total} clips"
    )
