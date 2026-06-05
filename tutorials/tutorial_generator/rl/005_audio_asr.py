# pyright: reportUndefinedVariable=false
"""Tutorial source for `005_audio_asr` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 2×H100",
    "summary": "Audio GRPO on Qwen3-ASR-1.7B — transcribe LibriSpeech, reward −WER",
    "difficulty": "Intermediate",
    "order": 30,
    "api_classes": [
        "Qwen3ASR",
        "Qwen3ASR_Recipe",
        "LibriSpeechASRDataset",
        "TrainConfig",
        "WandbConfig",
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

    1. Load LibriSpeech speech clips with `LibriSpeechASRDataset` (a
       `MultimodalDataset` with `modality="audio"`).
    2. slime serves Qwen3-ASR on SGLang's `/v1/audio/transcriptions` endpoint and
       the gym's audio-transcription rollout posts each clip, collecting the
       transcript.
    3. Your `wer_reward` scores each transcript as **−WER** (word error rate)
       against the reference text.
    4. That reward drives a GRPO update through slime/Megatron.

    The Training Gym takes care of the nitty gritty compatibility matching--
    you just pick `model=Qwen3ASR()` and `recipe=Qwen3ASR_Recipe(...)`,
    and bring the reward.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run python tutorials/rl/005_audio_asr/005_audio_asr.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        LibriSpeechASRDataset,
        Qwen3ASR,
        Qwen3ASR_Recipe,
        TrainConfig,
        WandbConfig,
        evaluate_asr,
    )


@markdown
def _dataset_intro():
    """
    ## Load LibriSpeech audio

    `LibriSpeechASRDataset` pulls a few clips from the standard LibriSpeech dummy
    set. Each row is a prompt with an `<audio>` placeholder, the clip itself (as a
    base64 `data:audio/wav` URI), and the reference transcript as the label. As a
    `MultimodalDataset` it tells slime to forward the audio column to the rollout —
    the same passthrough images and video use.
    """


@code
def _dataset():
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

    Qwen3-ASR is already near-perfect on clean LibriSpeech, so the `Qwen3ASR_Recipe`
    defaults sample many transcripts per clip at temperature 1.0 — that's what gives
    the GRPO group enough within-group WER variance to produce a non-zero gradient.
    """


@code
def _reward():
    async def wer_reward(args, sample, **kwargs) -> float:
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

    `Qwen3ASR_Recipe` carries the ASR-specific defaults — the transcription
    rollout, padded (bshd) batches, the lighter SGLang memory fraction, and the
    many-samples/high-temperature settings that surface reward variance — so the
    recipe you write only sets the reward and (optionally) W&B logging. It defaults
    to a `H100:2` single node; pass `actor_num_gpus_per_node=8` (and a larger
    `num_rollout`) to use a full node.

    `TrainConfig.train()` builds the Modal app, runs GRPO, and exports the trained
    model as a standard HuggingFace checkpoint (audio tower included).
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3ASR(),
        dataset=dataset,
        recipe=Qwen3ASR_Recipe(
            custom_rm_function=wer_reward,
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

    Training exports a standard HF checkpoint, so we can evaluate it directly.
    `evaluate_asr` serves the trained model, transcribes the clips, scores −WER,
    and writes the per-clip results — audio, reference, hypothesis, WER — to the
    gym dashboard's **Evals** panel, tied to this run. Open the dashboard
    (`training-gym setup` prints the URL) and you'll see each clip with an audio
    player next to its reference and score.
    """


@code
def _eval():
    eval_result = evaluate_asr(train_result, dataset)
    print(f"Eval: mean WER {eval_result['mean_wer']:.3f} over {eval_result['rows']} clips")
