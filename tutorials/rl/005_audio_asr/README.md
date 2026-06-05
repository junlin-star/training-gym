# 005 — Audio GRPO on Qwen3-ASR-1.7B

Reinforcement learning (GRPO) on an **audio** model end-to-end through the gym, on the
native training-gym `main` stack. The loop:

1. Load LibriSpeech audio clips via `LibriSpeechASRDataset` (a `MultimodalDataset` with
   `modality="audio"` — the gym's multimodal passthrough wires `--multimodal-keys`).
2. slime serves Qwen3-ASR on SGLang's `/v1/audio/transcriptions` endpoint; our custom
   `transcription_rollout` posts each clip's audio and collects the transcript.
3. `wer_reward` scores the transcript as **−WER** against the reference.
4. That reward drives a GRPO update through slime/Megatron.

This is the first **audio** RL example in the gym and exercises the multimodal path
distinct from text/vision.

## Run it

```bash
# 2×H100 single node — the canonical demo (quick, bounded reward curve)
uv run python tutorials/rl/005_audio_asr/train_qwen3_asr.py

# 8×H100 single node — same example scaled out
uv run python tutorials/rl/005_audio_asr/train_qwen3_asr_scale.py
```

Both stream reward (−WER) + `pg_loss`/`grad_norm` to W&B (project `qwen3-asr-rl`).

## Files

| File | Role |
|------|------|
| `train_qwen3_asr.py` | the `TrainConfig`; `build_train_config(...)` is parametrized (GPU count, batch, clips) so the scale variant reuses it |
| `train_qwen3_asr_scale.py` | thin 8×H100 wrapper over `build_train_config` |
| `qwen3_asr_model.py` | `Qwen3ASR` gym `ModelConfig` (Qwen3 dense backbone arch) |
| `audio_data.py` | `LibriSpeechASRDataset(MultimodalDataset)` |
| `asr_rollout.py` | `transcription_rollout` — drives the audio-transcription endpoint |
| `reward.py` | `wer_reward` (−WER) — lives in its own module so it imports cleanly in the training container |
| `_native_qwen3asr_compat.py` | idempotent image-build workarounds for upstream gaps (below) |

## Native-stack notes (upstream gaps worked around)

The native stack (sglang 0.5.12 + megatron-bridge 0.5.0 + transformers 5.6) supports
Qwen3-ASR, but three small upstream gaps are patched at image build by
`_native_qwen3asr_compat.py` (each should be reported upstream, then the file can go):

1. **bridge config validate-order** — `hf_qwen3_asr` reads `self.thinker_config` in
   `get_text_config()`, which transformers 5.6 calls during `super().__init__()` before
   it's set. Guarded.
2. **slime processor loading** — `qwen3_asr` falls to the GLM-4V processor and crashes;
   use sglang's `Qwen3ASRProcessor`.
3. **bridge `pg_collection`** — `Qwen3ASRThinkerModel` hard-raises on a `None`
   `pg_collection`; default to `use_mpu_process_groups()`.

Plus two config choices in `train_qwen3_asr.py`:

- **`qkv_format="bshd"`** (+ `use_dynamic_batch_size=False`, an explicit
  `micro_batch_size`): the native bridge's forward doesn't implement THD/packed
  sequences, so we use padded batches. (Packed batching is a capability the bridge
  hasn't wired through yet — worth restoring upstream.)
- **`disable_hf_conversion=True`**: the bridge's MB→HF export doesn't map the audio
  tower, so we skip the post-training HF export (the example's deliverable is the
  training loop + reward curve, not a checkpoint).

## Known caveat — GRPO stability at scale

Validated green end-to-end at both 2×H100 and 8×H100 on this data slice. On a larger
slice (`n_clips=64`) we hit a `NaN` grad from a numerically pathological clip — with
`micro_batch_size=1` (forced by the no-packing limitation) a single bad sample isn't
diluted. `clip_grad=1.0` absorbs ordinary spikes; the proven slice trains clean. Tuning
GRPO stability on larger/noisier audio data (advantage handling, packing) is a good
follow-up.
