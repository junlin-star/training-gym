# 005 — Audio GRPO on Qwen3-ASR-1.7B

Reinforcement learning (GRPO) on an **audio** model end-to-end through the gym in training-gym. The loop:

1. Load LibriSpeech audio clips via `LibriSpeechASRDataset` 
  - a `MultimodalDataset` with `modality="audio"` 
2. slime serves Qwen3-ASR on SGLang's `/v1/audio/transcriptions` endpoint; our custom `transcription_rollout` posts each clip's audio and collects the transcript.
3. `wer_reward` scores the transcript as **−WER** against the reference.
4. That reward drives a GRPO update through slime/Megatron.


## Run it

This model is small enough that you don't need a whole node of GPUs. To run the minimal version, try:
```bash
# 2×H100 single node — quickest test
uv run python tutorials/rl/005_audio_asr/train_qwen3_asr.py
```
For a whole-node version, add `--scale` (8×H100, same example, longer run):
```bash
# 8×H100 single node
uv run python tutorials/rl/005_audio_asr/train_qwen3_asr.py --scale
```

Multinode coming soon!

## Files

| File | Role |
|------|------|
| `train_qwen3_asr.py` | the `TrainConfig`; `build_train_config(...)` Defaults to minimal `H100:2` demo; `--scale` uses `H100:8` |
| `qwen3_asr_model.py` | `Qwen3ASR` gym `ModelConfig` (Qwen3 dense backbone arch) |
| `audio_data.py` | `LibriSpeechASRDataset(MultimodalDataset)` |
| `asr_rollout.py` | `transcription_rollout` |
| `reward.py` | `wer_reward` (−WER) |
| `_native_qwen3asr_compat.py` | idempotent image-build workarounds for upstream gaps (below) |

## Upstream Workarounds

The native stack (sglang 0.5.12 + megatron-bridge 0.5.0 + transformers 5.6) supports
Qwen3-ASR, but four small upstream gaps are patched at image build by `_native_qwen3asr_compat.py`:

1. **bridge config validate-order** — `hf_qwen3_asr` reads `self.thinker_config` in `get_text_config()`, which transformers 5.6 calls during `super().__init__()` before it's set.
2. **slime processor loading** — `qwen3_asr` falls to the GLM-4V processor and crashes; instead uses SGLang's `Qwen3ASRProcessor`.
3. **bridge `pg_collection`** — `Qwen3ASRThinkerModel` hard-raises on a `None` `pg_collection`; default to `use_mpu_process_groups()`.
4. **HF export** — slime's MB→HF tool dispatches `"qwen3"` to the qwen2 converter,
   which can't map the audio tower (`Unknown parameter name: ...thinker.audio_model...`).
   Ship a `qwen3_asr` converter into slime's `megatron_to_hf` (mirrors megatron-bridge's
   mapping_registry: standard Qwen3 decoder splits + `thinker.audio_model.* →
   thinker.audio_tower.*` passthrough) and dispatch to it before the generic `qwen3`
   branch. Uses slime's own checkpoint loader (no `megatron.bridge.training`, avoiding a
   version skew with the bundled Megatron-LM). The trained model exports as a standard,
   loadable HF checkpoint.

Plus one config choice in `train_qwen3_asr.py`:

- **`qkv_format="bshd"`** (+ `use_dynamic_batch_size=False`, an explicit `micro_batch_size`): the native bridge's forward doesn't implement THD/packed sequences, so we use padded batches (packed batching is a capability the bridge hasn't wired through yet).

## Known caveat — GRPO stability at scale

Validated green end-to-end at both 2×H100 and 8×H100 on this data slice. On a larger slice (`n_clips=64`) we hit a `NaN` grad from a numerically pathological clip — with `micro_batch_size=1` (forced by the no-packing limitation) a single bad sample isn't diluted. `clip_grad=1.0` absorbs ordinary spikes; the proven slice trains clean. Tuning GRPO stability on larger/noisier audio data (advantage handling, packing) is a good follow-up.
