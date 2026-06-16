# Qwen3-Omni-30B with Vime — scoping & journal

Status: **partial — training bridge works; omni rollout blocked on an upstream
vLLM-omni bug.** Branch `ben/vime-qwen3-omni-30b`, off `ben/VimeRecipe` / #148.
Goal: GRPO-train **Qwen3-Omni-30B-A3B** with vime on a multimodal (audio+image+text)
dataset, serve+eval, and visualize all modalities on the dashboard.

## TL;DR state
- ✅ **vime multimodal foundation** (image): `MultimodalDataset` `multimodal_keys` →
  vime rollout → bridge train. Validated end-to-end with **Qwen3-VL-4B image+text GRPO**
  on Modal (committed). This is the reusable base for omni.
- ✅ **Qwen3-Omni-30B `qwen_omni` training bridge loads the 30B MoE Thinker into
  Megatron** (the research-gated unknown — resolved). Cleared arch validation, MoE
  expert-parallel args, the chat-template path, and the audio-tower TP-divisibility.
- ❌ **Blocked: the omni vLLM *rollout* engine** fails at init with
  `RuntimeError: cu_seqlens_q must be on CUDA` — an upstream vLLM bug in the
  Qwen3-Omni vision/audio encoder warmup (vLLM 0.22 in the image; known omni encoder
  issues exist upstream). Not a gym/vime config bug. Until rollout serves omni, GRPO
  can't run. Options: bump/patch vLLM in the image, a serving workaround
  (limit-mm / encoder attn backend), or upstream fix.
- 📋 **Not started** (gated on rollout): audio in the rollout (vime is vision-only),
  `convert_qwen3omni_to_hf` for serving, the audio+image+text dataset, dashboard.

## Feasibility probe (vime image `inferactinc/public:vime-latest`)
- transformers **5.9.0**, vllm **0.22.0**, torch 2.11.0+cu129.
- transformers has `Qwen3OmniMoeConfig` / `Qwen3OmniMoeForConditionalGeneration` /
  `Qwen3OmniMoeThinkerForConditionalGeneration`, and `Qwen3VLMoe*`.
- `AutoConfig(Qwen/Qwen3-Omni-30B-A3B-Instruct)` → `Qwen3OmniMoeConfig`, subconfigs
  `thinker_config` (perception→text — **what we train**), `talker_config` (speech
  out — not needed for GRPO), `code2wav_config` (vocoder — not needed).
- vLLM **serves** `Qwen3OmniMoeForConditionalGeneration` (+ `Qwen3VLMoe`, `Qwen3ASR`,
  `Qwen2_5Omni`). → **rollout side is supported.**
- megatron→HF: vime ships `megatron_to_hf/qwen3_vl.py` (vision+text MoE) and
  megatron.bridge has a `qwen3_asr` model dir (audio+text), but **no `qwen3_omni`**.
  → **training/convert bridge for the omni Thinker is the patch gap.**

## Architecture
Qwen3-Omni Thinker = Qwen3-30B-A3B **MoE text backbone** + **SigLIP-2 vision** encoder
+ **AuT audio** encoder (≈ Qwen3-VL-30B-A3B + an audio tower). So:
- the text/vision half ≈ vime's supported **Qwen3-VL-30B-A3B** (qwen3_vl bridge),
- the audio half ≈ the gym's existing **Qwen3-ASR** work (audio tower via megatron.bridge).

## Plan (one experiment at a time; de-risk cheap → expensive)
**Phase 1 — validate vime multimodal plumbing on a small VL model (cheap).**
Build the gym's vime multimodal path (multi-media `MultimodalDataset`, `--multimodal-keys`,
VLM rollout, VL bridge load + megatron→HF convert) on **Qwen3-VL-4B-Instruct**, image+text
GRPO, single node. Mirrors vime's `examples/geo3k_vlm_multi_turn`. Isolates "does gym×vime
multimodal work" from the omni audio patch.

**Phase 2 — Qwen3-Omni-30B-A3B (the target).**
Add: `Qwen3_Omni_30B` model config (Thinker text-MoE `architecture` + vision/audio specs),
a vime omni recipe preset, an **audio+image+text** dataset, and the **omni Thinker megatron
bridge patch** (extend qwen3_vl with the AuT audio tower; reference the gym's qwen3_asr
audio handling). Multinode. Train → serve (vLLM omni) → eval.

**Dashboard** — EvalsPage already renders audio/image (from the ASR work). Multi-modality
rows may need a small extension; per the ask, bundle if cheap, else a follow-up PR.

## Open risks
- Omni Thinker megatron bridge (HF→megatron load for training, megatron→HF for serve) —
  no upstream mapping; the core patch. May need to combine qwen3_vl + qwen3_asr logic.
- Multimodal rollout: does vime's vLLM rollout pass audio (not just image) to the engine?
  geo3k proves image; audio is unverified.
- Training only the Thinker (ignore Talker/code2wav) cleanly.
- 30B MoE multinode cost/topology.

## Plumbing facts (from vime source)
- Multimodal rollout: vime's **default** vLLM rollout forwards `multimodal_inputs["images"]`
  as image content when `--multimodal-keys '{"image":"images"}'` is set (no custom fn for
  single-turn). **Audio is NOT in that default path** (vllm_rollout only handles images) →
  omni audio needs a rollout extension. Prompt must contain the modality placeholder
  (`<image>`/`<audio>`/`<video>`). Image column accepts data-URI/URL/path → PIL.
- Gym wiring: `VimeConfig._dataset_to_fields` now forwards `ds.multimodal_keys` (added).
- Conversion (megatron→HF) dispatches by `model_name` to per-model converters incl.
  `convert_qwen3vl_to_hf` (maps `vision_model.*`→`model.visual.*`). **No `qwen3_omni`
  converter** → must add one (extend VL + audio tower) and patch it into the vime image.

## Omni bridge: GOOD NEWS — training bridge likely exists
megatron.bridge ships model dirs incl. **`qwen_omni`**, `qwen_vl`, `qwen3_asr`,
`qwen_audio` (`.../megatron/bridge/models/`). So the **training-side** HF→megatron
load for the Qwen3-Omni Thinker is probably built-in (AutoBridge dispatch), unlike
the megatron→HF *converter* (no `qwen3_omni` in vime's `megatron_to_hf` → serving
gap remains).

Qwen3-Omni-30B-A3B **Thinker** text backbone (exact, from config): num_layers=48,
hidden=2048, intermediate_size=768 (==moe_intermediate_size; all-MoE), heads=32,
kv_heads=4, head_dim=128, vocab=**152064**, rms_norm_eps=1e-6, 128 experts top-8,
tie_word_embeddings=False. `rope_theta`/`rope_scaling` **absent** (MRoPE via
`rope_parameters`) — validation behavior TBD (try rotary_base=1e6). MoE/vision/audio
sub-configs are read from the HF checkpoint (vime only validates the 6 base text fields).

Cost lever: vime's `hf_validate_args` runs at `parse_args` **before** the 60GB
download, so arch flags can be cleared on a cheap **1-GPU fast-fail** run, then go
multinode for the real 30B train.

## Omni patch surface (Phase 2)
1. Training bridge: HF→megatron load of the omni Thinker (VL text-MoE + vision + AuT audio).
2. megatron→HF converter `convert_qwen3omni_to_hf` (qwen3_vl + audio tower), registered in dispatch.
3. Rollout: pass audio (+image) to the vLLM engine (extend default rollout / custom fn).
4. Dataset: multi-modality (image+audio+text); extend single-modality `MultimodalDataset`.

## Audio rollout gap (confirmed)
vime's rollout + data pipeline are **vision-only**: `process_vision_info` and the
image path in `vllm_rollout.py` (line ~303 reads only `multimodal_inputs["images"]`).
No `process_audio_info` / no audio passed to the engine. So audio needs a patch:
extract audio from the `<audio>`-placeholdered messages and send it to vLLM as
`input_audio` content (vLLM omni accepts it). Patch into the vime image like the
gym's slime ASR patches. This is the hardest omni-specific piece — gated on the
image+text omni run proving the `qwen_omni` bridge first.

## Experiments log
- ✅ Probe (CPU): versions + omni support mapped. run `ap-gHxzjIHu4HYbMKJUL97LcZ`.
- Phase 1 Qwen3-VL-4B image+text GRPO (1×H100), iterating:
  - ❌ rope_theta mismatch (VL-4B uses 5e6, not Qwen3-4B's 1e6) — vime `hf_validate_args`.
  - ❌ `architecture=None` → all base fields mismatch (vime *requires* the 6 base text
    fields: hidden_size/heads/layers/ffn/norm_eps/rope_theta; MoE/vision read from HF).
  - ❌ vLLM KV cache OOM: VL default max_model_len=262144 (256K) → 36 GiB KV. Fix:
    `--vllm-max-model-len 8192`.
  - ✅ **Multimodal rollout VALIDATED**: image → vime default rollout → vLLM VL inference
    → `<|vision_start|><|image_pad|><|vision_end|>` prompt, model answered "yellow",
    **reward 1.0**. The gym×vime image plumbing works end to end.
  - ❌ then CUDA OOM at the megatron train step (VL-4B optim states + colocated vLLM >80GB
    on 1 GPU). Fix in flight: `optimizer_cpu_offload` (+ precision-aware/overlap), vllm
    util 0.5. → the same memory lever omni-30B needs.
  - ✅ re-run `bxz2simu4`: **VL image+text GRPO trains end-to-end** (8 steps, image
  rollout reward 1.0, optimizer offload, checkpoint `convex-bit-2dc9af577e02`).
  Multimodal foundation validated + committed.
- Phase 2 Qwen3-Omni-30B-A3B image+text on **1×8 H100** (TP4/EP8/SP + offload, vime's
  proven 30B-A3B parallelism):
  - ✅ arch cleared `hf_validate_args` (MoE/rope/intermediate checks skipped as predicted).
  - ❌ `num_experts must be non None to use expert model parallelism` — MoE fields
    aren't auto-read; must pass `--num-experts` etc. Added to the model arch (128
    experts, moe_ffn 768, top-8, softmax) + `moe_token_dispatcher_type=alltoall`.
  - ✅ `bqcqearo7`: **`qwen_omni` bridge loaded the 30B Thinker into Megatron** (MoE
    model built, num_experts=128 accepted). Failed at RolloutManager init:
    `tokenizer.chat_template is not set` — omni keeps its template on the processor,
    not the bare tokenizer vime templates with client-side.
  - Fix: `apply_chat_template=False` (vime still builds the multimodal message list
    via multimodal_keys; vLLM applies the omni template server-side — same as the
    ASR config).
  - ✅ `bqzyveu7p`: chat-template fixed. Failed in vLLM rollout engine: the omni
    **audio tower** has 20 attn heads → `divide(20, tp=8)` assertion (rollout TP=8).
    vLLM loads the full omni (audio+vision towers) even for image+text rollout.
  - Fix: `rollout_num_gpus_per_engine=4` (TP=4 divides both 20 and 32). ⏳ `omni_smoke4`.
  - ❌ `few-circuit-0fe326978cc4` (rollout TP=4): audio-tower heads OK; new wall —
    omni vLLM engine init `RuntimeError: cu_seqlens_q must be on CUDA` (encoder warmup,
    upstream vLLM-omni bug). **Stopped expensive omni iteration here** (the major wall).
