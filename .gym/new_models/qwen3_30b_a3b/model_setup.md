# Qwen3-30B-A3B NVFP4 on the stitch (miles disagg) path

Phase-1 artifact for `Qwen3_30B_A3B_Stitch_Recipe`. The subject is not the model —
`Qwen/Qwen3-30B-A3B` is already supported everywhere in this stack — it is the
NVFP4 QAT + sparse-delta serving contract ported from stitch's own cookbook config
`cookbook/miles_disagg/configs/qwen3_30b_a3b_nvfp4_46.py` (stitch `697cda7`).

## Is the model (or its architecture) already validated?

Yes, on all three legs:

- **miles / Megatron**: miles ships `scripts/models/qwen3-30B-A3B.sh`, which this
  recipe sources (`miles_model_script`), so the architecture args are upstream's
  verbatim and `_model_to_fields` is informational only. Upstream's own
  `run_qwen3_30b_a3b.py` runs the same TP4/SP/PP1/CP1 + EP8 topology.
- **SGLang**: Qwen3 MoE is a first-class SGLang architecture. What is version
  sensitive is the *pool* contract: `--enable-cpu-weight-cache`,
  `--weight-loader-prefetch-checkpoints`, `/stage_weight_update`, and
  `--enable-return-routed-experts`, which come from the pinned fork
  `modal-projects/sglang@1051a95a` (branch `stitch-sglang-v0.5.16`, base image
  `lmsysorg/sglang:v0.5.16`).
- **NVFP4 QAT**: needs TransformerEngine 2.17 plus the dequantized-backward fix
  and the TE-direct NVFP4 quantizer (miles PR #1261). Both are in the pinned
  trainer image `radixark/miles:dev-202607290235` + `modal-projects/miles@1eb7520`.

## Does upstream support postdate the pinned images?

No — this is the inverse of the usual slime problem. The pins are chosen *from*
stitch main, not inherited: the recipe's features are the reason for the pins.

| component | pin | why |
|---|---|---|
| stitch | `697cda79666fad8cfa7ab4a98b9f9f4f11cce1da` | `miles_disagg` + `Store`/`Engine`/`Pool` APIs |
| miles image | `radixark/miles:dev-202607290235` | TE 2.17 + Megatron with native `--fp4-format` |
| miles fork | `1eb7520018446cb94b7406715f66dff1a271b53b` | bulletin protocol + PR #1261 quantizer |
| SGLang fork | `1051a95a6ab16773037f8795a51aa03a1664a3b2` | stage/commit weight updates, CPU weight cache |
| Megatron patches | `megatron-r3-dispatch`, `megatron-hdo-dp-reshardable-step` | R3 dropless dispatch; reshardable CPU-offload optimizer step |

The Megatron patches are applied to the image's source checkout at container
start (not build time) because every Ray actor imports its own copy.

## Train configuration plan

- Trainer: 1×8 B200, TP4 × DP2, EP8/ETP1, SP on, dynamic batch,
  `max_tokens_per_gpu=8192`, full uniform recompute (1 layer).
- Optimizer: Adam 1e-6 constant, CPU-offloaded with D2H/H2D overlap and
  precision-aware states — ~3B active params keeps GPU optimizer state small.
- Algorithm: GRPO + TIS, `eps_clip 0.2 / 0.28`, `kl_loss_coef 0.0`,
  DAPO-Math-17k, `rm_type=deepscaler`, `n_samples_per_prompt=8`,
  `rollout_batch_size=32`, `rollout_max_response_len=12288`.
- Precision: NVFP4 on the routed-expert GEMMs only
  (`*.mlp.experts.linear_fc1/fc2` matchers), last 7 of 48 layers BF16, attention
  BF16 on FlashAttention (TE's cuDNN backward fails on these dynamic shapes).
- Serving: 1 B200 per replica (NVFP4 packs ~30B into ~17 GB), `min 1 / max 3`
  containers, per-container target 24 inputs against a trainer client concurrency
  of 128 so a 256-request wave scales the pool out rather than queueing on one
  engine. Delta applies run in `cpu` mode against the pinned weight cache,
  `commit_mode=in_place`, no cache flush.
- Checkpoints: `prepare_checkpoints` materializes BF16 masters at
  `/checkpoints/qwen3-30b-a3b-bf16` (the trainer's bridge `ref_load`) and converts
  the served baseline at `/checkpoints/qwen3-30b-a3b-nvfp4-46` with
  `miles/tools/convert_hf_to_nvfp4.py` under the trainer's exact `NVTE_*` env, so
  served packing == export packing and the XOR delta stays sparse.

Deliberate deviation from upstream: `async_mode=False`. Upstream runs async with
bounded lag 1, which requires the publish hook to wake replicas by *deployed* app
name; the single-call ephemeral flow has no such lookup, so replicas would only
self-sync on their poll and fall outside the lag bound (observed as rollout
`409 WeightVersionNotReady` on the earlier BF16 attempt). Consequently the request
gate runs `mode=exact, lag=0` with a 240×1s retry budget.

## Expected timings (to be checked against the run, Phase 2)

Estimates, not measurements. Reference points: Qwen3-4B on this path staged in
6–9s / committed in 1.5–2.2s per replica; Moonlight-16B-A3B (BF16, 1×H200
replicas) staged 19–43s / committed ~24s.

| substep | expectation |
|---|---|
| trainer image pull + Megatron patches | 5–10 min cold |
| `prepare_checkpoints` (download + BF16 copy + NVFP4 convert) | 45–90 min, once per checkpoints Volume |
| SGLang replica cold start (NVFP4, prefetch loader) | 3–6 min |
| rollout wave (32 prompts × 8 samples, ≤12288 tok) | 3–8 min |
| train step (TP4/EP8, CPU-offload optimizer, recompute) | 2–5 min |
| weight sync per replica (~17 GB base, sparse XOR, cpu mode) | stage 10–30s, commit 5–15s |

NVFP4 should sync *faster* than Moonlight BF16 despite the larger model: the base
is ~17 GB rather than ~32 GB and the apply lands in the pinned CPU weight cache.

## One-step smoke feasibility

Feasible, but it is 8 B200 for the trainer plus 1–3 B200 for the pool (≤11
concurrent) and a one-off ~1h GPU-hour checkpoint conversion. The signals worth
gating on, in order:

1. `prepare_checkpoints` completes and the NVFP4 dir carries a `quantization_config`.
2. Trainer reaches `train.py` step 0 with the TE precision config applied (no
   `CUDNN_STATUS_BAD_PARAM`, no NVFP4 matcher misses).
3. A replica boots the local NVFP4 dir and serves a rollout.
4. First delta publishes and every replica applies it in place (`reason=None`).
5. `train_rollout_logprob_abs_diff` is the precision-contract check — upstream
   reports ~0.031–0.044. A much larger value means an `NVTE_*`/`FLASHINFER_*` drift,
   not a training bug.
