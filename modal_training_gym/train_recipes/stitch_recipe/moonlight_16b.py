from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Moonlight_16B_A3B_Stitch_Recipe(StitchRecipe):
    """Moonlight-16B-A3B disaggregated GRPO: 2×8×H200 trainer + Modal Flash pool.

    Mirrors the stitch ``moonlight`` cookbook config — the cheap rung of the
    Kimi-K2.6 ladder. Two things the Qwen3-4B recipe doesn't exercise: a
    multi-node RDMA trainer with expert parallelism, and MoE routing replay
    (arxiv 2510.11370), where the train forward reuses the rollout engine's
    expert routing (``--enable-return-routed-experts`` on the served engine).

    Multi-latent attention keeps the KV cache small, so a rollout replica is
    still a single H200 even at 16B.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu_type: str = "H200"
    region: str | None = "us"
    # CPU-offloaded optimizer state for 64 experts wants host RAM headroom.
    memory: int | tuple[int, int] | None = (128, 2_097_152)
    actor_num_nodes: int = 2
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus_per_engine: int = 1
    # A rollout step issues ``rollout_batch_size * n_samples_per_prompt`` = 256
    # concurrent requests, and a replica accepts ``sglang_server_concurrency``
    # = 64. Keep the whole batch's worth of replicas warm: a 16B MoE takes ~4min
    # to load, far longer than slime's own HTTP retry budget, so letting Flash
    # cold-scale into the first rollout step 503s the run out.
    rollout_min_containers: int = 4
    rollout_max_containers: int | None = 4

    # ── Model (arch comes from slime's model script, not ModelArchitecture) ─
    # fastsafetensors is the fastest cold start, but its copier stages each shard
    # block through a GPU buffer (800 MB at a time) — unaffordable beside a live
    # engine, and pointless for a ~0.5 GB sparse delta.
    sidecar_disk_load_format: str = "auto"

    slime_model_script: str = "scripts/models/moonlight.sh"
    hf_checkpoint: str = "moonshotai/Moonlight-16B-A3B-Instruct"
    ref_load: str = "moonshotai/Moonlight-16B-A3B-Instruct"
    megatron_to_hf_mode: str = "bridge"

    # ── Rollout pool (SGLang) ───────────────────────────────────────────────
    sglang_server_concurrency: int = 64
    sglang_server_args: dict[str, str] = field(
        default_factory=lambda: {
            "--load-format": "fastsafetensors",
            "--model-loader-extra-config": '{"enable_gds":false}',
            "--weight-loader-drop-cache-after-load": "",
            "--context-length": "8192",
            "--mem-fraction-static": "0.85",
            # An in-place apply has to fit beside the live weights, and SGLang's
            # KV pool otherwise autosizes into every spare byte (it left ~8 MB
            # free on an H200 and the apply OOM'd mid-flight). Cap the pool at
            # what full concurrency can actually address — 64 requests × 8k ctx —
            # which is ~15 GB of MLA KV instead of ~59 GB.
            "--max-total-tokens": "524288",
            # Routing replay: slime runs no local engine in publish-only mode, so
            # the served engine has to be told to return routed experts.
            "--enable-return-routed-experts": "",
        }
    )

    # ── Rollout / algorithm ─────────────────────────────────────────────────
    # Upstream stitch runs moonlight one-step-async with bounded-lag requests, but
    # that regime needs the trainer to wake the pool on publish, and Flash wake is
    # a deployed-app lookup: in the single-call ephemeral flow replicas only
    # self-sync on their reconcile poll, so they fall behind the lag bound and
    # rollout requests 409 until slime gives up. Publish synchronously instead —
    # the trainer waits for the pool at each version, which is what this example
    # is demonstrating anyway.
    async_mode: bool = False
    update_weights_interval: int = 1
    num_rollout: int = 5
    rollout_batch_size: int = 32
    rollout_max_response_len: int = 4096
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    # R3: replay the rollout engine's expert routing in the train forward.
    use_rollout_routing_replay: bool = True

    # ── Trainer parallelism (world = TP4 × DP4 = 16) ────────────────────────
    tensor_model_parallel_size: int = 4
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    max_tokens_per_gpu: int = 8192

    # ── Optimizer (CPU offload keeps GPU state tiny for ~3B active) ──────────
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    rm_type: str | None = "math"
    save_interval: int = 20
