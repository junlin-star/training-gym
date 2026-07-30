from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe
from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe
from modal_training_gym.train_recipes.stitch_recipe.serve import StitchServeRecipe
from modal_training_gym.train_recipes.stitch_recipe.train import SlimeStitchTrainRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Moonlight_16B_A3B_Stitch_Train(SlimeStitchTrainRecipe):
    """2×8×H200 multi-node slime actor cluster for Moonlight-16B-A3B (TP4/EP8).

    The architecture comes from slime's dedicated model script rather than
    ``ModelArchitecture`` flags — Moonlight is DeepSeek-V3 shaped (MLA, 64 routed
    + 2 shared experts, sigmoid router with expert bias), the same ladder as
    Kimi K2, and slime ships ``scripts/models/moonlight.sh`` for it.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu_type: str = "H200"
    region: str | None = "us"
    # CPU-offloaded optimizer state for 64 experts wants host RAM headroom.
    memory: int | tuple[int, int] | None = (128, 2_097_152)
    actor_num_nodes: int = 2
    actor_num_gpus_per_node: int = 8

    # ── Model (arch comes from slime's model script) ─────────────────────────
    slime_model_script: str = "scripts/models/moonlight.sh"
    hf_checkpoint: str = "moonshotai/Moonlight-16B-A3B-Instruct"
    ref_load: str = "moonshotai/Moonlight-16B-A3B-Instruct"
    megatron_to_hf_mode: str = "bridge"

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
    # A rollout step issues rollout_batch_size * n_samples_per_prompt concurrent
    # requests; kept at half upstream's 64 so the pool's warm replicas cover a
    # whole step (see Moonlight_16B_A3B_Stitch_Serve.min_containers).
    rollout_batch_size: int = 32
    rollout_max_response_len: int = 4096
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    # R3 (arxiv 2510.11370): replay the rollout engine's expert routing in the
    # train forward.
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


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Moonlight_16B_A3B_Stitch_Serve(StitchServeRecipe):
    """Flash pool of single-H200 SGLang replicas serving Moonlight rollouts.

    Multi-latent attention keeps the KV cache small, so a replica is still a
    single H200 even at 16B — but a 16B MoE takes minutes to load, which is what
    the warm-replica floor below is about.
    """

    sglang: SglangRecipe = field(
        default_factory=lambda: SglangRecipe(
            gpu="H200",
            tp=1,
            context_length=8192,
            # An in-place apply loads incoming shards beside the live weights, so
            # the engine has to leave room: upstream's 0.85 (with an engine it
            # drains) leaves ~200 MB free on an H200 and the apply OOMs. Both
            # levers are needed — the fraction alone doesn't help, because the KV
            # pool then autosizes into whatever the fraction freed. Capping it at
            # what full concurrency can address (64 requests × 8k ctx, ~15 GB of
            # MLA KV instead of ~59 GB) leaves ~64 GB for the apply.
            mem_fraction_static=0.72,
            extra_server_args={
                "--max-total-tokens": "524288",
                # Boot on the default loader, not fastsafetensors: SGLang reuses
                # the server's load format (and its --model-loader-extra-config)
                # for online updates too, and the fastsafetensors copier stages
                # each shard block through an 800 MB GPU buffer. A replica
                # applying a delta while it serves has nowhere to put that — the
                # engine's static pool plus in-flight decode leave ~200 MB free —
                # so it OOMs, while an idle replica squeaks through. Streaming
                # tensor by tensor costs a slower cold start and applies under
                # load.
                "--weight-loader-drop-cache-after-load": "",
                # Routing replay: slime runs no local engine in publish-only
                # mode, so the served engine has to be told to return routed
                # experts.
                "--enable-return-routed-experts": "",
            },
        )
    )
    concurrency: int = 64
    # A rollout step issues 32 × 8 = 256 concurrent requests and a replica
    # accepts `concurrency` = 64. Keep the whole batch's worth of replicas warm:
    # a 16B MoE takes ~4min to load, far longer than slime's own HTTP retry
    # budget, so letting Flash cold-scale into the first rollout step 503s the
    # run out.
    min_containers: int = 4
    max_containers: int | None = 4


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Moonlight_16B_A3B_Stitch_Recipe(StitchRecipe):
    """Moonlight-16B-A3B disaggregated GRPO: 2×8×H200 trainer + Modal Flash pool.

    Mirrors the stitch ``moonlight`` cookbook config — the cheap rung of the
    Kimi-K2.6 ladder. Two things the Qwen3-4B recipe doesn't exercise: a
    multi-node RDMA trainer with expert parallelism, and MoE routing replay,
    where the train forward reuses the rollout engine's expert routing.
    """

    train: SlimeStitchTrainRecipe = field(  # type: ignore[assignment]
        default_factory=Moonlight_16B_A3B_Stitch_Train
    )
    serve: StitchServeRecipe = field(default_factory=Moonlight_16B_A3B_Stitch_Serve)
