from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe
from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe
from modal_training_gym.train_recipes.stitch_recipe.serve import StitchServeRecipe
from modal_training_gym.train_recipes.stitch_recipe.train import SlimeStitchTrainRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_30B_A3B_Stitch_Train(SlimeStitchTrainRecipe):
    """1×8×H200 slime actor cluster for Qwen3-30B-A3B (TP4/EP8), BF16.

    Trainer topology and algorithm come from stitch's
    ``miles_disagg/configs/qwen3_30b_a3b_nvfp4_46`` config, which is itself
    slime's ``run-qwen3-30B-A3B.sh`` recipe under no-colocate. The architecture
    comes from slime's model script (128 routed experts, no shared expert,
    softmax router) rather than ``ModelArchitecture`` flags.

    What is *not* ported from that config is its subject: NVFP4 QAT on B200
    (TE row-scaled NVFP4 experts, dequantized backward, 4-over-6 block scaling,
    and a separately quantized served base). That path runs on miles, which this
    launcher does not wire, so this preset trains and serves BF16.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu_type: str = "H200"
    region: str | None = "us"
    # CPU-offloaded optimizer state for 128 experts wants host RAM headroom.
    memory: int | tuple[int, int] | None = (128, 2_097_152)
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    # ── Model (arch comes from slime's model script) ─────────────────────────
    slime_model_script: str = "scripts/models/qwen3-30B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3-30B-A3B"
    ref_load: str = "Qwen/Qwen3-30B-A3B"
    megatron_to_hf_mode: str = "bridge"

    # ── Rollout / algorithm ─────────────────────────────────────────────────
    # Synchronous publish, for the same reason as the Moonlight preset: async
    # bounded-lag requests need the trainer to wake the pool on publish, and
    # Flash wake is a deployed-app lookup that the single-call ephemeral flow
    # doesn't have, so replicas fall behind the lag bound and requests 409.
    async_mode: bool = False
    update_weights_interval: int = 1
    num_rollout: int = 3
    # A rollout step issues rollout_batch_size × n_samples_per_prompt = 256
    # concurrent requests; the pool keeps that much concurrency warm.
    rollout_batch_size: int = 32
    # Upstream runs 12288 so math traces terminate instead of truncating; kept
    # short here because this preset's default budget is a bring-up smoke.
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 0.8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    # R3 (arxiv 2510.11370): replay the rollout engine's expert routing in the
    # train forward.
    use_rollout_routing_replay: bool = True

    # ── Trainer parallelism (world = TP4 × DP2 = 8, EP over the node) ───────
    tensor_model_parallel_size: int = 4
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    max_tokens_per_gpu: int = 8192
    # Qwen3 MoE has no MLA, and TE's cuDNN fused-attention backward fails on the
    # dynamic sequence shapes here (CUDNN_STATUS_BAD_PARAM), which is why both
    # upstream configs pin FlashAttention.
    attention_backend: str = "flash"

    # ── Optimizer (CPU offload keeps GPU state tiny for ~3B active) ──────────
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    rm_type: str | None = "deepscaler"
    save_interval: int = 20


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_30B_A3B_Stitch_Serve(StitchServeRecipe):
    """Flash pool of single-H200 SGLang replicas serving Qwen3-30B-A3B rollouts.

    ~30B BF16 is ~61 GB of weights, so one H200 per replica — but unlike the
    MLA models on this ladder, Qwen3 MoE is GQA, whose KV cache is large enough
    that the engine's autosized pool would eat the headroom an in-place delta
    apply needs. Hence both levers below.
    """

    sglang: SglangRecipe = field(
        default_factory=lambda: SglangRecipe(
            gpu="H200",
            tp=1,
            context_length=16384,
            # An in-place apply loads incoming shards beside the live weights,
            # so the engine has to leave room for them on a replica that is
            # already serving at full concurrency.
            mem_fraction_static=0.72,
            chunked_prefill_size=4096,
            extra_server_args={
                # GQA KV is ~98 KB/token here (48 layers × 4 groups × 128
                # channels, bf16), so the fraction alone isn't enough: capping
                # the pool at what full concurrency can address (~25 GB) leaves
                # the apply its headroom next to ~61 GB of weights.
                "--max-total-tokens": "262144",
                # Boot on the default loader, not fastsafetensors: SGLang reuses
                # the server's load format for online updates, and that copier
                # stages shard blocks through an 800 MB GPU buffer a loaded
                # replica has nowhere to put.
                "--weight-loader-drop-cache-after-load": "",
                # Routing replay: slime runs no local engine in publish-only
                # mode, so the served engine has to return routed experts.
                "--enable-return-routed-experts": "",
                "--reasoning-parser": "qwen3",
            },
        )
    )
    concurrency: int = 64
    # A rollout step issues 32 × 8 = 256 concurrent requests and a replica takes
    # `concurrency` = 64, so keep the whole step's worth warm: a 30B MoE takes
    # minutes to load, far longer than slime's HTTP retry budget for a request,
    # so letting Flash cold-scale into the first rollout step 503s the run out.
    min_containers: int = 4
    max_containers: int | None = 4


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_30B_A3B_Stitch_Recipe(StitchRecipe):
    """Qwen3-30B-A3B disaggregated GRPO: 1×8×H200 trainer + Modal Flash pool.

    The BF16 port of stitch's ``qwen3_30b_a3b_nvfp4_46`` cookbook config —
    single-node TP4/EP8 trainer with a CPU-offloaded optimizer, MoE routing
    replay, and rollouts served by single-H200 SGLang replicas that apply sparse
    deltas in place. NVFP4 (the point of the upstream config) needs the miles
    trainer, which this launcher does not wire yet.
    """

    train: SlimeStitchTrainRecipe = field(  # type: ignore[assignment]
        default_factory=Qwen3_30B_A3B_Stitch_Train
    )
    serve: StitchServeRecipe = field(default_factory=Qwen3_30B_A3B_Stitch_Serve)
