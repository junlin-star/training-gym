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
    rollout_min_containers: int = 3

    # ── Model (arch comes from slime's model script, not ModelArchitecture) ─
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
            # An in-place delta apply loads the incoming shard alongside the live
            # weights; 16B-A3B leaves more room than 4B dense, hence 0.85.
            "--mem-fraction-static": "0.85",
            # Routing replay: slime runs no local engine in publish-only mode, so
            # the served engine has to be told to return routed experts.
            "--enable-return-routed-experts": "",
        }
    )

    # ── Rollout / algorithm ─────────────────────────────────────────────────
    async_mode: bool = True
    update_weights_interval: int = 1
    num_rollout: int = 5
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    # Async mode publishes while rollouts are in flight, so a request pinned to
    # the exact newest version would stall; allow one version of lag.
    rollout_request_weight_version_mode: str = "min"
    rollout_request_weight_version_lag: int = 1
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
