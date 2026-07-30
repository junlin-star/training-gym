from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4b_Stitch_Recipe(StitchRecipe):
    """Qwen3-4B disaggregated GRPO: 1×8×H200 trainer + Modal Flash rollout pool.

    Mirrors the stitch ``qwen3_4b_delta_flash`` cookbook config: publish-only
    disk-delta weight sync over a Modal Volume bulletin board, rollouts served
    by a warm Flash pool of single-GPU SGLang servers.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu_type: str = "H200"
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus_per_engine: int = 1
    rollout_min_containers: int = 3

    # ── Rollout pool (SGLang) ───────────────────────────────────────────────
    sglang_server_concurrency: int = 64
    sglang_server_args: dict[str, str] = field(
        default_factory=lambda: {
            # The no-GDS fastsafetensors path — hosts have no nvidia-fs, and it is
            # also the load format the sidecar reuses for each delta reload.
            "--load-format": "fastsafetensors",
            "--model-loader-extra-config": '{"enable_gds":false}',
            "--weight-loader-drop-cache-after-load": "",
            "--reasoning-parser": "qwen3",
            "--context-length": "16384",
            # An in-place delta apply loads the new shard alongside the live
            # weights, so the static pool has to leave that much GPU headroom on
            # a replica that is already serving at full concurrency.
            "--mem-fraction-static": "0.80",
            "--chunked-prefill-size": "4096",
            "--max-prefill-tokens": "4096",
        }
    )

    # ── Rollout / algorithm ─────────────────────────────────────────────────
    num_rollout: int = 3
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128

    # ── Training ────────────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    max_tokens_per_gpu: int = 9216
    lr: float = 1e-6

    rm_type: str | None = "math"
    save_interval: int = 20
