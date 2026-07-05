from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4b_Stitch_Recipe(StitchRecipe):
    """Qwen3-4B disaggregated GRPO: 1×8×H200 trainer + Flash rollout pool."""

    # ── Trainer cluster ────────────────────────────────────────────────────
    gpu_type: str = "H200"
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus_per_engine: int = 1

    num_rollout: int = 3
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0

    save_interval: int = 10

    # ── Disaggregated rollout pool ─────────────────────────────────────────
    rollout_gpu_type: str = "H200"
    rollout_min_containers: int = 4
    sglang_server_concurrency: int = 64
    sglang_server_args: dict[str, str] = None  # type: ignore[assignment]
    delta_volume_name: str = "stitch-delta-bulletin-qwen3-4b"

    # ── RL algorithm ────────────────────────────────────────────────────────
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    lr: float = 1e-6
    max_tokens_per_gpu: int = 9216
    eval_interval: int | None = None
    eval_max_response_len: int = 8192
    megatron_to_hf_mode: str = "bridge"

    # ── Weight sync ─────────────────────────────────────────────────────────
    use_fault_tolerance: bool = False

    def __post_init__(self) -> None:
        if self.sglang_server_args is None:
            object.__setattr__(
                self,
                "sglang_server_args",
                {
                    "--reasoning-parser": "qwen3",
                    "--context-length": "16384",
                    "--mem-fraction-static": "0.84",
                    "--chunked-prefill-size": "4096",
                    "--max-prefill-tokens": "4096",
                },
            )
