"""Qwen3-4B disaggregated GRPO with stitch-managed Flash rollout pool."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4b_Stitch_Recipe(StitchRecipe):
    """Qwen3-4B GRPO with Modal Flash sparse-delta rollout servers.

    1×8×H100 trainer; a warm Flash pool of 1-GPU SGLang replicas handles
    rollouts. Weight deltas are published via a shared Modal Volume.
    """

    # ── Cluster ────────────────────────────────────────────────────────────
    gpu_type: str = "H100"
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus_per_engine: int = 1
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False

    # ── Rollout pool ───────────────────────────────────────────────────────
    rollout_min_containers: int = 4
    sglang_server_extra_args: dict[str, str] = None  # type: ignore[assignment]
    sidecar_commit_mode: str = "in_place"
    sidecar_debug_requests: bool = True

    # ── Rollout ────────────────────────────────────────────────────────────
    num_rollout: int = 3
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    sglang_server_concurrency: int = 64
    use_fault_tolerance: bool = False

    # ── Training ───────────────────────────────────────────────────────────
    megatron_to_hf_mode: str = "bridge"
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 9216
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True

    # ── Optimizer ──────────────────────────────────────────────────────────
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98

    # ── Algorithm ──────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0

    # ── Eval ───────────────────────────────────────────────────────────────
    eval_interval: int | None = None
    n_samples_per_eval_prompt: int = 4
    eval_max_response_len: int = 8192
    eval_top_p: float = 1.0

    # ── Checkpointing ──────────────────────────────────────────────────────
    save_interval: int = 10

    def __post_init__(self):
        if self.sglang_server_extra_args is None:
            self.sglang_server_extra_args = {
                "--reasoning-parser": "qwen3",
                "--context-length": "16384",
                "--mem-fraction-static": "0.84",
                "--chunked-prefill-size": "4096",
                "--max-prefill-tokens": "4096",
            }
