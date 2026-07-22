from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26b_A4b_Recipe(SlimeRecipe):
    """Gemma-4-26B-A4B (MoE) on 1x8xH100, colocated GRPO.

    Uses slime's upstream model script verbatim (custom ``--spec`` +
    ``--custom-model-provider-path`` gemma4 provider), so no MoE/architecture
    flags are set here — only Modal infra, parallelism, rollout, and optimizer
    settings. Parallelism: TP2/PP1/CP1/EP4/ETP1 across 8 GPUs
    (128 experts % EP4 == 0; world 8 % TP2 == 0 and % (ETP1*EP4*PP1) == 0).

    Smoke default is a single rollout (``num_rollout=1``); bump for real runs.
    """

    # ── Modal infra ─────────────────────────────────────────────────────────
    gpu_type: str = "H100"
    slime_model_script: str = "scripts/models/gemma4-26B-A4B.sh"
    hf_checkpoint: str = "google/gemma-4-26b-a4b"
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    # ── Cluster ─────────────────────────────────────────────────────────────
    colocate: bool = True
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    # ── Parallelism ─────────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 4
    expert_tensor_parallel_size: int = 1

    # ── Rollout ─────────────────────────────────────────────────────────────
    num_rollout: int = 1
    rollout_batch_size: int = 16
    rollout_num_gpus_per_engine: int = 2
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    global_batch_size: int = 128
    sglang_mem_fraction_static: float = 0.7

    # ── Training ────────────────────────────────────────────────────────────
    n_samples_per_prompt: int = 8
    lr: float = 1e-6
    max_tokens_per_gpu: int = 8192
    balance_data: bool = True
    use_kl_loss: bool = True
    attention_backend: str = "flash"

    # ── Optimizer (MoE: offload to fit on a single node) ────────────────────
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Checkpointing / eval ────────────────────────────────────────────────
    no_save_optim: bool = True
    save_interval: int = 10
    eval_interval: int | None = None
