"""MegaGem Qwen3-4B SFT Stage-C topology for Slime.

Stage C is the 40-GPU migration target: one H200 actor node trains full
Megatron weights, while four H200 rollout nodes host 32 one-GPU SGLang engines.
The rollout hook in :mod:`megagem_stage_c_rollout` generates current-policy
MegaGem self-play K-groups and feeds Slime selected turn rows with MegaGem's
precomputed reward and advantage contract.

This is intentionally explicit about the migration boundary: it ports the
current-policy K-group GRPO mechanics and row-level credit assignment. The old
Modal Phase-3 LoRA snapshot/anchor league is not silently emulated here because
Slime is synchronizing full weights into SGLang, not swapping LoRA adapters.
"""

from __future__ import annotations

import json
import os
from dataclasses import field
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models.megagem_qwen3_4b_sft import (
    MEGAGEM_QWEN3_4B_SFT_MODEL,
)
from modal_training_gym.frameworks.slime.megagem_stage_c_rollout import (
    MEGAGEM_STAGE_C_ADVANTAGE_PATH,
    MEGAGEM_STAGE_C_REWARD_PATH,
    MEGAGEM_STAGE_C_ROLLOUT_CONTRACT,
    megagem_stage_c_rollout,
)
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


MEGAGEM_QWEN3_4B_SFT_MODEL_TAG = MEGAGEM_QWEN3_4B_SFT_MODEL.replace("/", "-")
MEGAGEM_STAGE_C_REMOTE_ROOT = "/root/MegagemBench"


class MegaGemStageCDataset(DatasetConfig):
    """Synthetic prompt groups for on-policy MegaGem rollouts.

    Slime samples ``rollout_batch_size`` prompt rows and repeats each prompt
    ``n_samples_per_prompt`` times. Each prompt row labels one MegaGem seed
    group; the custom rollout hook assigns the repeated siblings to selected
    turn-row slots from that generated K-group.
    """

    dataset_id = "megagem-stage-c"
    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = False
    always_prepare = True

    def __init__(
        self,
        *,
        num_prompts: int = 128,
        seed_start: int = 9000,
        seed_stride: int | None = None,
        k: int = 16,
        rows_per_group: int = 16,
        value_chart: str = "A",
        trainable_seat: int = 0,
        rotate_seats: bool = True,
        num_players: int = 3,
        opponent_actor_id: str = "current_self",
        dataset_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.num_prompts = int(num_prompts)
        self.seed_start = int(seed_start)
        self.seed_stride = int(seed_stride or num_prompts)
        self.k = int(k)
        self.rows_per_group = int(rows_per_group)
        self.value_chart = str(value_chart)
        self.trainable_seat = int(trainable_seat)
        self.rotate_seats = bool(rotate_seats)
        self.num_players = int(num_players)
        self.opponent_actor_id = str(opponent_actor_id)
        self.dataset_id = dataset_id or (
            f"megagem-stage-c-{self.seed_start}-{self.num_prompts}"
            f"-k{self.k}-r{self.rows_per_group}"
        )
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return self.dataset_id

    def _rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for i in range(self.num_prompts):
            seed = self.seed_start + i
            trainable_seat = (
                (self.trainable_seat + i) % self.num_players
                if self.rotate_seats
                else self.trainable_seat
            )
            label = {
                "seed": seed,
                "seed_stride": self.seed_stride,
                "value_chart": self.value_chart,
                "trainable_seat": trainable_seat,
                "num_players": self.num_players,
                "k": self.k,
                "rows_per_group": self.rows_per_group,
                "opponent_actor_id": self.opponent_actor_id,
            }
            rows.append(
                {
                    "prompt": (
                        "[megagem-stage-c] "
                        f"seed={seed} chart={self.value_chart} "
                        f"seat={trainable_seat} k={self.k} "
                        f"stride={self.seed_stride} rows={self.rows_per_group}"
                    ),
                    "label": json.dumps(label, sort_keys=True),
                }
            )
        return rows

    def _write_jsonl(self, rows: list[dict[str, str]], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        rows = self._rows()
        self._write_jsonl(rows, path)
        if eval_paths:
            eval_rows = rows[: min(8, len(rows))] or rows
            for eval_path in eval_paths.values():
                self._write_jsonl(eval_rows, eval_path)

    def load(self, split: Literal["all", "train", "eval"] = "all") -> list[dict[str, str]]:
        rows = self._rows()
        if split == "eval":
            return rows[: min(8, len(rows))]
        return rows


def megagem_stage_c_image_overlay(image: Any) -> Any:
    """Add the local MegaGemBench checkout and its lightweight deps."""

    source = Path(os.environ.get("MEGAGEM_BENCH_SOURCE", "/home/ec2-user/MegagemBench"))
    if not source.exists():
        raise FileNotFoundError(
            f"MegaGemBench checkout not found at {source}. Set "
            "MEGAGEM_BENCH_SOURCE before building Stage C if it lives elsewhere."
        )
    image = image.add_local_dir(
        str(source),
        remote_path=MEGAGEM_STAGE_C_REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".venv",
            ".venv/**",
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            "results",
            "results/**",
            "wandb",
            "wandb/**",
            "docs",
            "docs/**",
            "paper-research",
            "paper-research/**",
        ],
    )
    return image.uv_pip_install(
        "datasets>=3.0.0",
        "openai>=1.0.0",
        "rich>=13.0.0",
        "verifiers>=0.1.5.post0",
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class MegaGem_Qwen3_4B_StageC_Recipe(SlimeRecipe):
    """40-GPU Stage C: 1 actor node plus 4 rollout nodes on H200."""

    # ── Cluster geometry ─────────────────────────────────────────────────
    gpu_type: str = "H200"
    colocate: bool = False
    async_mode: bool = True
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int = 32
    rollout_num_gpus_per_engine: int = 1
    memory: int | tuple[int, int] | None = (128, 2_097_152)

    # ── Dense 4B actor parallelism ───────────────────────────────────────
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False

    # ── Large-scale rollout shape ────────────────────────────────────────
    num_rollout: int = 50
    rollout_batch_size: int = 128
    n_samples_per_prompt: int = 16
    global_batch_size: int = 2048
    rollout_max_response_len: int = 2048
    eval_max_response_len: int = 2048
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    sglang_mem_fraction_static: float = 0.90
    sglang_max_running_requests: int | None = 1024
    sglang_cuda_graph_bs: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 1025, 16))
    )

    # ── GRPO optimizer defaults ──────────────────────────────────────────
    advantage_estimator: str = "grpo"
    rm_type: str | None = None
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    use_kl_loss: bool = True
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.01
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    calculate_per_token_loss: bool = True
    lr: float = 2e-5
    lr_decay_style: str = "cosine"
    min_lr: float = 2e-6
    weight_decay: float = 0.0
    max_tokens_per_gpu: int = 16384
    use_distributed_optimizer: bool = True
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1

    # ── Checkpointing / eval ─────────────────────────────────────────────
    save_interval: int = 25
    eval_interval: int | None = None
    no_save_optim: bool = False

    # ── MegaGem rollout bridge ───────────────────────────────────────────
    custom_generate_function: Any = megagem_stage_c_rollout
    image_overlay: Any = megagem_stage_c_image_overlay
    extra_config: dict | None = field(
        default_factory=lambda: {
            "custom_rm_path": MEGAGEM_STAGE_C_REWARD_PATH,
            "custom_advantage_function_path": MEGAGEM_STAGE_C_ADVANTAGE_PATH,
            "megagem_max_parallel_games": 256,
            "megagem_extra_games_per_group": 4,
            "megagem_min_success_games": 12,
            "megagem_game_timeout_s": 600,
            "megagem_fail_open_groups": True,
            "log_multi_turn": False,
        }
    )
    app_tags: dict = field(
        default_factory=lambda: {
            "megagem": "stage-c",
            "base_model": MEGAGEM_QWEN3_4B_SFT_MODEL_TAG,
            "rollout_contract": MEGAGEM_STAGE_C_ROLLOUT_CONTRACT,
        }
    )
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": (
                "/root/MegagemBench:/root/MegagemBench/scripts/phase2:"
                "/root/Megatron-LM/"
            ),
            "MEGAGEM_BENCH_ROOT": MEGAGEM_STAGE_C_REMOTE_ROOT,
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TRAINING_GYM_RAY_WORKER_WAIT_RETRIES": "240",
            # Mirrors the current MegaGemBench Phase-3 reward cut.
            "PHASE3_REWARD_WIN_BONUS": "",
            "PHASE3_SHAPING_LAMBDA": "",
            "PHASE3_ILLEGAL_PENALTY": "",
            "PHASE3_CORRECTION_SCALE": "",
            "PHASE3_TERMINAL_CORRECTION": "1",
            "PHASE3_TERMINAL_SHAPE": "tanh",
            "PHASE3_REWARD_SCALE": "19.6",
            "PHASE3_REVEAL_SHAPING_WEIGHT": "0.0",
        }
    )


def megagem_stage_c_summary(recipe: SlimeRecipe) -> dict[str, Any]:
    """Stable machine-readable geometry summary for tests and dry runs."""

    actor_gpus = recipe.actor_num_nodes * recipe.actor_num_gpus_per_node
    rollout_gpus = int(recipe.rollout_num_gpus or 0)
    mp = (
        recipe.tensor_model_parallel_size
        * getattr(recipe, "pipeline_model_parallel_size", 1)
        * getattr(recipe, "context_parallel_size", 1)
    )
    return {
        "model_name": MEGAGEM_QWEN3_4B_SFT_MODEL,
        "recipe_class": type(recipe).__name__,
        "gpu_type": recipe.gpu_type,
        "total_nodes": recipe.total_nodes,
        "total_gpus": recipe.gpu_allocation.total_gpus,
        "actor_nodes": recipe.actor_num_nodes,
        "actor_gpus": actor_gpus,
        "actor_data_parallel": actor_gpus // mp,
        "rollout_gpus": rollout_gpus,
        "rollout_engines": recipe.gpu_allocation.rollout_engines,
        "rollout_num_gpus_per_engine": recipe.rollout_num_gpus_per_engine,
        "tp": recipe.tensor_model_parallel_size,
        "pp": getattr(recipe, "pipeline_model_parallel_size", 1),
        "cp": getattr(recipe, "context_parallel_size", 1),
        "rollout_batch_size": recipe.rollout_batch_size,
        "n_samples_per_prompt": recipe.n_samples_per_prompt,
        "rollout_top_p": recipe.rollout_top_p,
        "games_per_rollout": recipe.rollout_batch_size * recipe.n_samples_per_prompt,
        "global_batch_size": recipe.global_batch_size,
        "updates_per_rollout": (
            recipe.rollout_batch_size
            * recipe.n_samples_per_prompt
            // recipe.global_batch_size
        ),
        "kl_loss_coef": recipe.kl_loss_coef,
        "entropy_coef": recipe.entropy_coef,
        "lr": recipe.lr,
        "min_lr": recipe.min_lr,
        "rollout_contract": MEGAGEM_STAGE_C_ROLLOUT_CONTRACT,
        "custom_rm_path": (recipe.extra_config or {}).get("custom_rm_path"),
        "custom_advantage_function_path": (recipe.extra_config or {}).get(
            "custom_advantage_function_path"
        ),
        "megagem_max_parallel_games": (recipe.extra_config or {}).get(
            "megagem_max_parallel_games"
        ),
        "megagem_extra_games_per_group": (recipe.extra_config or {}).get(
            "megagem_extra_games_per_group"
        ),
        "megagem_min_success_games": (recipe.extra_config or {}).get(
            "megagem_min_success_games"
        ),
        "megagem_game_timeout_s": (recipe.extra_config or {}).get(
            "megagem_game_timeout_s"
        ),
        "megagem_fail_open_groups": (recipe.extra_config or {}).get(
            "megagem_fail_open_groups"
        ),
    }
