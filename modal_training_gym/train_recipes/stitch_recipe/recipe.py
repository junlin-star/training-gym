"""Recipe for disaggregated slime GRPO training on Modal via stitch.

``StitchRecipe`` mirrors :class:`SlimeRecipe` — same field style, same
model/dataset/wandb → CLI converters — but drives slime's *disaggregated*
mode: the actor cluster trains and publishes sparse weight deltas to a Modal
Volume "bulletin board", while a separate Modal Flash pool of SGLang servers
(each with a stitch weight-sync sidecar) serves rollouts and self-syncs to the
newest published version.

This is the training-gym packaging of the ``stitch`` ``slime_disagg`` cookbook
(https://github.com/modal-projects/stitch/tree/main/cookbook/slime_disagg): the
``stitch`` library supplies the bulletin protocol + sidecar + slime hooks, and
this recipe + :func:`build_stitch_app` play the role the cookbook's config +
``modal_train.py`` play there.

Unlike ``SlimeRecipe`` (colocated, ephemeral ``modal run ::train``), the Flash
rollout pool is a *deployed* app: build the app with :func:`build_stitch_app`,
``modal deploy`` it, then spawn a run via the ``launch_train`` entrypoint.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import field
from typing import Any

import modal
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType
from modal_training_gym.train_recipes.slime_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    JSON_CONFIG_FIELDS,
    SlimeRecipe,
)

__all__ = [
    "CHECKPOINTS_PATH",
    "DATA_PATH",
    "HF_CACHE_PATH",
    "SLIME_IMAGE_TAG",
    "SLIME_REPO_REF",
    "SLIME_REPO_URL",
    "YAML_CONFIG_FIELDS",
    "StitchRecipe",
]

# ── Pinned slime fork ──────────────────────────────────────────────────────────
# The fork carries the generic HTTP rollout endpoint + publish-only disk-delta
# hooks the disagg flow drives; the stock ``slimerl/slime`` image does not. Pin
# an exact commit (not a branch tip): the build's ``git fetch && checkout`` is a
# cached image layer, so a moving tip would silently leave a stale slime.
SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
SLIME_REPO_URL = "https://github.com/modal-projects/slime.git"
SLIME_REPO_REF = "ebfe153949b1a69c39e92f947ed5d475166dd724"

# Fields slime reads as YAML files at runtime. Recipes set them as inline dicts;
# the launcher materializes them to temp YAML files before building the command.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")

# Fields that are launcher/infra instructions, NOT slime CLI flags. Everything
# else on the dataclass is forwarded to slime via ``slime_fields()``.
_STITCH_SKIP = {
    "recipe_type",
    "name",
    "app_tags",
    "wandb",
    "environment",
    "async_mode",
    "slime_model_script",
    "image_overlay",
    "image_run_commands",
    "image_env",
    "custom_rm_function",
    # Modal infra
    "gpu_type",
    "cloud",
    "region",
    "memory",
    "rollout_min_containers",
    "proxy_regions",
    "rollout_sync_barrier",
    "rollout_sync_barrier_timeout_seconds",
    "rollout_sync_barrier_poll_seconds",
    # stitch / rollout-pool infra
    "delta_volume_name",
    "delta_bulletin_root",
    "sidecar_commit_mode",
    "sidecar_debug_requests",
    "sglang_server_concurrency",
    "sglang_server_args",
    "slime_image_tag",
    "slime_repo_url",
    "slime_repo_ref",
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class StitchRecipe(BaseTrainRecipe):
    """Recipe for disaggregated slime GRPO training on Modal via stitch.

    Don't see the configuration flag you need? Pass it through the
    ``extra_config`` escape hatch (materialized into slime's ``--custom-config-path``)
    or, for the SGLang rollout pool, ``sglang_server_args``.
    """

    # ── App identity / launcher instructions (not slime CLI flags) ──────────
    recipe_type: RecipeType = RecipeType.STITCH
    name: str = ""
    app_tags: dict = field(default_factory=dict)
    wandb: WandbConfig | None = None
    environment: dict = field(
        default_factory=lambda: {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )
    async_mode: bool = False
    slime_model_script: str = ""
    image_overlay: Callable[[modal.Image], modal.Image] | None = None
    image_run_commands: list[str] = field(default_factory=list)
    image_env: dict[str, str] = field(default_factory=dict)

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu_type: str = "H200"
    cloud: str | None = None
    region: str | None = None
    memory: int | tuple[int, int] | None = None
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    rollout_min_containers: int = 3
    proxy_regions: list[str] = field(default_factory=lambda: ["us-east"])
    # Post-publish sync barrier (launcher instruction, injected into the
    # trainer's custom_config, not a slime CLI flag). After publishing a delta,
    # the trainer blocks until the Flash pool reports the new version before the
    # next rollout generates — otherwise generation races servers mid-reload,
    # which drops in-flight requests and hangs the rollout. Bounded by the
    # timeout, then proceeds (staleness-gated requests are the backstop).
    rollout_sync_barrier: bool = True
    rollout_sync_barrier_timeout_seconds: int = 600
    rollout_sync_barrier_poll_seconds: float = 3.0

    # ── stitch / Flash rollout pool infrastructure ──────────────────────────
    # Modal Volume that backs the weight-delta bulletin board. Empty → derived
    # from the app name at build time.
    delta_volume_name: str = ""
    delta_bulletin_root: str = "/delta-bulletin"
    # How each rollout sidecar applies a published version. "in_place" pauses,
    # applies, and resumes (cross-version isolation via extra_key stamping);
    # "quiesce" drains in-flight requests before applying.
    sidecar_commit_mode: str = "in_place"
    sidecar_debug_requests: bool = False
    # SGLang rollout concurrency per container (drives Modal concurrency +
    # --max-running-requests / --cuda-graph-max-bs on the served engine).
    sglang_server_concurrency: int = 64
    # Extra SGLang server args for the rollout pool, merged over the structural
    # args the launcher sets (e.g. {"--reasoning-parser": "qwen3"}).
    sglang_server_args: dict[str, str] = field(default_factory=dict)
    slime_image_tag: str = SLIME_IMAGE_TAG
    slime_repo_url: str = SLIME_REPO_URL
    slime_repo_ref: str = SLIME_REPO_REF

    # ── Cluster / parallelism (slime CLI) ───────────────────────────────────
    # Disagg: rollouts run on the external Flash pool, so the actor cluster
    # allocates no rollout GPUs.
    colocate: bool = False
    rollout_num_gpus: int = 0
    rollout_num_gpus_per_engine: int = 1
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False

    # ── Rollout (slime CLI) ─────────────────────────────────────────────────
    num_rollout: int = 3
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_shuffle: bool = True
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    rollout_function_path: str = "slime.rollout.sglang_rollout.generate_rollout"

    # ── Disaggregated rollout routing (slime CLI) ───────────────────────────
    # Pins each rollout request to a served weight version; a lagging replica
    # returns a retryable 409 so requests flow across a weight update.
    custom_rollout_request_hook_path: str = (
        "stitch.trainers.slime.rollout_request_weight_version_hook"
    )
    rollout_request_weight_version_mode: str = "exact"
    rollout_request_weight_version_lag: int = 0
    rollout_request_retry_attempts: int = 240
    rollout_request_retry_sleep: float = 1.0
    # The trainer hits the Flash gateway directly, which routes session affinity
    # on Modal-Session-ID; emit that so GRPO siblings co-locate.
    rollout_session_affinity_header: str = "Modal-Session-ID"

    # ── Weight sync: publish sparse deltas to the bulletin board (slime CLI) ─
    update_weight_mode: str = "delta"
    update_weight_transport: str = "disk"
    update_weight_delta_encoding: str = "xor"
    update_weight_delta_checksum: str = "xxh3-128"
    # rank-0 publish hook: advance the pointer, commit the Volume, wake the pool.
    custom_delta_pre_push_path: str = (
        "modal_training_gym.frameworks.stitch.bulletin_hooks.commit_and_wake"
    )

    # ── Checkpointing (slime CLI) ───────────────────────────────────────────
    save: str = str(CHECKPOINTS_PATH)
    load: str = ""
    save_interval: int = 20
    megatron_to_hf_mode: str = "bridge"
    ref_load: str = ""
    use_fault_tolerance: bool = False

    # ── Reward model (slime CLI) ────────────────────────────────────────────
    rm_type: str | None = None
    custom_rm_function: Callable | None = None

    # ── Eval (slime CLI) ────────────────────────────────────────────────────
    eval_interval: int | None = None
    n_samples_per_eval_prompt: int = 4
    eval_max_response_len: int = 8192
    eval_top_p: float = 1.0

    # ── Training (slime CLI) ────────────────────────────────────────────────
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 9216
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True

    # ── Optimizer (slime CLI) ───────────────────────────────────────────────
    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98

    # ── RL algorithm (slime CLI) ────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0

    # ── SGLang / config overrides (slime CLI, YAML/JSON) ────────────────────
    extra_config: dict | None = None
    sglang_config: dict | None = None
    eval_config: dict | None = None
    apply_chat_template_kwargs: dict | str = ""

    # ── Converters (reused from SlimeRecipe) ────────────────────────────────

    @staticmethod
    def _resolve_data_paths(ds: DatasetConfig) -> tuple[str, dict[str, str] | None]:
        return SlimeRecipe._resolve_data_paths(ds)

    def slime_fields(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> dict[str, Any]:
        """Resolved slime CLI fields (name → value), excluding infra + the
        three run-time-injected fields the Trainer fills in per launch
        (``rollout_endpoint_url``, ``update_weight_disk_dir``,
        ``custom_config_path``)."""
        import dataclasses as _dc

        fields: dict[str, Any] = {
            f.name: getattr(self, f.name) for f in _dc.fields(self)
        }
        if dataset is not None:
            SlimeRecipe._validate_dataset(dataset)
            fields.update(SlimeRecipe._dataset_to_fields(dataset))
        if model is not None and not self.slime_model_script:
            model_fields = SlimeRecipe._model_to_fields(model)
            fields.update(model_fields)
            # bridge mode loads HF weights directly as the reference; default
            # ref_load to the base checkpoint when the user didn't set one.
            if self.megatron_to_hf_mode == "bridge" and not fields.get("ref_load"):
                fields["ref_load"] = model_fields["hf_checkpoint"]
        if self.wandb is not None:
            fields.update(SlimeRecipe._wandb_to_fields(self.wandb))

        out = {k: v for k, v in fields.items() if k not in _STITCH_SKIP}

        # extra_config is the per-recipe escape hatch and must always win over a
        # top-level field: slime's --<flag> overrides the YAML custom-config, so
        # drop any CLI flag a recipe also set in extra_config, then rename.
        extra_cfg = fields.get("extra_config")
        if isinstance(extra_cfg, dict):
            for key in extra_cfg:
                out.pop(key, None)
        if "extra_config" in out:
            out["custom_config_path"] = out.pop("extra_config")
        return out

    def cli_args(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> list[str]:
        """Flatten :meth:`slime_fields` to a slime CLI argv list. YAML config
        fields (:data:`YAML_CONFIG_FIELDS`) are skipped here — the launcher
        materializes them to files and appends the resolved flags."""
        return fields_to_argv(self.slime_fields(model=model, dataset=dataset))

    def to_payload(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> dict[str, Any]:
        """Plain-data payload shipped to the deployed Trainer at launch."""
        return {
            "fields": self.slime_fields(model=model, dataset=dataset),
            "environment": dict(self.environment),
            "async_mode": self.async_mode,
            "slime_model_script": self.slime_model_script,
        }


def fields_to_argv(fields: dict[str, Any]) -> list[str]:
    """slime CLI argv from a resolved field dict.

    Rules: ``field_name`` → ``--field-name``; ``True`` → bare flag;
    ``False``/``None``/``""`` → omitted; list → ``--flag v1 v2 …``; dict in
    :data:`YAML_CONFIG_FIELDS` → skipped (materialized to a file by the
    launcher); dict in :data:`JSON_CONFIG_FIELDS` → ``--flag '<json>'``.
    """
    out: list[str] = []
    for key, val in fields.items():
        if val is None or val is False or val == "":
            continue
        if key in YAML_CONFIG_FIELDS and isinstance(val, dict):
            continue
        flag = f"--{key.replace('_', '-')}"
        if val is True:
            out.append(flag)
        elif isinstance(val, dict) and key in JSON_CONFIG_FIELDS:
            out += [flag, json.dumps(val)]
        elif isinstance(val, (list, tuple)):
            out += [flag] + [str(v) for v in val]
        else:
            out += [flag, str(val)]
    return out
