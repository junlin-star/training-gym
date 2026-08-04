import json
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modal_training_gym.train_recipes.gpu_allocation import (
    GpuAllocation,
    resolve_gpu_allocation,
)

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfig
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths (shared by every framework) ───────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# Recipe fields whose dict values are emitted as JSON CLI arguments.
JSON_CONFIG_FIELDS = ("train_env_vars", "apply_chat_template_kwargs", "multimodal_keys")


def carry_explicit_fields(source: Any, rebuilt: Any) -> Any:
    """Restore ``source``'s record of caller-set fields onto a rebuilt recipe.

    Rebuilding as ``type(r)(**all_fields)`` passes every field as a kwarg, so the
    validator would record them all as caller-set and make ``_for_dataset`` a no-op.
    """
    explicit = getattr(source, "explicit_fields", None)
    if explicit is not None:
        object.__setattr__(rebuilt, "_explicit_fields", frozenset(explicit))
    return rebuilt


class RecipeType(Enum):
    SLIME = "slime"
    MILES = "miles"


class BaseTrainRecipe(ABC):
    recipe_type: RecipeType

    # ── Container → framework flag converters ────────────────────────────────

    @staticmethod
    def _resolve_data_paths(
        ds: "DatasetConfig",
    ) -> tuple[str, dict[str, str] | None]:
        """Derive on-volume file paths from a dataset's properties."""
        hf_repo = getattr(ds, "hf_repo", "")
        name = hf_repo.replace("/", "_") if hf_repo else type(ds).__name__
        fmt = getattr(ds, "output_format", "parquet")
        ext = "jsonl" if fmt == "jsonl" else "parquet"
        split = getattr(ds, "hf_split", "train")
        prompt_data = f"{DATA_PATH}/{name}/{split}.{ext}"
        if getattr(ds, "writes_eval_paths", True):
            return prompt_data, {"eval": f"{DATA_PATH}/{name}/eval.{ext}"}
        return prompt_data, None

    @classmethod
    def _dataset_to_fields(cls, ds: "DatasetConfig") -> dict[str, Any]:
        prompt_data, eval_paths = cls._resolve_data_paths(ds)
        eval_prompt_data: list[str] | None = None
        if eval_paths:
            eval_prompt_data = [
                v for name, path in eval_paths.items() for v in (name, path)
            ]
        return {
            "prompt_data": prompt_data,
            "eval_prompt_data": eval_prompt_data,
            "input_key": ds.input_key,
            "label_key": ds.label_key,
            "apply_chat_template": ds.apply_chat_template,
        }

    @staticmethod
    def _wandb_to_fields(w: "WandbConfig") -> dict[str, Any]:
        return {
            "use_wandb": True,
            "wandb_project": w.project,
            "wandb_group": w.group,
            "wandb_key": w.key,
            "disable_wandb_random_suffix": w.disable_random_suffix,
        }

    # ── CLI serialization ─────────────────────────────────────────────────────

    def _fields(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfig | None" = None,
    ) -> dict[str, Any]:
        """Recipe fields to emit as CLI flags, merged with dataset/model/wandb.

        Not abstract: lightweight subclasses (e.g. test doubles) may skip it,
        in which case ``cli_args`` is unavailable.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _fields() to use cli_args()"
        )

    def cli_args(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfig | None" = None,
    ) -> list[str]:
        out: list[str] = []
        for key, val in self._fields(dataset=dataset, model=model).items():
            if val is None or val is False or val == "":
                continue
            flag = f"--{key.replace('_', '-')}"
            if val is True:
                out.append(flag)
            elif isinstance(val, dict) and key in JSON_CONFIG_FIELDS:
                out += [flag, json.dumps(val)]
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    # ── Cluster topology ──────────────────────────────────────────────────────

    @property
    def total_nodes(self) -> int:
        return self.gpu_allocation.total_nodes

    @property
    def gpu_allocation(self) -> GpuAllocation:
        return resolve_gpu_allocation(self, warn=False)
