from __future__ import annotations

from typing import Any, Literal
import json
import random
import shutil
import tomllib
import uuid
from pathlib import Path

from modal_training_gym.common.dataset import DatasetConfig, DatasetType



class HarborDataset(DatasetConfig):
    """Dataset backed by a Harbor task directory structure.

    Each task folder contains an instruction file and optional label metadata.
    Tasks are discovered by globbing the task_root directory.
    """

    _type: DatasetType = DatasetType.HARBOR
    dataset_name: str = ""
    path: str | None = None
    task_root: str = ""
    task_glob: str = "*"
    task_names: list[str] | None = None
    instruction_path: str = "instruction.md"
    label_metadata_path: str | None = None
    test_data_dir: str | None = None
    output_format: str = "parquet"
    prompt_template: str = "{instruction}"
    system_prompt: str = ""
    train_size: int | None = None
    eval_size: int | None = None
    train_repeats: int = 1
    eval_repeats: int = 1
    shuffle_tasks: bool = False
    shuffle_seed: int = 0

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not self.input_key:
            self.input_key = "messages"
        if not self.label_key:
            self.label_key = "label"
        if "dataset_id" not in kwargs:
            if self.dataset_name:
                slug = self.dataset_name.replace("/", "-")
            elif self.path:
                slug = self.path.replace("/", "_")
            elif self.task_root:
                slug = self.task_root.replace("/", "_")
            else:
                slug = "harbor"
            self.dataset_id = f"{slug}-{uuid.uuid4()}"
        self._validate()

    @property
    def name(self) -> str:
        return self.dataset_name

    def _harbor_dataset_ref(self) -> str:
        if "@" in self.dataset_name:
            return self.dataset_name
        return f"{self.dataset_name}@latest"

    def _harbor_cache_dir(self) -> Path:
        slug = self._harbor_dataset_ref().replace("/", "--").replace("@", "--")
        return Path.home() / ".cache" / "harbor" / "datasets" / slug

    def _download_harbor_dataset(self, cache_dir: Path) -> None:
        import subprocess

        ref = self._harbor_dataset_ref()
        harbor_bin = shutil.which("harbor")
        if harbor_bin is not None:
            cmd = [
                harbor_bin,
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(cache_dir),
            ]
        else:
            uvx_bin = shutil.which("uvx")
            if uvx_bin is None:
                raise FileNotFoundError(
                    "Harbor CLI not found. Install `harbor` or `uvx` to download "
                    f"{self.dataset_name!r}."
                )
            cmd = [
                uvx_bin,
                "harbor",
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(cache_dir),
            ]
        subprocess.run(cmd, check=True)

    def _write_split(self, rows: list[dict[str, Any]], path: str) -> None:
        import os

        from datasets import Dataset

        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds = Dataset.from_list(rows)
        if self.output_format == "jsonl":
            ds.to_json(path, orient="records", lines=True)
            return
        ds.to_parquet(path)

    def _pull_harbor_dataset(self) -> Path:
        cache_dir = self._harbor_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not any(cache_dir.rglob(self.instruction_path)):
            self._download_harbor_dataset(cache_dir)
        task_root = self._discover_task_root(cache_dir)
        if not any(task_root.rglob(self.instruction_path)):
            raise FileNotFoundError(f"No Harbor tasks found under {cache_dir}")
        return task_root

    def _discover_task_root(self, search_root: Path) -> Path:
        task_dirs = sorted(
            {
                instruction_file.parent
                for instruction_file in search_root.rglob(self.instruction_path)
                if instruction_file.is_file()
            }
        )
        if not task_dirs:
            return search_root
        if len(task_dirs) == 1:
            return task_dirs[0].parent
        import os

        return Path(os.path.commonpath([str(path) for path in task_dirs]))

    def _resolve_task_root(self) -> Path:
        if self.path:
            task_root = Path(self.path).resolve()
        elif self.dataset_name:
            task_root = self._pull_harbor_dataset()
        elif self.task_root:
            task_root = Path(self.task_root).resolve()
        else:
            raise ValueError(
                f"{type(self).__name__} requires dataset_name, path, or task_root"
            )
        if not task_root.exists():
            raise FileNotFoundError(f"task root does not exist: {task_root}")
        if not task_root.is_dir():
            raise ValueError(f"task root is not a directory: {task_root}")
        return task_root

    def _candidate_task_dirs(self, task_root: Path) -> list[Path]:
        if self.task_names is not None:
            return [
                (task_root / name).resolve()
                for name in self.task_names
                if (task_root / name).is_dir()
            ]
        return sorted(
            path.resolve() for path in task_root.glob(self.task_glob) if path.is_dir()
        )

    def _iter_task_dirs(self) -> list[Path]:
        task_root = self._resolve_task_root()
        task_dirs = self._candidate_task_dirs(task_root)
        if not task_dirs:
            discovered_root = self._discover_task_root(task_root)
            if discovered_root != task_root:
                task_root = discovered_root
                task_dirs = self._candidate_task_dirs(task_root)
        if self.shuffle_tasks:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(task_dirs)
        if not task_dirs:
            raise ValueError(f"No Harbor tasks found under {task_root}")
        if self.train_size is not None:
            max_tasks = self.train_size + (self.eval_size or 0)
            task_dirs = task_dirs[:max_tasks]
        return task_dirs

    def _read_label_metadata(self, task_dir: Path) -> dict[str, Any]:
        if not self.label_metadata_path:
            return {}
        metadata_path = task_dir / self.label_metadata_path
        if not metadata_path.exists():
            return {}
        if metadata_path.suffix == ".json":
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        elif metadata_path.suffix == ".toml":
            data = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            raise ValueError(
                f"Unsupported label metadata file type for {metadata_path}; expected .json or .toml"
            )
        if not isinstance(data, dict):
            raise ValueError(
                f"Label metadata must decode to an object: {metadata_path}"
            )
        return data

    def _read_test_data(self, task_dir: Path) -> list[dict[str, str]]:
        assert self.test_data_dir is not None
        tests_dir = task_dir / self.test_data_dir
        test_cases: list[dict[str, str]] = []
        if not tests_dir.is_dir():
            return test_cases
        for in_file in sorted(tests_dir.glob("*.in")):
            out_file = in_file.with_suffix(".out")
            if out_file.exists():
                test_cases.append(
                    {
                        "input": in_file.read_text(encoding="utf-8"),
                        "expected_output": out_file.read_text(encoding="utf-8"),
                    }
                )
        return test_cases

    def _build_label(self, task_root: Path, task_dir: Path) -> dict[str, Any]:
        rel = task_dir.relative_to(task_root)
        rel_with_root = (Path(task_root.name) / rel).as_posix()
        label: dict[str, Any] = {
            "harbor_task_name": task_dir.name,
            "harbor_task_path": task_dir.as_posix(),
            "harbor_task_rel": rel_with_root,
        }
        label.update(self._read_label_metadata(task_dir))
        if self.test_data_dir:
            label["test_cases"] = self._read_test_data(task_dir)
        return label

    def _format_prompt(
        self, *, instruction: str, task_dir: Path, label: dict[str, Any]
    ) -> str:
        context = {
            "instruction": instruction,
            "task_name": task_dir.name,
            "task_path": task_dir.as_posix(),
            **label,
        }
        return self.prompt_template.format(**context).strip()

    def _build_row(self, task_root: Path, task_dir: Path) -> dict[str, Any]:
        instruction_file = task_dir / self.instruction_path
        if not instruction_file.exists():
            raise FileNotFoundError(
                f"instruction file does not exist for Harbor task {task_dir.name}: {instruction_file}"
            )
        instruction = instruction_file.read_text(encoding="utf-8").strip()
        label = self._build_label(task_root, task_dir)
        user_prompt = self._format_prompt(
            instruction=instruction, task_dir=task_dir, label=label
        )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return {
            "messages": messages,
            "label": json.dumps(label, separators=(",", ":")),
        }

    @staticmethod
    def _repeat_rows(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
        repeats = max(1, repeats)
        return [row for row in rows for _ in range(repeats)]

    def load(self, split: Literal["all", "train", "eval"] = "all") -> Any:
        task_root = self._resolve_task_root()
        out = []
        for task_dir in self._iter_task_dirs():
            instruction_file = task_dir / self.instruction_path
            if not instruction_file.exists():
                raise FileNotFoundError(
                    f"instruction file does not exist for Harbor task {task_dir.name}: {instruction_file}"
                )
            label = self._build_label(task_root, task_dir)
            out.append(
                {
                    "task_name": task_dir.name,
                    "task_path": task_dir.as_posix(),
                    "instruction": instruction_file.read_text(encoding="utf-8").strip(),
                    "label": label,
                }
            )
        if self.train_size is not None:
            if split == "train":
                return out[: self.train_size]
            if split == "eval":
                return out[self.train_size : self.train_size + (self.eval_size or 0)]
        return out

    def to_pandas(self, *, formatted: bool = False):
        import pandas as pd

        if not formatted:
            return pd.DataFrame(self.load())

        task_root = self._resolve_task_root()
        rows = [
            self._build_row(task_root, task_dir) for task_dir in self._iter_task_dirs()
        ]
        return pd.DataFrame(rows)

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        task_root = self._resolve_task_root()
        base_rows = [
            self._build_row(task_root, task_dir) for task_dir in self._iter_task_dirs()
        ]

        if self.train_size is None:
            train_base = base_rows
            eval_base = base_rows
        else:
            train_size = max(1, min(int(self.train_size), len(base_rows)))
            train_base = base_rows[:train_size]
            eval_base = (
                base_rows[train_size : train_size + (self.eval_size or 0)] or base_rows
            )

        train_rows = self._repeat_rows(train_base, int(self.train_repeats))
        eval_rows = self._repeat_rows(eval_base, int(self.eval_repeats))

        self._write_split(train_rows, path)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_split(eval_rows, eval_path)

