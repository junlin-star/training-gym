"""Dataset config + ``prepare()`` hook, shared across training frameworks.

Pure data — each framework config writes its own converter from a
``DatasetConfig`` instance to its specific CLI flags (e.g. SlimeRecipe emits
``--prompt-data``, ``--input-key``, …).

Subclass and override ``prepare()`` to materialize the data into a shared
volume.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
import json
import uuid
from pathlib import Path

DatasetRow = dict[str, Any]


class DatasetType(Enum):
    DEFAULT = "default"
    HUGGING_FACE = "hugging_face"
    HARBOR = "harbor"


class DatasetConfig:
    """Dataset configuration shared across training frameworks.

    Describes *what* the data is. Where it gets written on disk is decided
    by the recipe/launcher layer, not by the dataset itself.
    """

    _type: DatasetType = DatasetType.DEFAULT
    dataset_id: str = ""
    input_key: str = ""
    label_key: str = ""
    apply_chat_template: bool = True
    always_prepare: bool = False

    def __init__(self, **kwargs: Any) -> None:
        if not self.dataset_id:
            self.dataset_id = str(uuid.uuid4())
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()

    def _validate(self) -> None:
        """Required-field check; subclasses call this at the end of their own ``__init__``."""
        if not self.label_key:
            raise ValueError(
                f"{type(self).__name__} requires `label_key` to be set. "
                "It names the column on the materialized dataset that holds "
                "per-sample ground-truth / reward-function input. "
                'Declare it as a class attribute (`label_key = "label"`) on '
                "your subclass, or pass `label_key=...` as a kwarg. Frameworks "
                "like slime index `data[label_key]` at load time, so an unset "
                "value reliably crashes deep in a remote Ray actor."
            )

    @property
    def name(self) -> str:
        return self.dataset_id

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        """Materialize training data to ``path`` (and eval splits to ``eval_paths``)."""
        raise NotImplementedError(f"{type(self).__name__} has no prepare()")

    def load(self, split: Literal["all", "train", "eval"] = "all") -> Any:
        """Load raw examples, optionally filtered by split."""
        raise NotImplementedError(f"{type(self).__name__} has no load()")

    def _expected_columns(self) -> set[str]:
        cols: set[str] = set()
        if self.input_key:
            cols.add(self.input_key)
        if self.label_key:
            cols.add(self.label_key)
        return cols

    def validate_prepared(self, path: str) -> None:
        """Sniff what ``prepare()`` wrote and confirm the columns the framework will index.

        Catches the common ``KeyError: 'label'`` (and friends) that otherwise
        only fire deep inside a Ray actor on a remote container, after image
        build and cluster bringup.
        """
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{type(self).__name__}.prepare() did not produce {path!r}. "
                "Ensure your prepare(path, ...) override writes to the `path` arg."
            )

        expected = self._expected_columns()
        if not expected:
            return

        try:
            if path.endswith(".parquet"):
                import pyarrow.parquet as pq

                cols = set(pq.read_schema(path).names)
            elif path.endswith((".jsonl", ".json")):
                with open(path) as f:
                    first = f.readline().strip()
                if not first:
                    raise ValueError(f"{path!r} is empty")
                cols = set(json.loads(first).keys())
            else:
                return
        except Exception as e:  # don't shadow the user's real bug with a sniff bug
            print(
                f"[{type(self).__name__}.validate_prepared] could not sniff "
                f"{path!r} ({e!r}); skipping schema check."
            )
            return

        missing = expected - cols
        if missing:
            raise ValueError(
                f"{type(self).__name__}.prepare() wrote {path!r} but it is "
                f"missing required column(s) {sorted(missing)} "
                f"(input_key={self.input_key!r}, label_key={self.label_key!r}). "
                f"Columns present: {sorted(cols)}. "
                "Either rename the column(s) your prepare() writes, or set "
                "input_key/label_key on your DatasetConfig subclass to match."
            )


class HuggingFaceDataset(DatasetConfig):
    """Dataset backed by a HuggingFace ``datasets`` repo.

    Subclass and set ``hf_repo`` plus column mappings. When
    ``input_column`` and ``output_column`` are set, ``prepare()``
    auto-wraps rows into chat-message format
    (``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``).
    """

    _type: DatasetType = DatasetType.HUGGING_FACE
    hf_repo: str = ""
    hf_split: str = "train"
    hf_config: str | None = None
    output_format: str = "parquet"
    input_column: str = ""
    output_column: str = ""
    system_prompt: str = ""
    prompt_template: str = "{input}"
    n_rows: int = 0
    label_key: str = "label"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not self.input_key and self.input_column and self.output_column:
            self.input_key = "messages"
        if "dataset_id" not in kwargs:
            self.dataset_id = f"{self.hf_repo}-{self.hf_split}-{uuid.uuid4()}"
        self._validate()

    @property
    def name(self) -> str:
        return self.hf_repo

    def load(self, split: Literal["all", "train", "eval"] = "all") -> Any:
        from datasets import load_dataset

        ds = load_dataset(
            self.hf_repo,
            self.hf_config,
            split=self.hf_split,
        )
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds

    def _format_for_training(self, ds):
        if not (self.input_column and self.output_column):
            return ds

        in_col, out_col = self.input_column, self.output_column
        sys_prompt = self.system_prompt
        template = self.prompt_template
        label_key = self.label_key

        def _to_chat(row: dict) -> dict:
            user_content = template.format(input=row[in_col])
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": user_content})
            msgs.append({"role": "assistant", "content": row[out_col]})
            return {"messages": msgs, label_key: str(row[out_col])}

        return ds.map(_to_chat, remove_columns=ds.column_names)

    def to_pandas(self, *, formatted: bool = False):
        ds = self.load()
        if formatted:
            ds = self._format_for_training(ds)
        return ds.to_pandas()

    def _write_split(self, ds, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.output_format == "jsonl":
            ds.to_json(path, orient="records", lines=True)
        else:
            ds.to_parquet(path)

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        ds = self._format_for_training(self.load())
        self._write_split(ds, path)

        if eval_paths:
            for eval_path in eval_paths.values():
                eval_ds = self._format_for_training(self.load())
                self._write_split(eval_ds, eval_path)


class MultimodalDataset(DatasetConfig):
    """Modality-agnostic dataset for image / audio / video RL.

    Each row pairs a text ``prompt`` with one or more ``media`` items and a
    ``label``. ``prepare()`` writes the media verbatim into a column named by
    ``media_column`` (default ``"<modality>s"``), and the column is surfaced to
    the trainer/rollout via ``multimodal_keys`` (``{modality: media_column}``,
    e.g. slime's ``--multimodal-keys``). Media items may be URLs, local paths,
    or base64 data — whatever the serving engine accepts; the gym never
    inspects them.

    Pass ``rows=[{"prompt": str, "media": list, "label": Any}, ...]`` or
    subclass and override the ``rows`` property.
    """

    input_key: str = "prompt"
    label_key: str = "label"
    modality: Literal["image", "audio", "video"] = "audio"
    # TODO(ben/joy): gate-check media at this boundary so the evals dashboard can
    # reliably visualize it. Two parts: (1) normalize each emitted media item to a
    # canonical, browser-renderable container per modality (audio->wav, image->png/
    # jpeg) instead of trusting whatever format the user brings — the audio tutorial's
    # dataset already re-encodes to wav, make it the convention here; (2) validate the
    # data-URI
    # MIME matches `modality`. Pairs with the dashboard fallback in EvalsPage.svelte.
    media_column: str = ""
    output_format: str = "jsonl"

    def __init__(self, rows: list[dict[str, Any]] | None = None, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.modality not in ("image", "audio", "video"):
            raise ValueError(
                f"modality must be one of image/audio/video, got {self.modality!r}"
            )
        if not self.media_column:
            self.media_column = f"{self.modality}s"
        if self.input_key == self.media_column or self.label_key == self.media_column:
            raise ValueError("media_column must differ from input_key and label_key")
        # The whole feature in one line: name the media column for the framework.
        self.multimodal_keys = {self.modality: self.media_column}
        self._rows = list(rows or [])
        if not self.dataset_id:
            self.dataset_id = f"mm-{self.modality}-{uuid.uuid4()}"
        self._validate()

    @property
    def rows(self) -> list[DatasetRow]:
        return self._rows

    def _to_row(self, r: dict[str, Any]) -> DatasetRow:
        media = r["media"]
        return {
            self.input_key: r["prompt"],
            self.media_column: list(media)
            if isinstance(media, (list, tuple))
            else [media],
            self.label_key: r["label"],
        }

    def load(self) -> list[DatasetRow]:
        return [self._to_row(r) for r in self.rows]

    def _write_jsonl(self, rows: list[dict[str, Any]], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        rows = self.load()
        self._write_jsonl(rows, path)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_jsonl(rows, eval_path)


class ToolathlonTrajectoryDataset(DatasetConfig):
    """Expert Toolathlon trajectories as one ``(input, label)`` row per task.

    Each row's ``label`` packs the full ground-truth trajectory — the ordered golden tool calls, the
    aligned tool observations, the task request, and the tool catalog — which a rollout/eval later
    *indexes into* (pick a start step ``K`` and rebuild the prompt prefix). That trajectory indexing
    lives in the consumer (the Toolathlon environment), not here; this dataset just emits the pairs.

    Train and eval are split by task-name allowlists (``train_tasks`` / ``eval_tasks``);
    :meth:`golden_by_task` exposes the per-task expert action sequence that drives a Toolathlon
    environment's snapshot-library builder.

    The original workspace prefix in each raw trace is remapped onto ``workspace_path``, which must
    equal the Toolathlon environment's ``ToolathlonEnvConfig.workspace_path`` (both default to
    ``/task/workspace``) so the trace resolves against the live sandbox.
    """

    hf_repo: str = "hkust-nlp/Toolathlon-Trajectories"
    source_file: str = "deepseek-v3.2-exp_1.jsonl"
    output_format: str = "jsonl"
    label_key: str = "label"
    obs_limit: int = 1500  # truncate each observation to N chars in context
    train_tasks: tuple[str, ...] = ()
    eval_tasks: tuple[str, ...] = ()
    # Workspace-path remap; keep ``workspace_path`` in sync with ToolathlonEnvConfig.workspace_path.
    orig_workspace: str = "/workspace/dumps/workspace"
    workspace_path: str = "/task/workspace"
    # Placeholder prompt for the materialized row; the real step-K prefix is rebuilt at rollout time,
    # so ``sample.prompt`` is not used directly — this just lets the framework load/tokenize the row.
    placeholder_system_prompt: str = (
        "You are a tool-calling agent. Output a JSON tool call."
    )

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not self.input_key:
            self.input_key = "messages"
        if "dataset_id" not in kwargs:
            self.dataset_id = f"toolathlon-trajectories-{uuid.uuid4()}"
        self._validate()

    @property
    def name(self) -> str:
        return self.hf_repo

    def load(self, split: Literal["all", "train", "eval"] = "all") -> list[dict]:
        return [
            self._make_row(traj)
            for traj in self._load_trajectories(split)
            if self._golden_calls(
                traj["messages"]
            )  # skip degenerate traces with no tool calls
        ]

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._write_rows(self.load("train"), path)
        if eval_paths:
            eval_rows = self.load("eval")
            for eval_path in eval_paths.values():
                self._write_rows(eval_rows, eval_path)

    def golden_by_task(self) -> dict[str, list[dict]]:
        """Map ``task_name -> golden tool-call list`` over all tasks (drives the env snapshot builder)."""
        return {
            traj["task_name"]: self._golden_calls(traj["messages"])
            for traj in self._load_trajectories("all")
        }

    # ── trace loading + parsing ──────────────────────────────────────────────

    @staticmethod
    def _write_rows(rows: list[dict], path: str) -> None:
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    def _split_tasks(self, split: str) -> set[str]:
        if split == "train":
            return set(self.train_tasks)
        if split == "eval":
            return set(self.eval_tasks)
        return set(self.train_tasks) | set(self.eval_tasks)

    def _load_trajectories(
        self, split: Literal["all", "train", "eval"] = "all"
    ) -> list[dict]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=self.hf_repo, filename=self.source_file, repo_type="dataset"
        )
        keep = self._split_tasks(split)  # empty -> no filtering (keep all tasks)
        out: list[dict] = []
        with open(path) as f:
            for line in f:
                raw = json.loads(line)
                if keep and raw.get("task_name") not in keep:
                    continue
                # Remap the original workspace prefix -> our workspace_path on the raw serialized
                # trajectory (covers golden-call args, observations, and the task text in one pass,
                # including arguments stored as nested JSON strings).
                msgs_text = (raw.get("messages") or "[]").replace(
                    self.orig_workspace, self.workspace_path
                )
                msgs = json.loads(msgs_text)
                if len(msgs) < 3:
                    continue
                out.append(
                    {
                        "task_name": raw["task_name"],
                        "messages": msgs,
                        "tool_calls_meta": json.loads(raw.get("tool_calls", "{}"))
                        if raw.get("tool_calls")
                        else {},
                    }
                )
        return out

    def _make_row(self, traj: dict) -> dict:
        msgs = traj["messages"]
        golden = self._golden_calls(msgs)
        task_request = self._task_request(msgs)
        label = json.dumps(
            {
                "task_name": traj["task_name"],
                "total_steps": len(golden),
                "task_request": task_request,
                "golden_calls": golden,
                "observations": self._observations(msgs),
                "tool_schemas": self._tool_schemas(traj["tool_calls_meta"]),
            }
        )
        messages = [
            {"role": "system", "content": self.placeholder_system_prompt},
            {"role": "user", "content": task_request},
        ]
        return {"messages": messages, "label": label}

    def _golden_calls(self, msgs: list[dict]) -> list[dict]:
        """The ordered list of expert (golden) tool calls in a trajectory."""
        calls = []
        for m in msgs:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                tc = m["tool_calls"]
                if tc and isinstance(tc[0], dict) and "function" in tc[0]:
                    fn = tc[0]["function"]
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append({"name": fn["name"], "arguments": args})
        return calls

    def _observations(self, msgs: list[dict]) -> list[str]:
        """Ordered tool observations (one per executed call), truncated to ``obs_limit``."""
        return [
            str(m.get("content", ""))[: self.obs_limit]
            for m in msgs
            if m.get("role") == "tool"
        ]

    def _task_request(self, msgs: list[dict]) -> str:
        for m in msgs:
            if m.get("role") == "user" and str(m.get("content", "")).strip():
                return str(m["content"])
        return ""

    def _tool_schemas(self, meta: dict) -> dict:
        # The exact tool catalog the expert was given via the OpenAI tools= param (gateway tools/list).
        # name -> {description, parameters} so the prompt can show which tools exist + their args.
        schemas = {}
        for t in meta.get("tools", []):
            if isinstance(t, dict) and "function" in t:
                fn = t["function"]
                schemas[fn["name"]] = {
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
        return schemas
