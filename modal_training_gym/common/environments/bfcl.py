"""In-process Berkeley Function Calling Leaderboard (BFCL) v3/v4 multi-turn environment.

BFCL's multi-turn categories
(https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) grade agents against small,
*in-process* Python state machines (``GorillaFileSystem``, ``TwitterAPI``, ``TicketAPI``,
``TravelAPI``, ``VehicleControlAPI``, ...) rather than live sandboxes or MCP servers: a tool call is a
Python expression (``mv(source='a', destination='b')``) evaluated against the live instance, and
grading is an attribute-level state diff plus a response-subsequence check. This means the Modal
sandbox / directory-snapshot apparatus in :mod:`.toolathlon` is unnecessary here — replaying ground
truth and grading both happen in plain Python, in milliseconds, with no container or network involved.

This module is a thin, RL-rollout-friendly wrapper around the upstream ``bfcl_eval`` package
(https://pypi.org/project/bfcl-eval/). It is *not* vendored — ``bfcl_eval`` is imported lazily (see
:func:`_bfcl_eval`) so importing :mod:`modal_training_gym` doesn't pick up its large, mostly-unrelated
dependency closure (sentence-transformers, faiss, several model-vendor SDKs, ...). Callers must
``pip install bfcl-eval`` themselves (the tutorial's image/install cells do this).

Three pieces, mirroring :mod:`.toolathlon`'s shape:

- **Data** — :class:`BfclMultiTurnDataset` loads a BFCL multi-turn category (default:
  ``BFCL_v4_multi_turn_base``) plus its ``possible_answer`` ground truth straight out of the installed
  ``bfcl_eval`` package, flattens each task's per-turn ground-truth calls into one ordered sequence,
  and replays it once (in-process) to capture the observation text for every step — the BFCL analogue
  of Toolathlon's expert trajectory.
- **Environment** — :class:`BfclTurnEnvironment` holds one fresh set of live class instances per
  episode (no global/module-level caching, unlike upstream's own
  ``execute_multi_turn_func_call`` — that caching is unsafe for concurrent online rollouts of the same
  task) and exposes the same ``step``/``evaluate`` shape as :mod:`.base`. Grading reuses ``bfcl_eval``'s
  own ``state_checker``/``response_checker`` rather than reimplementing them.
- **Prompting** — :func:`build_prefix_messages` / :func:`tool_schemas_to_openai` mirror the Toolathlon
  helpers of the same name, so a caller can swap one environment for the other with minimal churn.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.environments.base import (
    EvalVerdict,
    Environment,
    Observation,
    StepResult,
    ToolCall,
)

DEFAULT_CATEGORY = "BFCL_v4_multi_turn_base"
# The leaderboard ships no train/eval split (it's a benchmark, not a training set), so we carve our
# own: the last N ids of the category are held out for eval.
DEFAULT_EVAL_TAIL = 30

# BFCL has no explicit "done" tool — a turn simply ends when the model stops emitting calls. Kept
# (empty) for interface parity with callers that special-case Toolathlon's ``DONE_TOOLS``.
DONE_TOOLS: frozenset[str] = frozenset()

_JSON_TYPE_MAP = {
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "float": "number",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
}


def _bfcl_eval():
    """Import ``bfcl_eval`` lazily (see module docstring)."""
    try:
        import bfcl_eval
    except ImportError as e:
        raise ImportError(
            "This requires the `bfcl-eval` package (`uv pip install bfcl-eval`); "
            "see https://pypi.org/project/bfcl-eval/."
        ) from e
    return bfcl_eval


def _backend_mappings() -> tuple[dict[str, str], dict[str, str], list[str]]:
    from bfcl_eval.constants.executable_backend_config import (
        CLASS_FILE_PATH_MAPPING,
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
        STATELESS_CLASSES,
    )

    return CLASS_FILE_PATH_MAPPING, MULTI_TURN_FUNC_DOC_FILE_MAPPING, STATELESS_CLASSES


def _data_dir() -> str:
    bfcl_eval = _bfcl_eval()
    return os.path.join(os.path.dirname(bfcl_eval.__file__), "data")


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Call-string <-> dict ─────────────────────────────────────────────────────


def parse_call_string(call: str) -> dict[str, Any]:
    """Parse a BFCL ground-truth call string (e.g. ``"mv(source='a', destination='b')"`` or the
    positional ``"cd('archives')"``) into ``{"name": str, "arguments": dict}``.

    Ground-truth call strings are always a single call with literal/keyword arguments, so
    ``ast.literal_eval`` per-argument is sufficient — and, unlike upstream's ``eval()``-based
    execution helpers, this never executes arbitrary code.
    """
    node = ast.parse(call.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Not a call expression: {call!r}")
    name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
    arguments: dict[str, Any] = {
        f"_pos{i}": ast.literal_eval(arg) for i, arg in enumerate(node.args)
    }
    for kw in node.keywords:
        arguments[kw.arg] = ast.literal_eval(kw.value)
    return {"name": name, "arguments": arguments}


def _normalize_arguments(
    owner: Any, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Resolve ``_posN`` placeholders (from positional ground-truth calls) to real keyword names
    using ``owner``'s live method signature, so stored calls are always keyword-args (clean to show
    a model in a tool-call message and to re-execute later)."""
    positional = sorted(
        ((k, v) for k, v in arguments.items() if k.startswith("_pos")),
        key=lambda kv: int(kv[0][4:]),
    )
    if not positional:
        return arguments
    try:
        param_names = [
            p for p in inspect.signature(getattr(owner, name)).parameters if p != "self"
        ]
    except (TypeError, ValueError):
        param_names = []
    normalized = dict(zip(param_names, (v for _, v in positional)))
    normalized.update({k: v for k, v in arguments.items() if not k.startswith("_pos")})
    return normalized


# ── Execution ────────────────────────────────────────────────────────────────


def _instantiate(
    class_name: str, initial_config: dict, *, long_context: bool = False
) -> Any:
    import importlib

    class_file_path_mapping, _, stateless_classes = _backend_mappings()
    module = importlib.import_module(class_file_path_mapping[class_name])
    instance = getattr(module, class_name)()
    if class_name not in stateless_classes:
        instance._load_scenario(
            deepcopy(initial_config.get(class_name, {})), long_context=long_context
        )
    return instance


def build_instances(
    involved_classes: list[str], initial_config: dict
) -> dict[str, Any]:
    """One fresh instance per involved class.

    Unlike upstream's ``execute_multi_turn_func_call`` — which memoizes instances in module
    ``globals()`` keyed by ``(model_name, test_entry_id, class_name)`` for its offline batch-eval
    harness — this always builds fresh instances, which is the correct (and only safe) behavior for
    concurrent online rollouts that may replay the same task at the same time.
    """
    return {name: _instantiate(name, initial_config) for name in involved_classes}


def _method_owner(instances: dict[str, Any], method_name: str) -> Any | None:
    if method_name.startswith("_"):
        return None
    for instance in instances.values():
        if hasattr(type(instance), method_name):
            return instance
    return None


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except TypeError:
            return str(result)
    return str(result)


def execute_call(instances: dict[str, Any], call: dict[str, Any]) -> tuple[str, bool]:
    """Run one ``{"name", "arguments"}`` call against ``instances``. Returns ``(result_text,
    is_error)``. ``arguments`` must already be a plain keyword dict (no ``_posN`` placeholders) —
    student-generated calls always are; ground-truth calls are normalized once during :func:`replay`.

    Arguments are deep-copied before the call: some upstream BFCL methods store a list/dict argument
    by reference into instance state (e.g. ``TwitterAPI.post_tweet``'s ``mentions=[]`` default), so a
    caller that reuses the same argument object across multiple ``step()`` calls (or across two
    environments) would otherwise silently corrupt unrelated state through that aliasing.
    """
    owner = _method_owner(instances, call["name"])
    if owner is None:
        return f"Error during execution: unknown function {call['name']!r}", True
    try:
        result = getattr(owner, call["name"])(**deepcopy(call.get("arguments") or {}))
        return _stringify(result), False
    except Exception as e:  # mirrors upstream's eval()-based catch-all
        return f"Error during execution: {e}", True


def replay(
    involved_classes: list[str], initial_config: dict, calls: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Fresh instances + the observation text for each of ``calls``, executed in order.

    Mutates each ``call["arguments"]`` in place to resolve any ``_posN`` placeholders to real keyword
    names (see :func:`_normalize_arguments`), so callers that pass in freshly-parsed ground-truth
    calls get back clean, display-ready call dicts as a side effect.
    """
    instances = build_instances(involved_classes, initial_config)
    observations = []
    for call in calls:
        owner = _method_owner(instances, call["name"])
        if owner is not None:
            call["arguments"] = _normalize_arguments(
                owner, call["name"], call.get("arguments") or {}
            )
        text, _is_error = execute_call(instances, call)
        observations.append(text)
    return instances, observations


# ── Tool schemas ─────────────────────────────────────────────────────────────


def to_json_schema(node: Any) -> Any:
    """Recursively convert a BFCL func-doc schema fragment (``dict``/``list`` typed) to JSON-Schema
    typing (``object``/``array``), e.g. for use with the ``jsonschema`` package."""
    if isinstance(node, dict):
        out = {k: to_json_schema(v) for k, v in node.items() if k != "default"}
        if out.get("type") in _JSON_TYPE_MAP:
            out["type"] = _JSON_TYPE_MAP[out["type"]]
        return out
    if isinstance(node, list):
        return [to_json_schema(v) for v in node]
    return node


def tool_schemas_to_openai(tool_schemas: dict) -> list[dict]:
    """Render ``{name: {"description", "parameters"}}`` (BFCL's func-doc shape) as the OpenAI
    ``tools=`` list a chat template expects, converting BFCL's ``dict``/``list`` JSON types to
    JSON-Schema's ``object``/``array``."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": to_json_schema(spec.get("parameters", {})),
            },
        }
        for name, spec in (tool_schemas or {}).items()
    ]


def load_func_docs(
    involved_classes: list[str], excluded_function: list[str] | None = None
) -> dict:
    """The BFCL function docs for ``involved_classes``, minus any ``excluded_function`` (BFCL omits
    these from the offered catalog when an equivalent function already covers the same ground truth,
    e.g. excluding ``cp`` when the task is solved with ``mv``)."""
    _, doc_file_mapping, _ = _backend_mappings()
    excluded = set(excluded_function or [])
    data_dir = _data_dir()
    schemas: dict[str, dict] = {}
    for class_name in involved_classes:
        doc_file = doc_file_mapping.get(class_name)
        if not doc_file:
            continue
        for doc in _load_jsonl(os.path.join(data_dir, "multi_turn_func_doc", doc_file)):
            if doc["name"] in excluded:
                continue
            schemas[doc["name"]] = {
                "description": doc.get("description", ""),
                "parameters": doc.get("parameters", {}),
            }
    return schemas


# ── Prompt-prefix reconstruction ────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are a tool-using agent completing a user's request with the function tools provided to you. Work one step at a time.

Rules:
- Make EXACTLY ONE tool call per turn. Emit only the tool call — no extra prose, narration, or markdown fences around it.
- Use only the tools provided to you, with their exact names. Do not invent tools, arguments, or file paths.
- Each user message may require several tool calls before the request is satisfied; keep calling tools until the request is complete, then stop calling tools.
- After each tool result, check whether it succeeded before continuing; do not blindly repeat a failed call with the same arguments.

Emit every tool call in exactly this format — a single <function> block wrapped in <emoji>, with one <parameter> block per argument. For example, a real call that moves a file (values abbreviated with … here) looks like:

<emoji>
<function=mv>
<parameter=source>
final_report.pdf
</parameter>
<parameter=destination>
temp
</parameter>
</function>
</emoji>

Use the actual tool name, parameters, and full (untruncated) values required by your task; the call above is only an illustration of the wire format."""


def render_tool_catalog(tool_schemas: dict, desc_chars: int = 160) -> str:
    """Render the available-tools catalog as ``name(arg, required*)`` lines + a short description."""
    lines = []
    for name in sorted(tool_schemas or {}):
        spec = tool_schemas[name]
        desc, params = spec.get("description", ""), spec.get("parameters", {})
        props = (params or {}).get("properties", {}) if isinstance(params, dict) else {}
        required = (
            set((params or {}).get("required", []) or [])
            if isinstance(params, dict)
            else set()
        )
        sig = ", ".join(f"{k}*" if k in required else k for k in props)
        lines.append(f"- {name}({sig}): {desc[:desc_chars]}")
    return "\n".join(lines)


def default_system_prompt(tool_schemas: dict) -> str:
    return f"{DEFAULT_SYSTEM_PROMPT}\n\nAvailable tools:\n{render_tool_catalog(tool_schemas)}"


def build_prefix_messages(label: dict, K: int, *, obs_limit: int = 1500) -> list[dict]:
    """Reconstruct the chat-message prefix after the first ``K`` ground-truth calls have happened.

    Walks BFCL's turns in order, emitting each turn's user message followed by an assistant
    tool-call + tool-result pair for every ground-truth call in that turn that falls within the
    first ``K`` calls overall. Stops as soon as ``K`` calls have been shown — which may land exactly
    on a fresh, not-yet-acted-on user turn, or mid-turn.
    """
    observations = label.get("observations")
    if observations is None:
        _, observations = replay(
            label["involved_classes"],
            label["initial_config"],
            deepcopy(label["flattened_calls"]),
        )
    messages = [
        {
            "role": "system",
            "content": default_system_prompt(label.get("tool_schemas", {})),
        }
    ]
    shown = 0
    for turn in label["turns"]:
        messages.append({"role": "user", "content": turn["user"]})
        for call in turn["calls"]:
            if shown >= K:
                return messages
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{shown}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                # A raw dict, not a JSON string: this prefix is fed straight to
                                # ``tokenizer.apply_chat_template`` (not an OpenAI-compatible HTTP
                                # API layer that would parse a JSON string first), and Qwen3.6's
                                # chat template iterates arguments with Jinja's ``|items`` filter,
                                # which requires an actual mapping (mirrors
                                # ``.toolathlon.build_prefix_messages``'s convention).
                                "arguments": call["arguments"],
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{shown}",
                    "content": str(observations[shown])[:obs_limit],
                }
            )
            shown += 1
    return messages


def prune_prefix(messages: list[dict], max_messages: int) -> list[dict]:
    """Keep the system + first user message and the most recent ``max_messages`` exchanges —
    mirrors :func:`.toolathlon.prune_prefix`'s role, for callers that need to budget context length."""
    if len(messages) <= max_messages:
        return messages
    head = messages[:2]
    tail = messages[-(max_messages - len(head)) :]
    return head + tail


# ── Environment ──────────────────────────────────────────────────────────────


@dataclass
class BfclTurnEnvironment(Environment):
    """One live multi-turn BFCL episode: a set of in-process class instances seeded by replaying the
    first ``K`` ground-truth calls, then driven forward by an agent's own tool calls via :meth:`step`.
    """

    label: dict
    instances: dict[str, Any] = field(default_factory=dict)
    exec_results: list[str] = field(default_factory=list)
    # How many ground-truth calls were already replayed to seed `instances` before this episode
    # started (see `build_env`). Only the *remaining* ground-truth responses (from this index
    # onward) need to show up in `exec_results` — the prefix was identical by construction.
    K: int = 0

    def step(self, action: ToolCall) -> StepResult:
        call = {"name": action.name, "arguments": action.arguments or {}}
        text, is_error = execute_call(self.instances, call)
        self.exec_results.append(text)
        return StepResult(
            observation=Observation(text=text, is_error=is_error), done=False
        )

    def evaluate(self) -> EvalVerdict:
        """Grade the episode: deep state diff (``state_checker``) + response-subsequence check
        (``response_checker``) against a freshly-replayed copy of the *full* ground-truth trajectory —
        reusing ``bfcl_eval``'s own checkers rather than reimplementing them."""
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            response_checker,
            state_checker,
        )

        ground_truth_instances, ground_truth_results = replay(
            self.label["involved_classes"],
            self.label["initial_config"],
            deepcopy(self.label["flattened_calls"]),
        )
        state_result = state_checker(self.instances, ground_truth_instances)
        if not state_result["valid"]:
            return EvalVerdict(passed=False, detail=state_result["error_message"])
        # The prefix (calls [0:K]) is identical for the model and ground truth by construction
        # (both replayed it to seed state) — only what the agent did from K onward needs checking.
        response_result = response_checker(
            self.exec_results, ground_truth_results[self.K :], 0
        )
        if not response_result["valid"]:
            return EvalVerdict(passed=False, detail=response_result["error_message"])
        return EvalVerdict(passed=True)


def build_env(label: dict, K: int) -> BfclTurnEnvironment:
    """Build a fresh :class:`BfclTurnEnvironment` with state fast-forwarded through the first ``K``
    ground-truth calls — the in-process analogue of restoring a Toolathlon directory snapshot."""
    calls = deepcopy(label["flattened_calls"][:K])
    instances, _ = replay(label["involved_classes"], label["initial_config"], calls)
    return BfclTurnEnvironment(label=label, instances=instances, K=K)


# ── Trajectory dataset ───────────────────────────────────────────────────────


def _first_user_text(turn_messages: list[dict]) -> str:
    for m in turn_messages:
        if m.get("role") == "user" and str(m.get("content", "")).strip():
            return str(m["content"])
    return ""


@dataclass(frozen=True)
class BfclMultiTurnConfig:
    """Which BFCL multi-turn category to load and how to carve a train/eval split out of it."""

    category: str = DEFAULT_CATEGORY
    eval_tail: int = DEFAULT_EVAL_TAIL
    obs_limit: int = 1500


DEFAULT_CONFIG = BfclMultiTurnConfig()


class BfclMultiTurnDataset(DatasetConfig):
    """Loads a BFCL multi-turn category straight out of the installed ``bfcl_eval`` package.

    Each row's ``label`` carries everything the curriculum/rollout/reward code needs: the per-turn
    user text, the (keyword-normalized) ground-truth calls flattened across all turns, the observation
    text for each of those calls (so the K-step prefix never needs to re-execute anything), the tool
    catalog for the task's involved classes, and the raw ``initial_config``/``involved_classes`` so an
    episode can be rebuilt at any step. There is no train/eval split upstream (BFCL is a benchmark,
    not a training set) — :attr:`config`.eval_tail ids are held out as eval.
    """

    input_key: str = "messages"
    label_key: str = "label"
    # `prepare()` writes JSON-lines (below); tell the launcher's data-path
    # resolver (`SlimeRecipe._resolve_data_paths`, default: "parquet") to name
    # the on-volume file `*.jsonl` accordingly — otherwise slime tries to read
    # a JSON-lines file as Parquet and fails with "Parquet magic bytes not found".
    output_format: str = "jsonl"

    def __init__(
        self,
        split: str = "train",
        config: BfclMultiTurnConfig = DEFAULT_CONFIG,
        **kwargs: Any,
    ) -> None:
        self._split = split
        self.config = config
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _entries(self) -> list[dict]:
        return _load_jsonl(os.path.join(_data_dir(), f"{self.config.category}.json"))

    def _ground_truths(self) -> dict[str, list[list[str]]]:
        path = os.path.join(
            _data_dir(), "possible_answer", f"{self.config.category}.json"
        )
        return {e["id"]: e["ground_truth"] for e in _load_jsonl(path)}

    def _ids_for_split(self, ids: list[str]) -> list[str]:
        tail = self.config.eval_tail
        if self._split == "eval":
            return ids[-tail:] if tail else []
        if self._split == "train":
            return ids[:-tail] if tail else list(ids)
        return ids

    def _make_row(self, entry: dict, ground_truth: list[list[str]]) -> dict:
        turns = [
            {
                "user": _first_user_text(turn_messages),
                "calls": [parse_call_string(c) for c in calls],
            }
            for turn_messages, calls in zip(entry["question"], ground_truth)
        ]
        flattened_calls = [c for turn in turns for c in turn["calls"]]
        _, observations = replay(
            entry["involved_classes"], entry["initial_config"], flattened_calls
        )
        label = {
            "task_id": entry["id"],
            "initial_config": entry["initial_config"],
            "involved_classes": entry["involved_classes"],
            "excluded_function": entry.get("excluded_function", []),
            "turns": [{"user": t["user"], "calls": t["calls"]} for t in turns],
            "flattened_calls": flattened_calls,
            "observations": [str(o)[: self.config.obs_limit] for o in observations],
            "tool_schemas": load_func_docs(
                entry["involved_classes"], entry.get("excluded_function")
            ),
            "total_steps": len(flattened_calls),
        }
        messages = [
            {
                "role": "system",
                "content": "You are a tool-calling agent. Output a JSON tool call.",
            },
            {"role": "user", "content": turns[0]["user"] if turns else ""},
        ]
        return {"messages": messages, "label": json.dumps(label)}

    def _load_split(self) -> list[dict]:
        entries_by_id = {e["id"]: e for e in self._entries()}
        gt_by_id = self._ground_truths()
        ids = self._ids_for_split(list(entries_by_id.keys()))
        return [
            self._make_row(entries_by_id[i], gt_by_id[i])
            for i in ids
            if i in entries_by_id and i in gt_by_id and gt_by_id[i] and any(gt_by_id[i])
        ]

    def load(self, split: str = "all") -> list[dict]:
        if split in ("train", "eval"):
            self._split = split
        return self._load_split()

    def prepare(self, path: str, eval_paths: dict | None = None) -> None:
        """Write this instance's split to ``path``.

        ``eval_paths`` is ignored on purpose: BFCL training and offline eval each
        use their own ``BfclMultiTurnDataset(split=...)`` instance. Recipe
        resolvers still invent an ``eval.*`` companion path; callers must not
        require that file (see ``run_prepare_dataset`` / slime-miles ``exists``
        guards).
        """
        del eval_paths  # train/eval are separate DatasetConfig instances
        rows = self._load_split()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
