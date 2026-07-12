"""MegaGem Stage-C custom rollout bridge for slime.

The Slime recipe trains full Megatron weights and serves synchronized SGLang
engines. MegaGem's RL signal, however, is a per-turn row exported from
schema-v3 game transcripts. This module bridges those contracts:

* one dataset label describes a MegaGem seed group plus the selected row slot;
* the first slot request rolls a K-group of current-policy self-play games;
* the existing MegaGem row exporter computes rewards and credit assignment;
* each Slime sample is filled with the selected row tokens and metadata; and
* a custom advantage hook broadcasts the row's precomputed advantage to tokens.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


MEGAGEM_STAGE_C_ROLLOUT_CONTRACT = (
    "slime_current_self_kgroup_precomputed_advantage_v1"
)
MEGAGEM_STAGE_C_ADVANTAGE_PATH = (
    "modal_training_gym.frameworks.slime.megagem_stage_c_rollout."
    "megagem_precomputed_advantages"
)
MEGAGEM_STAGE_C_REWARD_PATH = (
    "modal_training_gym.frameworks.slime.megagem_stage_c_rollout."
    "megagem_precomputed_reward"
)

_TRAINABLE_ACTOR_ID = "trainable"
_DEFAULT_OPPONENT_ACTOR_ID = "current_self"
_REMOTE_MEGAGEM_ROOT = "/root/MegagemBench"
_PROMPT_LABEL_RE = re.compile(
    r"\[megagem-stage-c\]\s+seed=(?P<seed>\d+)\s+chart=(?P<chart>\S+)\s+"
    r"seat=(?P<seat>\d+)\s+k=(?P<k>\d+)"
    r"(?:\s+stride=(?P<stride>\d+))?"
    r"(?:\s+rows=(?P<rows>\d+))?"
)
_GROUP_STATES: dict[str, "_GroupState"] = {}
_GENERATE_STATES: dict[int, Any] = {}
_GAME_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}
_SIGNAL_SHIM_INSTALLED = False


@dataclass
class _GroupState:
    lock: asyncio.Lock
    task: asyncio.Task[list[dict[str, Any]]] | None = None
    rows: list[dict[str, Any]] | None = None
    assigned_slots: set[int] | None = None
    generation: int = -1


def _ensure_megagem_path() -> Path:
    root = Path(os.environ.get("MEGAGEM_BENCH_ROOT", _REMOTE_MEGAGEM_ROOT))
    if not root.exists():
        raise RuntimeError(
            f"MegaGemBench root {root} is not present in the training image. "
            "Stage C requires MegaGem_Qwen3_4B_StageC_Recipe.image_overlay "
            "to add the repository at /root/MegagemBench."
        )
    for path in (root, root / "scripts" / "phase2"):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def _install_non_main_thread_signal_shim(*, force: bool = False) -> None:
    """Disable signal-based timeouts in Ray rollout worker threads.

    MegaGem imports ``verifiers``, and that stack may install signal handlers for
    timeouts.  Slime executes custom generation inside Ray worker threads, where
    Python intentionally raises ``ValueError: signal only works in main thread``.
    Signals cannot fire correctly there, so the least surprising behavior is to
    no-op the setup in those non-main threads before importing MegaGem.
    """

    global _SIGNAL_SHIM_INSTALLED
    if _SIGNAL_SHIM_INSTALLED:
        return

    import signal
    import threading

    if not force and threading.current_thread() is threading.main_thread():
        return

    handlers: dict[int, Any] = {}

    def safe_signal(sig: int, handler: Any) -> Any:
        old = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return old

    signal.signal = safe_signal
    if hasattr(signal, "alarm"):
        signal.alarm = lambda seconds: 0
    if hasattr(signal, "setitimer"):
        signal.setitimer = lambda *args, **kwargs: (0.0, 0.0)
    _SIGNAL_SHIM_INSTALLED = True


def _json_label(sample: Any) -> dict[str, Any]:
    raw = getattr(sample, "label", None)
    if raw is None:
        metadata = getattr(sample, "metadata", None)
        if isinstance(metadata, dict):
            raw = metadata.get("label")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, str):
        match = _PROMPT_LABEL_RE.search(prompt)
        if match:
            return {
                "seed": int(match.group("seed")),
                "seed_stride": int(match.group("stride") or 128),
                "value_chart": match.group("chart"),
                "trainable_seat": int(match.group("seat")),
                "num_players": 3,
                "k": int(match.group("k")),
                "rows_per_group": int(match.group("rows") or 16),
                "opponent_actor_id": _DEFAULT_OPPONENT_ACTOR_ID,
            }
    raise RuntimeError("MegaGem Stage C sample is missing a JSON label")


def _int_label(label: dict[str, Any], key: str, default: int) -> int:
    value = label.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"MegaGem Stage C label field {key!r} is not an int") from exc


def _group_key(label: dict[str, Any]) -> str:
    payload = {
        "seed": _int_label(label, "seed", 0),
        "value_chart": str(label.get("value_chart", "A")),
        "trainable_seat": _int_label(label, "trainable_seat", 0),
        "num_players": _int_label(label, "num_players", 3),
        "k": _int_label(label, "k", 16),
        "rows_per_group": _int_label(label, "rows_per_group", 16),
        "seed_stride": _int_label(label, "seed_stride", 128),
    }
    return json.dumps(payload, sort_keys=True)


def _sample_slot(sample: Any, label: dict[str, Any]) -> int | None:
    if label.get("row_slot") is not None:
        return _int_label(label, "row_slot", 0)
    # Slime's incidental indices are rollout/global bookkeeping, not a stable
    # sibling slot contract. Stage C assigns slots inside the group state unless
    # the dataset label explicitly pins one.
    return None


def _get_generate_state(args: Any) -> Any:
    loop_key = id(asyncio.get_running_loop())
    state = _GENERATE_STATES.get(loop_key)
    if state is None:
        from slime.rollout.sglang_rollout import GenerateState

        state = GenerateState(args)
        _GENERATE_STATES.clear()
        _GENERATE_STATES[loop_key] = state
    return state


def _max_parallel_games(args: Any) -> int:
    raw = (
        os.environ.get("MEGAGEM_STAGE_C_MAX_PARALLEL_GAMES")
        or getattr(args, "megagem_max_parallel_games", None)
        or 256
    )
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 256


def _int_setting(args: Any, env_name: str, attr_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None:
        raw = getattr(args, attr_name, None)
    if raw is None:
        raw = default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _max_extra_games_per_group(args: Any, k: int) -> int:
    default = max(2, int(math.ceil(k * 0.25)))
    return max(
        0,
        _int_setting(
            args,
            "MEGAGEM_STAGE_C_EXTRA_GAMES_PER_GROUP",
            "megagem_extra_games_per_group",
            default,
        ),
    )


def _game_timeout_s(args: Any) -> float:
    raw = (
        os.environ.get("MEGAGEM_STAGE_C_GAME_TIMEOUT_S")
        or getattr(args, "megagem_game_timeout_s", None)
        or 600
    )
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 600.0


def _bool_setting(args: Any, env_name: str, attr_name: str, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        raw = getattr(args, attr_name, None)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _fail_open_groups(args: Any) -> bool:
    return _bool_setting(
        args,
        "MEGAGEM_STAGE_C_FAIL_OPEN_GROUPS",
        "megagem_fail_open_groups",
        True,
    )


def _min_success_games(args: Any, k: int) -> int:
    default = max(2, int(math.ceil(k * 0.75)))
    return min(
        k,
        max(
            1,
            _int_setting(
                args,
                "MEGAGEM_STAGE_C_MIN_SUCCESS_GAMES",
                "megagem_min_success_games",
                default,
            ),
        ),
    )


def _game_semaphore(args: Any) -> asyncio.Semaphore:
    limit = _max_parallel_games(args)
    loop_key = id(asyncio.get_running_loop())
    key = (loop_key, limit)
    sem = _GAME_SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.Semaphore(limit)
        _GAME_SEMAPHORES.clear()
        _GAME_SEMAPHORES[key] = sem
    return sem


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


def _messages_to_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return _apply_chat_template(tokenizer, messages)
    return "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
    )


def _sampling_params(defaults: dict[str, Any], api_kwargs: dict[str, Any]) -> dict[str, Any]:
    params = dict(defaults)
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "stop_token_ids",
    ):
        if key in api_kwargs and api_kwargs[key] is not None:
            params[key] = api_kwargs[key]
    max_new = (
        api_kwargs.get("max_new_tokens")
        or api_kwargs.get("max_completion_tokens")
        or api_kwargs.get("max_tokens")
    )
    if max_new is not None:
        params["max_new_tokens"] = int(max_new)
    # ``extra_body.chat_template_kwargs`` is for OpenAI-compatible chat servers.
    # This adapter posts already-rendered text to SGLang's raw ``/generate``
    # endpoint, where chat-template kwargs are not valid sampling parameters.
    params.setdefault("temperature", 1.0)
    params.setdefault("top_p", 1.0)
    return {k: v for k, v in params.items() if v is not None}


class _SGLangChatAdapter:
    """Small OpenAI-chat-shaped adapter backed by slime's SGLang /generate."""

    def __init__(self, args: Any, defaults: dict[str, Any]):
        self.args = args
        self.defaults = defaults
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_chat_completion)
        )

    async def _create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        **api_kwargs: Any,
    ) -> Any:
        del model
        from slime.utils.http_utils import post

        state = _get_generate_state(self.args)
        url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"
        text = _messages_to_text(state.tokenizer, messages)
        params = _sampling_params(self.defaults, api_kwargs)
        try:
            output = await post(url, {"text": text, "sampling_params": params})
        except Exception as exc:
            import httpx
            from openai import APIConnectionError

            raise APIConnectionError(
                message=(
                    "MegaGem Stage C SGLang /generate failed "
                    f"url={url} text_chars={len(text)} "
                    f"sampling_keys={sorted(params)} cause={type(exc).__name__}: {exc}"
                ),
                request=httpx.Request("POST", url),
            ) from exc
        if not isinstance(output, dict):
            import httpx
            from openai import APIConnectionError

            raise APIConnectionError(
                message=(
                    "MegaGem Stage C SGLang /generate returned non-dict "
                    f"{type(output).__name__} url={url}"
                ),
                request=httpx.Request("POST", url),
            )
        finish = (output.get("meta_info") or {}).get("finish_reason") or {}
        if finish.get("type") == "abort":
            text = ""
        else:
            text = output.get("text") or ""
        message = SimpleNamespace(content=text, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _round_recorder(game_data: dict[str, Any], models: list[str]):
    def record_round(
        *,
        round_num: int,
        auction: Any,
        bids: list[int],
        bid_reasoning: list[str] | None = None,
        winner_id: int | None = None,
        winning_bid: int | None = None,
        gems_won: list[str] | None = None,
        revealed_gems_available: list[str] | None = None,
        gem_revealed: str | None = None,
        reveal_reasoning: str | None = None,
        missions_completed: dict[int, list[str]] | None = None,
        mission_reasoning: dict[int, str] | None = None,
        game_state: Any = None,
        tiebreak_order_before: list[int] | None = None,
        bid_turn_records: list[Any] | None = None,
        reveal_turn_record: Any | None = None,
        actor_ids: list[str] | None = None,
        **_: Any,
    ) -> None:
        auction_type = auction.type.value
        round_data: dict[str, Any] = {
            "round_number": round_num,
            "auction": {
                "type": auction_type,
                "description": auction.get_description(),
            },
            "players": [],
            "value_display": {},
            "tiebreak_order": list(game_state.tiebreak_order),
            "tiebreak_order_before": (
                list(tiebreak_order_before)
                if tiebreak_order_before is not None
                else list(game_state.tiebreak_order)
            ),
            "available_missions": [
                m.to_dict() for m in getattr(game_state, "available_missions", [])
            ],
            "missions_completed": {
                str(pid): mids for pid, mids in (missions_completed or {}).items()
            },
        }
        if auction_type == "treasure":
            round_data["auction"]["gems_available"] = (
                list(revealed_gems_available or [])[: auction.gems]
            )
        elif auction_type == "loan":
            round_data["auction"]["loan_amount"] = auction.amount
        elif auction_type == "investment":
            round_data["auction"]["bonus"] = auction.bonus

        bid_by_player = {
            int(rec.player_id): rec for rec in (bid_turn_records or [])
        }
        for player_id in range(len(bids)):
            player = game_state.players[player_id]
            rec = bid_by_player.get(player_id)
            coins_before = player.coins
            if isinstance(_, dict) and "coins_before" in _:
                coins = _["coins_before"]
                if player_id < len(coins):
                    coins_before = coins[player_id]
            player_round = {
                "player_id": player_id,
                "model": models[player_id] if player_id < len(models) else "unknown",
                "bid": bids[player_id],
                "coins_before": coins_before,
                "coins_after": player.coins,
                "collection": list(player.collection),
                "collection_counts": player.get_collection_counts(),
                "hand": list(player.hand),
                "reasoning": (
                    bid_reasoning[player_id]
                    if bid_reasoning and player_id < len(bid_reasoning)
                    else ""
                ),
                "is_winner": player_id == winner_id,
            }
            if rec is not None:
                player_round.update(
                    {
                        "actor_id": rec.actor_id,
                        "prompt": rec.prompt,
                        "raw_response": rec.raw_response,
                        "parsed_action": rec.parsed_action,
                        "parse_method": rec.parse_method,
                        "parse_valid": rec.parse_valid,
                        "legal_valid": rec.legal_valid,
                        "default_used": rec.default_used,
                        "length_split": dict(rec.length_split),
                        "parse_error": rec.parse_error,
                        "legal_error": rec.legal_error,
                    }
                )
            elif actor_ids and player_id < len(actor_ids):
                player_round["actor_id"] = actor_ids[player_id]
            if player_id == winner_id:
                player_round.update(
                    {
                        "winning_bid": winning_bid,
                        "gems_won": list(gems_won or []),
                        "gem_revealed": gem_revealed,
                        "reveal_reasoning": reveal_reasoning or "",
                    }
                )
                if reveal_turn_record is not None and player_id == reveal_turn_record.player_id:
                    player_round["reveal"] = {
                        "actor_id": reveal_turn_record.actor_id,
                        "prompt": reveal_turn_record.prompt,
                        "raw_response": reveal_turn_record.raw_response,
                        "parsed_action": reveal_turn_record.parsed_action,
                        "parse_method": reveal_turn_record.parse_method,
                        "parse_valid": reveal_turn_record.parse_valid,
                        "legal_valid": reveal_turn_record.legal_valid,
                        "default_used": reveal_turn_record.default_used,
                        "final_reveal": reveal_turn_record.final_reveal,
                        "reasoning": reveal_turn_record.reasoning,
                        "length_split": dict(reveal_turn_record.length_split),
                        "parse_error": reveal_turn_record.parse_error,
                        "legal_error": reveal_turn_record.legal_error,
                    }
            if missions_completed and player_id in missions_completed:
                player_round["missions_completed"] = missions_completed[player_id]
                player_round["mission_reasoning"] = (
                    (mission_reasoning or {}).get(player_id, "")
                )
            round_data["players"].append(player_round)

        value_counts = game_state.get_value_display_counts()
        round_data["value_display"] = {
            color: {
                "count": count,
                "value_per_gem": game_state.get_gem_value(color),
            }
            for color, count in sorted(value_counts.items())
        }
        game_data["rounds"].append(round_data)

    return record_round


async def _roll_schema_game(
    args: Any,
    *,
    seed: int,
    label: dict[str, Any],
    k_index: int,
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    _ensure_megagem_path()
    _install_non_main_thread_signal_shim()
    from megagem import load_environment
    from src.environment.prompts import generate_system_prompt

    num_players = _int_label(label, "num_players", 3)
    value_chart = str(label.get("value_chart", "A"))
    trainable_seat = _int_label(label, "trainable_seat", 0)
    opponent_actor_id = str(label.get("opponent_actor_id", _DEFAULT_OPPONENT_ACTOR_ID))
    served_model = str(
        label.get("served_model")
        or getattr(args, "served_model_name", None)
        or getattr(args, "hf_checkpoint", None)
        or "megagem-current-policy"
    )
    model_labels = [served_model] * num_players
    actor_ids = [
        _TRAINABLE_ACTOR_ID if i == trainable_seat else opponent_actor_id
        for i in range(num_players)
    ]

    base_generation = {
        "max_completion_tokens": int(
            sampling_params.get("max_new_tokens")
            or sampling_params.get("max_completion_tokens")
            or getattr(args, "rollout_max_response_len", 2048)
        ),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    per_seat_api_params = []
    for i in range(num_players):
        params = dict(base_generation)
        if i == trainable_seat:
            params.update(
                {
                    "temperature": float(sampling_params.get("temperature", 1.0)),
                    "top_p": float(sampling_params.get("top_p", 1.0)),
                }
            )
        else:
            params.update({"temperature": 0.0, "top_p": 1.0})
        per_seat_api_params.append(params)

    env = load_environment(
        num_players=num_players,
        value_chart_id=value_chart,
        seed=seed,
        player_to_evaluate=trainable_seat,
        per_seat_api_params=per_seat_api_params,
    )
    client = _SGLangChatAdapter(args, sampling_params)
    game_data = {
        "metadata": {
            "schema_version": 3,
            "models": model_labels,
            "value_chart": value_chart,
            "seed": seed,
            "num_players": num_players,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": generate_system_prompt(),
            "stage_c_k_index": k_index,
            "stage_c_contract": MEGAGEM_STAGE_C_ROLLOUT_CONTRACT,
        },
        "rounds": [],
        "final_results": None,
        "statistics": None,
        "telemetry": None,
    }
    completion, state = await env.rollout(
        client=client,
        model=served_model,
        prompt=[],
        clients=[client] * num_players,
        models=model_labels,
        sampling_args={},
        actor_ids=actor_ids,
        round_callback=_round_recorder(game_data, model_labels),
    )
    del completion
    game_state_dict = state.get("game_state", {})
    value_counts = game_state_dict.get("value_display_counts", {})
    game_data["final_results"] = {
        "winner_id": state["winner_id"],
        "num_rounds": state["num_rounds"],
        "final_scores": state["final_scores"],
        "available_missions": game_state_dict.get("available_missions", []),
        "value_display_final": {
            color: {
                "count": count,
                "value_per_gem": env.value_chart.get_gem_value(count),
            }
            for color, count in sorted(value_counts.items())
        },
    }
    game_data["metadata"]["available_missions"] = game_state_dict.get(
        "available_missions", []
    )
    if state.get("model_chat_times_seconds") is not None:
        game_data["metadata"]["model_chat_times_seconds"] = [
            round(float(t), 3) for t in state["model_chat_times_seconds"]
        ]
    return game_data


def _balanced_select(rows: list[dict[str, Any]], n: int, *, seed: int) -> list[dict[str, Any]]:
    buckets: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in rows:
        buckets.setdefault(str(row["group_key"]), []).append(row)
    rng = random.Random(seed)

    def clone_row(row: dict[str, Any]) -> dict[str, Any]:
        cloned = dict(row)
        if isinstance(cloned.get("stage_c"), dict):
            cloned["stage_c"] = dict(cloned["stage_c"])
        return cloned

    def fresh_queues() -> list[list[dict[str, Any]]]:
        queues = []
        for values in buckets.values():
            q = [clone_row(row) for row in values]
            rng.shuffle(q)
            queues.append(q)
        return queues

    queues = fresh_queues()
    out: list[dict[str, Any]] = []
    while len(out) < n and buckets:
        if not any(queues):
            queues = fresh_queues()
        for q in queues:
            if q:
                out.append(q.pop())
                if len(out) == n:
                    break
    return out


def _exc_summary(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _exc_chain_summary(exc: BaseException, *, limit: int = 8) -> str:
    parts = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and len(parts) < limit and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(_exc_summary(cur))
        cur = cur.__cause__ or cur.__context__
    return " <- ".join(parts)


async def _collect_k_successes(
    roll_one: Any,
    *,
    target_successes: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, Exception]]]:
    successes: list[dict[str, Any]] = []
    failures: list[tuple[int, Exception]] = []
    pending: dict[asyncio.Task[dict[str, Any]], int] = {}
    next_index = 0

    def launch() -> None:
        nonlocal next_index
        task = asyncio.create_task(roll_one(next_index))
        pending[task] = next_index
        next_index += 1

    while next_index < target_successes and next_index < max_attempts:
        launch()

    while pending and len(successes) < target_successes:
        done, _ = await asyncio.wait(
            pending.keys(), return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            k_index = pending.pop(task)
            try:
                successes.append(task.result())
            except Exception as exc:
                failures.append((k_index, exc))
        while (
            len(successes) + len(pending) < target_successes
            and next_index < max_attempts
        ):
            launch()

    if len(successes) >= target_successes and pending:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending.keys(), return_exceptions=True)
    return successes, failures


def _filter_games_with_valid_rows(
    harness: Any,
    games: list[dict[str, Any]],
    *,
    trainable_seat: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    good: list[dict[str, Any]] = []
    bad: list[tuple[int, str]] = []
    for i, game in enumerate(games):
        try:
            harness.flatten_training_rows([game], trainable_seat=trainable_seat)
        except Exception as exc:
            bad.append((i, _exc_summary(exc)))
        else:
            good.append(game)
    return good, bad


def _fallback_group_rows(
    label: dict[str, Any],
    *,
    generation: int,
    rows_per_group: int,
    reason: str,
) -> list[dict[str, Any]]:
    seed = _int_label(label, "seed", 0)
    seed_stride = _int_label(label, "seed_stride", 128)
    actual_seed = seed + generation * seed_stride
    trainable_seat = _int_label(label, "trainable_seat", 0)
    group_key = f"{_group_key(label)}:fallback:g{generation}"
    prompt = (
        "[megagem-stage-c fallback] "
        f"seed={actual_seed} seat={trainable_seat} generation={generation}"
    )
    rows = []
    for slot in range(rows_per_group):
        rows.append(
            {
                "prompt": prompt,
                "completion": "",
                "precomputed_reward": 0.0,
                "precomputed_advantage": 0.0,
                "reward_components": {
                    "legal": 0.0,
                    "shaping": 0.0,
                    "terminal_correction": 0.0,
                    "terminal": 0.0,
                },
                "actor_id": _TRAINABLE_ACTOR_ID,
                "bucket": ["stage_c_fallback"],
                "ema_bucket": ["stage_c_fallback"],
                "group_key": group_key,
                "game_id": f"fallback:{actual_seed}:{slot}",
                "round_index": -1,
                "player_id": trainable_seat,
                "phase": "fallback",
                "is_terminal_turn": False,
                "stage_c": {
                    "dataset_seed": seed,
                    "actual_seed": actual_seed,
                    "seed_generation": generation,
                    "row_slot": slot,
                    "k": _int_label(label, "k", 16),
                    "k_success": 0,
                    "k_failures": 0,
                    "k_attempts": 0,
                    "rows_per_group": rows_per_group,
                    "fallback": True,
                    "fallback_reason": reason[:1000],
                },
            }
        )
    return rows


async def _roll_group_rows(
    args: Any,
    label: dict[str, Any],
    generation: int,
    sampling_params: dict[str, Any],
) -> list[dict[str, Any]]:
    _ensure_megagem_path()
    import _toy_grpo_harness as H
    from src.rl.export import contract_check

    seed = _int_label(label, "seed", 0)
    seed_stride = _int_label(label, "seed_stride", 128)
    actual_seed = seed + generation * seed_stride
    k = _int_label(label, "k", 16)
    rows_per_group = _int_label(label, "rows_per_group", 16)
    trainable_seat = _int_label(label, "trainable_seat", 0)
    started = time.perf_counter()
    max_extra = _max_extra_games_per_group(args, k)
    min_success = _min_success_games(args, k)
    game_timeout_s = _game_timeout_s(args)

    async def _roll_one(k_index: int) -> dict[str, Any]:
        sem = _game_semaphore(args)
        try:
            async with sem:
                return await asyncio.wait_for(
                    _roll_schema_game(
                        args,
                        seed=actual_seed,
                        label=label,
                        k_index=k_index,
                        sampling_params=sampling_params,
                    ),
                    timeout=game_timeout_s,
                )
        except Exception as exc:
            raise RuntimeError(
                "MegaGem Stage C failed while rolling one K-game "
                f"seed={actual_seed} k_index={k_index} "
                f"trainable_seat={trainable_seat} timeout_s={game_timeout_s:.1f}"
            ) from exc

    games, failures = await _collect_k_successes(
        _roll_one,
        target_successes=k,
        max_attempts=k + max_extra,
    )
    if len(games) < min_success:
        failure_text = "; ".join(
            f"k_index={i}: {_exc_chain_summary(exc)}" for i, exc in failures[:5]
        )
        raise RuntimeError(
            "MegaGem Stage C K-group had too few successful games "
            f"seed={actual_seed} trainable_seat={trainable_seat} "
            f"successes={len(games)}/{k} attempts={len(games) + len(failures)} "
            f"min_success={min_success} failures=[{failure_text}]"
        )

    try:
        rows = H.flatten_training_rows(games, trainable_seat=trainable_seat)
    except Exception as group_exc:
        filtered_games, bad_games = _filter_games_with_valid_rows(
            H, games, trainable_seat=trainable_seat
        )
        if len(filtered_games) < min_success:
            bad_text = "; ".join(f"game={i}: {msg}" for i, msg in bad_games[:5])
            raise RuntimeError(
                "MegaGem Stage C K-group row export failed and too few games "
                "survived filtering "
                f"seed={actual_seed} trainable_seat={trainable_seat} "
                f"valid_games={len(filtered_games)}/{len(games)} "
                f"min_success={min_success} "
                f"group_error={_exc_summary(group_exc)} bad_games=[{bad_text}]"
            ) from group_exc
        games = filtered_games
        rows = H.flatten_training_rows(games, trainable_seat=trainable_seat)
    contract_check(rows)
    if not rows:
        raise RuntimeError(
            f"MegaGem Stage C produced no trainable rows for seed={actual_seed}; "
            f"successful_games={len(games)} failures={len(failures)}"
        )
    selected = _balanced_select(rows, rows_per_group, seed=generation)
    wall_s = time.perf_counter() - started
    for i, row in enumerate(selected):
        row.setdefault("stage_c", {})
        row["stage_c"].update(
            {
                "dataset_seed": seed,
                "actual_seed": actual_seed,
                "seed_generation": generation,
                "row_slot": i,
                "k": k,
                "k_success": len(games),
                "k_failures": len(failures),
                "k_attempts": len(games) + len(failures),
                "rows_per_group": rows_per_group,
                "roll_group_wall_s": wall_s,
            }
        )
    print(
        "[megagem-stage-c] group complete "
        f"seed={actual_seed} seat={trainable_seat} generation={generation} "
        f"successes={len(games)}/{k} failures={len(failures)} "
        f"rows={len(rows)} selected={len(selected)} wall={wall_s:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return selected


async def _rows_for_sample(
    args: Any,
    sample: Any,
    label: dict[str, Any],
    sampling_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    explicit_slot = _sample_slot(sample, label)
    rows_per_group = _int_label(label, "rows_per_group", 16)
    key = _group_key(label)
    state = _GROUP_STATES.get(key)
    if state is None:
        state = _GroupState(lock=asyncio.Lock(), assigned_slots=set())
        _GROUP_STATES[key] = state

    async with state.lock:
        assigned = state.assigned_slots or set()
        if explicit_slot is None:
            free_slots = [i for i in range(rows_per_group) if i not in assigned]
            should_start = state.task is None or not free_slots
            slot = free_slots[0] if free_slots else 0
        else:
            slot = explicit_slot
            should_start = state.task is None or slot in assigned
        if should_start:
            state.generation += 1
            state.rows = None
            state.assigned_slots = set()
            state.task = asyncio.create_task(
                _roll_group_rows(args, label, state.generation, sampling_params)
            )
        assert state.assigned_slots is not None
        state.assigned_slots.add(slot)
        task = state.task
        task_generation = state.generation

    assert task is not None
    try:
        rows = await task
    except Exception as exc:
        cause = _exc_chain_summary(exc)
        async with state.lock:
            if state.task is task:
                state.task = None
                state.rows = None
                state.assigned_slots = set()
        if _fail_open_groups(args):
            rows = _fallback_group_rows(
                label,
                generation=task_generation,
                rows_per_group=rows_per_group,
                reason=cause,
            )
            print(
                "[megagem-stage-c] group task failed; using zero-advantage "
                f"fallback rows group_key={key} generation={task_generation} "
                f"slot={slot} cause={cause}",
                file=sys.stderr,
                flush=True,
            )
            return rows, slot, task_generation
        raise RuntimeError(
            "MegaGem Stage C group task failed "
            f"group_key={key} generation={task_generation} slot={slot} "
            f"cause_chain={cause}"
        ) from exc

    async with state.lock:
        if state.task is task:
            state.rows = rows
            generation = state.generation
        else:
            generation = state.generation
    return rows, slot, generation


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids:
        return list(ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    return [int(eos if eos is not None else 0)]


def _fill_rollout_compat_fields(sample: Any, response_ids: list[int]) -> None:
    """Leave native Slime probability fields unset unless we measured them.

    The MegaGem bridge already has row-level precomputed rewards/advantages.
    Do not invent ``rollout_log_probs``: fake zeros make Slime's policy loss
    reuse a constant tensor and produce zero gradients.  Do not invent top-p
    candidate sets either: a one-token top-p set masks the training softmax to
    the sampled token, making the log-probability identically zero and killing
    the gradient.  Stage C therefore runs with ``rollout_top_p=1.0`` until the
    rollout bridge can collect real top-p candidate vocab sets.
    """


def _fill_sample_from_row(args: Any, sample: Any, row: dict[str, Any]) -> Any:
    _ensure_megagem_path()
    from slime.utils.types import Sample
    from src.environment.prompts import generate_system_prompt

    state = _get_generate_state(args)
    messages = [
        {"role": "system", "content": generate_system_prompt()},
        {"role": "user", "content": str(row["prompt"])},
    ]
    prompt_text = _apply_chat_template(state.tokenizer, messages)
    response_text = str(row["completion"])
    prompt_ids = _token_ids(state.tokenizer, prompt_text)
    response_ids = _token_ids(state.tokenizer, response_text)

    sample.prompt = str(row["prompt"])
    sample.tokens = prompt_ids + response_ids
    sample.response_length = len(response_ids)
    sample.response = response_text
    sample.loss_mask = [1] * len(response_ids)
    sample.reward = float(row["precomputed_reward"])
    sample.status = Sample.Status.COMPLETED
    _fill_rollout_compat_fields(sample, response_ids)

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["megagem"] = {
        "contract": MEGAGEM_STAGE_C_ROLLOUT_CONTRACT,
        "precomputed_reward": float(row["precomputed_reward"]),
        "precomputed_advantage": float(row["precomputed_advantage"]),
        "group_key": row.get("group_key"),
        "game_id": row.get("game_id"),
        "round_index": row.get("round_index"),
        "player_id": row.get("player_id"),
        "phase": row.get("phase"),
        "bucket": row.get("bucket"),
        "stage_c": row.get("stage_c") or {},
    }
    sample.metadata = metadata
    return sample


def _sample_debug(sample: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("label", "sample_index", "rollout_idx", "response_length", "reward"):
        value = getattr(sample, name, None)
        if value is not None:
            out[name] = value
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, str):
        out["prompt_prefix"] = prompt[:240]
    elif prompt is not None:
        out["prompt_type"] = type(prompt).__name__
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        out["metadata_keys"] = sorted(str(k) for k in metadata)
    elif metadata is not None:
        out["metadata_type"] = type(metadata).__name__
    return out


async def megagem_stage_c_rollout(args: Any, sample: Any, sampling_params: dict[str, Any]) -> Any:
    """Slime custom rollout hook returning one selected MegaGem turn row."""

    label: dict[str, Any] | None = None
    try:
        label = _json_label(sample)
        rows, slot, _generation = await _rows_for_sample(
            args, sample, label, sampling_params or {}
        )
        if slot >= len(rows):
            raise RuntimeError(
                f"MegaGem Stage C row_slot={slot} out of range {len(rows)}"
            )
        return _fill_sample_from_row(args, sample, rows[slot])
    except Exception as exc:
        print(
            "[megagem-stage-c] custom_generate_function failed\n"
            f"  label={label}\n"
            f"  sample={_sample_debug(sample)}\n"
            f"  sampling_keys={sorted((sampling_params or {}).keys())}",
            file=sys.stderr,
            flush=True,
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise RuntimeError(
            "MegaGem Stage C custom_generate_function failed "
            f"cause_chain={_exc_chain_summary(exc)}; see preceding "
            "[megagem-stage-c] traceback for context."
        ) from exc


def _reset_stage_c_runtime_state() -> None:
    """Test helper: clear per-process rollout caches."""

    _GROUP_STATES.clear()
    _GENERATE_STATES.clear()
    _GAME_SEMAPHORES.clear()


async def megagem_precomputed_reward(args: Any, sample: Any, **_: Any) -> float:
    """Slime custom RM hook: expose the MegaGem row reward as scalar reward."""

    del args
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        megagem = metadata.get("megagem")
        if isinstance(megagem, dict) and megagem.get("precomputed_reward") is not None:
            return float(megagem["precomputed_reward"])
    reward = getattr(sample, "reward", 0.0)
    return float(reward)


def megagem_precomputed_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    """Slime custom advantage hook for MegaGem row-level credit assignment."""

    del args
    import torch

    kl = rollout_data["kl"]
    scalars = rollout_data.get("megagem_precomputed_advantage")
    if scalars is None:
        raise RuntimeError(
            "MegaGem Stage C needs rollout_data['megagem_precomputed_advantage']; "
            "ensure patch_megagem_rollout_data is registered in the Slime image."
        )
    if len(scalars) != len(kl):
        raise RuntimeError(
            "MegaGem precomputed advantage count mismatch: "
            f"{len(scalars)} advantages for {len(kl)} KL tensors"
        )
    returns = []
    bad = []
    for i, value in enumerate(scalars):
        scalar = float(value)
        if not math.isfinite(scalar):
            bad.append((i, value))
            scalar = 0.0
        returns.append(torch.ones_like(kl[i], dtype=torch.float32) * scalar)
    if bad:
        raise RuntimeError(f"MegaGem non-finite precomputed advantages: {bad[:3]}")
    rollout_data["advantages"] = [r for r in returns]
    rollout_data["returns"] = returns
    rollout_data.pop("megagem_precomputed_advantage", None)
    rollout_data.pop("megagem_precomputed_reward", None)
    rollout_data.pop("megagem_group_key", None)
