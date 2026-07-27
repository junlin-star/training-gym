# # Multi-turn agentic RL: Harbor benchmarks with terminus-2
#
# This is the agentic analogue of the gym's `002_multiturn` tutorial. Instead of
# a hand-written number-guessing loop, the multi-turn rollout *is* harbor's
# terminus-2 agent driving a real TerminalBench or SWE-bench task in a Modal
# container, and the reward is the selected benchmark's own verifier.
#
# It keeps the exact same gym contract as every RL example:
#   TrainConfig(model, dataset, recipe=SlimeRecipe(
#       custom_generate_function=..., custom_rm_function=...))
# Only the *bodies* of the two hooks change:
#   - custom_generate_function: run one terminus-2 trial against the gym's sglang
#     router, then re-tokenize the agent's text trajectory locally into a single
#     (tokens, loss_mask) training sequence (stock sglang doesn't return token ids
#     over the OpenAI API, so we encode the messages ourselves).
#   - custom_rm_function: read the verifier reward off sample.metadata.
#
# Measurement (baseline + final) stays harbor-native via `harbor run`, so the
# number we optimize is the number the customer sees.
#
# The policy is Qwen3.6-35B-A3B (MoE). The rollout and reward hooks below are
# model-agnostic; only the model spec, slime parallelism/memory layout, and
# served name live in the model profile.
#
# Run the full train+measure flow:
#   python 002_multiturn_terminus.py
#   python 002_multiturn_terminus.py --benchmark swebench --task-set shortlist

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from modal_training_gym import (
    DeploymentConfig,
    HarborDataset,
    Qwen3_6_35B,
    TrainConfig,
    WandbConfig,
    list_checkpoints,
)
from modal_training_gym.common.deployment import _modal_proxy_auth_headers
from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_35b_Recipe


# ## Configuration
#
# Both benchmarks are Harbor datasets containing many tasks. `HarborDataset`
# downloads the selected dataset and addresses individual tasks by name. Keep
# shortlists tiny: agentic rollouts require an environment build and a
# multi-minute episode per sample.


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_ref: str
    smoke_tasks: tuple[str, ...]

    @property
    def namespace(self) -> str:
        """Harbor registry namespace, also used for run/deployment naming."""
        return self.dataset_ref.partition("@")[0].partition("/")[0]


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "terminalbench": BenchmarkConfig(
        dataset_ref="terminal-bench/terminal-bench-2",
        smoke_tasks=(
            "adaptive-rejection-sampler",
            "openssl-selfsigned-cert",
            "path-tracing",
            "break-filter-js-from-html",
            "log-summary-date-ranges",
            "vulnerable-secret",
        ),
    ),
    "swebench": BenchmarkConfig(
        dataset_ref="swe-bench/swe-bench-verified",
        smoke_tasks=(
            "django__django-11099",
            "django__django-11283",
            "sympy__sympy-20590",
        ),
    ),
}

SERVED_MODEL_NAME = "qwen3-6-35b"

# Per-episode wall-clock cap for the *training* rollout. 
ROLLOUT_TIMEOUT_SEC = 900.0

# litellm addresses an OpenAI-compatible endpoint via the `hosted_vllm/<name>`
# prefix; terminus-2 needs model_info to know its context budget.
DEFAULT_MODEL_INFO: dict[str, Any] = {
    "max_input_tokens": 32768,
    "max_output_tokens": 32768,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
}

# Pin the harbor version installed into the rollout image so remote workers can
# import harbor and launch `--env modal` task sandboxes.
HARBOR_PIP_SPEC = "harbor[modal]"

# Splits Qwen3's <think> block out of the chat `content` into `reasoning_content`
# so terminus-2 parses clean JSON actions. Used by the training rollout engine
# (via `sglang_reasoning_parser`).
SGLANG_REASONING_PARSER = "qwen3"


# ## Multi-turn re-tokenization
# Stock sglang does not return token ids so re-tokenize locally with the policy 
# tokenizer.

def _training_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize messages while preserving parsed reasoning and tool-call fields."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        normalized_message = dict(message)
        normalized_message["role"] = str(message.get("role") or "")
        content = normalized_message.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        normalized_message["content"] = content
        normalized.append(normalized_message)
    return normalized


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        # Qwen's template normally drops reasoning from earlier turns. Preserve it
        # here because those tokens were generated actions and need policy loss.
        preserve_thinking=True,
    )
    # Qwen3.6's tokenizer returns a BatchEncoding even when ``return_dict`` is
    # not requested. Other HF tokenizers return the input-id list directly.
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get("input_ids")
    # Coerce tensor / ndarray outputs (e.g. a BatchEncoding holding tensors) to
    # nested python lists so the checks below operate on plain ints.
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    # Unwrap a leading batch dimension: some tokenizers (incl. qwen3-6-35b) return
    # ``[[id, id, ...]]`` for a single conversation rather than a flat ``[id, ...]``.
    if (
        isinstance(token_ids, list)
        and len(token_ids) == 1
        and isinstance(token_ids[0], list)
    ):
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or (
        token_ids and not isinstance(token_ids[0], int)
    ):
        raise TypeError("tokenizer.apply_chat_template() did not return token ids")
    return token_ids


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    """Return the number of identical leading token ids."""
    return next(
        (i for i, (left_id, right_id) in enumerate(zip(left, right)) if left_id != right_id),
        min(len(left), len(right)),
    )


def retokenize(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_sequence_len: int | None = None,
) -> tuple[list[int], list[int], int]:
    """Render a chat into ``(tokens, loss_mask, response_length)``.

    The prompt contains every message before the first assistant turn plus the
    assistant generation marker. The response is the rest of the chat-templated
    conversation. Only assistant-generated content (including its end marker) is
    trainable; subsequent user/tool observations and role markers are masked.

    ``max_sequence_len`` bounds the *whole* emitted sequence (prompt + response),
    not just the response region. Terminus trajectories interleave assistant
    turns with full terminal observations, so the rendered response can be long.

    slime consumes what we emit verbatim: in ``_convert_samples_to_train_data``
    it sets ``total_length = len(sample.tokens)`` and ``response_length =
    sample.response_length`` (asserting ``len(loss_mask) == response_length``),
    then ``get_batch`` does ``F.pad(loss_mask, (prompt_length - 1, 1))`` with
    ``prompt_length = total_length - response_length``. It never truncates
    ``tokens`` itself, so ``prompt_length`` is fully ours to keep positive: if the
    response region ever equalled the whole sequence there'd be no prompt and
    ``prompt_length - 1`` would go negative, crashing Megatron. We therefore
    reserve room for the prompt (plus a 1-token shift/pad margin) and tail-truncate
    the response to fit; if the prompt alone exceeds the budget we return an empty
    response so the caller aborts the sample. Bounding the total also keeps any
    single sample within slime's per-GPU packing budget (``max_tokens_per_gpu``).
    """
    normalized = _training_messages(messages)
    first_assistant = next(
        (i for i, message in enumerate(normalized) if message["role"] == "assistant"),
        None,
    )
    if first_assistant is None:
        return [], [], 0

    prompt_ids = _apply_chat_template(
        tokenizer,
        normalized[:first_assistant],
        add_generation_prompt=True,
    )
    full_ids = _apply_chat_template(
        tokenizer,
        normalized,
        add_generation_prompt=False,
    )
    prompt_prefix_length = _common_prefix_length(prompt_ids, full_ids)
    # Qwen's generation prompt ends in ``<think>\n`` while a completed
    # assistant turn continues with another newline. Its tokenizer merges those
    # two newlines into one token, so the independently tokenized prompt differs
    # at its final token. Treat that one boundary token as response-side but
    # masked; divergence before the final prompt token is a real template
    # mismatch.
    if not prompt_ids or prompt_prefix_length < len(prompt_ids) - 1:
        raise ValueError(
            "chat template prompt diverges from the full trajectory before its "
            "final boundary token"
        )

    prompt_ids = full_ids[:prompt_prefix_length]
    response_ids = full_ids[prompt_prefix_length:]
    loss_mask = [0] * len(response_ids)
    for i in range(first_assistant, len(normalized)):
        if normalized[i]["role"] != "assistant":
            continue
        before = _apply_chat_template(
            tokenizer,
            normalized[:i],
            add_generation_prompt=True,
        )
        after = _apply_chat_template(
            tokenizer,
            normalized[: i + 1],
            add_generation_prompt=False,
        )
        before_prefix_length = _common_prefix_length(before, full_ids)
        if before_prefix_length < len(before) - 1:
            raise ValueError(
                f"chat template diverges before assistant turn {i}'s final "
                "generation-prompt token"
            )
        if full_ids[: len(after)] != after:
            raise ValueError(
                f"chat template is not prefix-stable after assistant turn {i}"
            )

        # If the final generation-prompt token merged with the first generated
        # token, conservatively mask that mixed boundary token.
        start_in_full = before_prefix_length
        if before_prefix_length < len(before):
            start_in_full += 1
        start = start_in_full - len(prompt_ids)
        end = len(after) - len(prompt_ids)
        loss_mask[max(0, start) : max(0, end)] = [1] * max(0, end - start)

    if max_sequence_len is not None:
        room = max_sequence_len - len(prompt_ids) - 1
        if room <= 0:
            return prompt_ids, [], 0
        if len(response_ids) > room:
            response_ids = response_ids[:room]
            loss_mask = loss_mask[:room]

    tokens = prompt_ids + response_ids
    return tokens, loss_mask, len(response_ids)


def _filler_token_id(tokenizer: Any) -> int:
    """A safe single token id (pad -> eos -> 0) for padding degenerate sequences."""
    for tok in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if tok is not None:
            return int(tok)
    return 0


def abort_sample(sample: Any, tokenizer: Any, reason: str = "") -> Any:
    """Return *sample* as a structurally valid, zero-loss ABORTED sample.

    slime keeps ABORTED samples inside their ``n_samples_per_prompt`` GRPO group
    and still feeds them to Megatron's ``get_batch``, which does
    ``F.pad(loss_mask, (prompt_length - 1, 1))`` with ``prompt_length =
    len(tokens) - response_length``. A sample that never produced a trajectory has
    empty ``tokens`` / ``response_length == 0``, so ``prompt_length`` collapses to
    0 and Megatron crashes with "narrow(): length must be non-negative".

    Rather than leave the sample degenerate, emit a minimal valid sequence: a real
    prompt region (>= 1 token, so ``prompt_length >= 1``) plus a single response
    token whose ``loss_mask`` is 0. The sample keeps its slot in the group and its
    reward feeds GRPO advantage normalization, but contributes no policy gradient.
    """
    from slime.utils.types import Sample

    prompt_text = getattr(sample, "prompt", None) or ""
    prompt_ids: list[int] = (
        tokenizer(prompt_text, add_special_tokens=False)["input_ids"] if prompt_text else []
    )
    if not prompt_ids:
        # Guarantee a non-empty prompt region so prompt_length >= 1.
        prompt_ids = [_filler_token_id(tokenizer)]

    sample.tokens = prompt_ids + [_filler_token_id(tokenizer)]
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.response = ""
    sample.status = Sample.Status.ABORTED
    if getattr(sample, "reward", None) is None:
        sample.reward = 0.0
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["abort_reason"] = reason or "unspecified"
    sample.metadata = metadata
    if reason:
        print(f"[terminus_generate] abort ({reason})")
    return sample


# ## Resolve a task name to a local task directory
#
# The dataset is downloaded with `HarborDataset` and cached on whatever machine
# runs this — including the remote rollout worker on first use. We map a task
# name to its task directory so the trial can build the env from it.


def _resolve_task_dir(task_name: str, dataset_ref: str) -> str:
    dataset = HarborDataset(
        dataset_name=dataset_ref,
        task_names=[task_name],
    )
    # Reuse HarborDataset's internal pull + task discovery so we don't reimplement
    # the harbor download logic.
    task_root = dataset._resolve_task_root()  # noqa: SLF001 (intentional reuse)
    candidate = task_root / task_name
    if candidate.is_dir():
        return str(candidate)
    # Scan every task dir (unfiltered) so a misspelled/unknown task name yields a
    # helpful "available tasks" error. We can't reuse `dataset` here because its
    # task_names filter makes `_iter_task_dirs` raise the generic "No Harbor tasks
    # found" before we ever get to build a useful message.
    all_task_dirs = HarborDataset(
        dataset_name=dataset_ref
    )._iter_task_dirs()  # noqa: SLF001
    for path in all_task_dirs:
        if path.name == task_name:
            return str(path)
    all_names = sorted(path.name for path in all_task_dirs)
    available = "\n  ".join(all_names)
    raise FileNotFoundError(
        f"task {task_name!r} not found in {dataset_ref}. "
        f"Available ({len(all_names)} tasks):\n  {available}"
    )


# ## Run one terminus-2 trial (the rollout core)
#
# Build a harbor TrialConfig that points terminus-2 at an OpenAI-compatible
# endpoint with `store_all_messages=True`, run the task in a Modal container, and
# return the full chat message list and the verifier reward. We re-tokenize the
# messages locally with the policy chat template (see `retokenize`).


async def run_one_trial(
    *,
    task_name: str,
    dataset_ref: str,
    api_base: str,
    model_name: str,
    model_info: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    trials_dir: str | None = None,
    max_turns: int | None = None,
    timeout_sec: float | None = None,
) -> tuple[list[dict[str, Any]] | None, float]:
    """Run one selected-benchmark task with terminus-2; return messages and reward.

    ``messages`` is terminus-2's full chat history (a list of ``{"role",
    "content"}`` dicts) which we re-tokenize for training. ``extra_headers`` is
    forwarded to litellm (e.g. Modal-Key/Modal-Secret for an authenticated gym
    endpoint); leave it ``None`` for the unauthenticated in-cluster training
    router.
    """
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )
    from harbor.trial.trial import Trial

    task_dir = _resolve_task_dir(task_name, dataset_ref)
    cleanup_dir = None
    if trials_dir is None:
        namespace = dataset_ref.partition("@")[0].partition("/")[0]
        cleanup_dir = tempfile.mkdtemp(prefix=f"{namespace}-trials-")
        trials_dir = cleanup_dir

    agent_kwargs: dict[str, Any] = {
        "api_base": api_base,
        # We re-tokenize locally, so we don't need server token ids; we just need
        # terminus to hand back its full chat history in the trial result.
        "store_all_messages": True,
        "model_info": model_info or DEFAULT_MODEL_INFO,
    }
    if extra_headers:
        agent_kwargs["llm_kwargs"] = {"extra_headers": extra_headers}
    if max_turns is not None:
        agent_kwargs["max_turns"] = max_turns

    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=trials_dir,
        agent=AgentConfig(
            name="terminus-2",
            model_name=f"hosted_vllm/{model_name}",
            kwargs=agent_kwargs,
            # Cap wall-clock so a floundering base model can't burn the full
            # task default (e.g. 900s) before we get a signal.
            override_timeout_sec=timeout_sec,
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.MODAL,
            kwargs={
                "sandbox_idle_timeout_secs": int(ROLLOUT_TIMEOUT_SEC),
                "sandbox_timeout_secs": int(ROLLOUT_TIMEOUT_SEC) + 300,
            },
            delete=True,                            # tear down resources on stop
        ),
        # environment=EnvironmentConfig(type=EnvironmentType.MODAL),
    )

    trial = await Trial.create(config)
    try:
        result = await trial.run()

        print(f"[run_one_trial] trial dir: {trials_dir}")

        # Surface the failure reason harbor records when the agent loop dies — without
        # this, a crashed trial just looks like "0 turns / reward 0".
        if result.exception_info is not None:
            ei = result.exception_info
            print(f"[run_one_trial] trial exception: {ei.exception_type}: {ei.exception_message}")
            tb_tail = "\n".join(ei.exception_traceback.strip().splitlines()[-15:])
            print(f"[run_one_trial] traceback tail:\n{tb_tail}")

        # Token counts tell us whether the LLM was actually called: if these are 0/None
        # the agent never reached a successful completion (auth / connection / setup).
        n_in, _, n_out, _ = result.compute_token_cost_totals()
        print(f"[run_one_trial] token usage: input={n_in} output={n_out}")

        messages: list[dict[str, Any]] | None = None
        if result.agent_result is not None and result.agent_result.metadata:
            messages = result.agent_result.metadata.get("all_messages")
        if messages:
            n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
            # With --sglang-reasoning-parser qwen3, assistant turns should carry a
            # separate `reasoning_content`. If this count is 0 while the parser is on,
            # terminus dropped it from all_messages and option-A retokenization
            # silently degrades to JSON-only (no reasoning trained) — worth knowing.
            n_reasoning = sum(
                1
                for m in messages
                if m.get("role") == "assistant" and str(m.get("reasoning_content") or "").strip()
            )
            print(
                f"[run_one_trial] messages: {len(messages)} total, {n_assistant} assistant "
                f"turn(s), {n_reasoning} with reasoning_content"
            )
        else:
            print("[run_one_trial] no all_messages on agent_result (store_all_messages?)")

        reward = 0.0
        if result.verifier_result is not None and result.verifier_result.rewards:
            rewards = result.verifier_result.rewards
            # Harbor benchmark verifiers conventionally write a single "reward"
            # key; fall back to the first value if an adapter uses another name.
            reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0)))

        return messages, reward
    finally:
        # Every trial writes a trials_dir (trajectory JSON, asciinema recording,
        # per-call litellm logs). We re-tokenize from the returned messages and
        # never read the dir again, so drop the temp dir we created — otherwise
        # tens of thousands of trials over a run accumulate in /tmp and fill the
        # rollout container's disk, crashing it mid-rollout with
        # "OSError: [Errno 28] No space left on device" (which loses all work
        # since the last checkpoint on the retry).
        if cleanup_dir is not None:
            import shutil

            shutil.rmtree(cleanup_dir, ignore_errors=True)


# ## The two slime hooks
#
# These run on the rollout workers. `args.sglang_router_ip/port` is the live
# policy server; terminus-2 talks to its OpenAI-compatible `/v1` endpoint.


def _sample_label(sample: Any) -> dict[str, Any]:
    raw = getattr(sample, "label", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


async def terminus_generate(args: Any, sample: Any, sampling_params: dict) -> Any:
    """Multi-turn rollout: run a Terminus trial for one selected benchmark task."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.types import Sample

    # Fetch the tokenizer up front so every abort path below can emit a valid,
    # zero-loss sample via abort_sample() -- slime trains on ABORTED samples that
    # stay in the group, so a degenerate one crashes Megatron (see abort_sample).
    tokenizer = GenerateState(args).tokenizer

    label = _sample_label(sample)
    task_name = label.get("harbor_task_name") or label.get("task_name")
    if not task_name:
        return abort_sample(sample, tokenizer, reason="no task_name")
    dataset_ref = getattr(args, "harbor_dataset_ref", None)
    if not dataset_ref:
        return abort_sample(sample, tokenizer, reason="no harbor_dataset_ref")

    api_base = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"
    served_model_name = getattr(args, "served_model_name", None) or getattr(
        args, "hf_checkpoint", None
    ) or getattr(args, "model_name", "policy")

    messages, reward = await run_one_trial(
        task_name=task_name,
        dataset_ref=dataset_ref,
        api_base=api_base,
        model_name=served_model_name,
        timeout_sec=ROLLOUT_TIMEOUT_SEC,
    )

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["reward"] = reward
    metadata["harbor_benchmark"] = {
        "dataset": dataset_ref,
        "task_name": task_name,
    }
    sample.metadata = metadata
    # Set the reward on the sample directly (not just metadata) so it survives the
    # ABORTED return paths below, which bypass `terminus_rm`. Without this, slime's
    # eval logging hits `sum(rewards)` over a None and raises
    # `TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'`.
    sample.reward = reward

    if not messages:
        # Agent produced no usable trajectory (crash / context overflow). Skip it.
        return abort_sample(sample, tokenizer, reason=f"no trajectory for {task_name}")

    # Cap the whole training sequence (prompt + response) to the *smaller* of
    # ``rollout_max_response_len`` and ``max_tokens_per_gpu``. slime packs multiple
    # samples per GPU up to ``max_tokens_per_gpu`` (first-fit over ``total_length``),
    # so a single sample longer than that budget can't be scheduled. It does NOT
    # silently truncate our ``tokens`` -- ``prompt_length = len(tokens) -
    # response_length`` is entirely ours to keep >= 1 -- but bounding total length
    # here keeps every sample packable and leaves ``retokenize`` room to reserve the
    # prompt. To train on genuinely longer sequences, raise ``max_tokens_per_gpu``
    # (memory permitting), not just ``rollout_max_response_len``.
    budget_candidates = [
        b
        for b in (
            getattr(args, "rollout_max_response_len", None),
            getattr(args, "max_tokens_per_gpu", None),
        )
        if b is not None
    ]
    max_sequence_len = min(budget_candidates) if budget_candidates else None
    try:
        tokens, loss_mask, response_length = retokenize(
            tokenizer,
            messages,
            max_sequence_len=max_sequence_len,
        )
    except (TypeError, ValueError) as exc:
        return abort_sample(
            sample,
            tokenizer,
            reason=f"chat-template mismatch for {task_name}: {exc}",
        )

    if response_length == 0:
        # Empty response region (e.g. assistant turns with no content after the
        # reasoning parser split it out and reasoning_content wasn't preserved).
        # A zero-length response yields a degenerate training sequence, so skip it.
        return abort_sample(
            sample, tokenizer, reason=f"empty response region for {task_name}"
        )

    # Guard against a degenerate sequence reaching Megatron: slime's get_batch
    # does F.pad(loss_mask, (prompt_length - 1, 1)) with
    # prompt_length = len(tokens) - response_length, which raises
    # "narrow(): length must be non-negative" when prompt_length < 1. retokenize
    # reserves prompt room, but keep a belt-and-suspenders check here.
    prompt_length = len(tokens) - response_length
    if prompt_length < 1:
        return abort_sample(
            sample, tokenizer, reason=f"non-positive prompt_length for {task_name}"
        )

    sample.tokens = tokens
    sample.response_length = response_length
    sample.loss_mask = loss_mask
    sample.response = tokenizer.decode(tokens[-response_length:], skip_special_tokens=False)
    sample.status = Sample.Status.COMPLETED
    return sample


async def terminus_rm(args: Any, sample: Any, **kwargs: Any) -> float:
    """Return the Harbor verifier reward stashed by terminus_generate."""
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        return float(metadata.get("reward", 0.0))
    return 0.0


# ## Dataset
#
# Reuse `HarborDataset` directly: its rows carry `harbor_task_name` in the label,
# which the rollout uses to launch the right task. The prompt text is unused by
# the rollout (terminus-2 builds its own), but slime still needs a dataset to
# sample task instances from.


def build_dataset(
    benchmark: BenchmarkConfig, task_names: list[str] | None
) -> HarborDataset:
    return HarborDataset(
        dataset_name=benchmark.dataset_ref,
        label_metadata_path="task.toml",
        task_names=task_names,
        train_repeats=8,
        always_prepare=True,
    )


# ## Baseline / final measurement (harbor-native)
#
# Headline metric uses the real `harbor run` against a deployed endpoint, so it's
# directly comparable to what the customer runs. This mirrors gym_v1.py.


def run_harbor_benchmark(
    deployment: Any,
    benchmark: BenchmarkConfig,
    task_names: list[str] | None,
    *,
    n: int = 16,
    concurrency: int = 16,
) -> None:
    import shutil
    import subprocess

    served_model_name = deployment.deployment_config.served_model_name
    api_base = f"{deployment.url}/v1"
    # Scope the headline measurement to the same task subset we train on, so the
    # baseline/final before-after is computed over exactly the trained tasks
    # (`-i` includes a task by name; harbor applies it before the `-l` cap).
    # Harbor's filter matches the registry-namespaced task name, not the bare
    # task directory name returned by HarborDataset.
    include_args: list[str] = []
    for task_name in task_names or []:
        include_args += ["-i", f"{benchmark.namespace}/{task_name}"]
    harbor_args = [
        "run",
        "-d", benchmark.dataset_ref,
        *include_args,
        "-a", "terminus-2",
        "-m", f"hosted_vllm/{served_model_name}",
        "-n", str(concurrency), "-l", str(n),
        "--ak", f"api_base={api_base}",
        "--ak", f"model_info={json.dumps(DEFAULT_MODEL_INFO)}",
        "--env", "modal",
        "-y",
    ]
    # Harbor calls the endpoint directly, so pass the same proxy-auth headers
    # used by ModelDeployment.generate() and wait_until_ready().
    extra_headers = _modal_proxy_auth_headers()
    if extra_headers:
        harbor_args[-1:-1] = ["--ak", f"llm_kwargs={json.dumps({'extra_headers': extra_headers})}"]
    harbor_bin = shutil.which("harbor")
    cmd = [harbor_bin, *harbor_args] if harbor_bin else ["uvx", "harbor", *harbor_args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ## Full flow: deploy -> measure -> train -> measure


def train(
    benchmark: BenchmarkConfig,
    task_names: list[str] | None,
    rollout_steps: int = 50,
) -> None:
    dataset = build_dataset(benchmark, task_names)

    # The recipe bakes in `served_model_name` (threaded to the rollout so
    # terminus-2 addresses the right model on the sglang router) and the
    # model-specific slime parallelism/memory layout. Modal creds + harbor must
    # be present on the rollout workers so they can launch `--env modal` sandboxes.
    training_run = TrainConfig(
        model=Qwen3_6_35B(),
        dataset=dataset,
        recipe=Qwen3_6_35b_Recipe(
            custom_generate_function=terminus_generate,
            custom_rm_function=terminus_rm,
            app_tags={
                "dataset_and_num_tasks": (
                    f"{benchmark.dataset_ref.replace('/', '_').replace('@', '_')}"
                    f"__n{len(task_names) if task_names else -1}"
                ),
            },
            wandb=WandbConfig(
                project=f"{benchmark.namespace}-rl",
                group="qwen3-6-35b-terminus",
                exp_name="terminus-smoke",
            ),
            extra_config={
                "served_model_name": SERVED_MODEL_NAME,
                "harbor_dataset_ref": benchmark.dataset_ref,
                "log_multi_turn": False, # ~20MB per rollout step
            },
            sglang_reasoning_parser=SGLANG_REASONING_PARSER,
            colocate=False,
            actor_num_nodes=1,
            rollout_num_gpus=32,
            async_mode=False,
            num_rollout=rollout_steps,
            rollout_batch_size=32,
            n_samples_per_prompt=16,
            rollout_max_response_len=32768,
            max_tokens_per_gpu=32768,
            global_batch_size=512,
            lr=5e-7,
            # Increases rollout time by ~3x (15m->55m)
            # over_sampling_batch_size=64, 
            # dynamic_sampling_filter_path=(
                #"slime.rollout.filter_hub.dynamic_sampling_filters."
                #"check_reward_nonzero_std"
            #),
            use_kl_loss=False,
            save_interval=5,
            no_save_optim=False,
            eval_interval=5,
            n_samples_per_eval_prompt=4,
            eval_max_response_len=32768,
            image_overlay=lambda image: image.run_commands(
                f"uv pip install --system 'modal>=1.2.0' '{HARBOR_PIP_SPEC}'",
            ),
        ),
    )
    print(f"Starting training ({SERVED_MODEL_NAME})...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")

    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    trained_deployment = DeploymentConfig(
        model=Qwen3_6_35B(),
        checkpoint=checkpoint,
        app_name=f"{SERVED_MODEL_NAME}-{benchmark.namespace}-serve",
        served_model_name=f"{SERVED_MODEL_NAME}-{benchmark.namespace}",
    ).serve()
    trained_deployment.wait_until_ready()
    print(f"Trained model URL: {trained_deployment.url}")

    print(f"Measuring trained model on {benchmark.dataset_ref} (harbor run)...")
    run_harbor_benchmark(trained_deployment, benchmark, task_names)
    print("Compare the two harbor run scores above for the RL gain.")


def _add_benchmark_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARKS),
        default="swebench",
        help=(
            "Harbor benchmark to train and evaluate on. 'swebench' selects "
            "SWE-bench Verified (default: %(default)s)"
        ),
    )


def _add_task_set_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--task-set",
        choices=("all", "shortlist"),
        default="all",
        help=(
            "Which tasks to use: 'all' is the whole selected benchmark; "
            "'shortlist' uses its smoke subset for fast iteration "
            "(default: %(default)s)"
        ),
    )


def _add_rollout_steps_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=100,
        help=(
            "Number of rollout steps to train for (slime's num_rollout) "
            "(default: %(default)s)"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_benchmark_arg(parser)
    _add_task_set_arg(parser)
    _add_rollout_steps_arg(parser)
    args = parser.parse_args()
    benchmark = BENCHMARKS[args.benchmark]
    task_names = (
        list(benchmark.smoke_tasks) if args.task_set == "shortlist" else None
    )
    print(
        f"Selected benchmark: {benchmark.dataset_ref}; task set: {args.task_set}; "
        f"rollout steps: {args.rollout_steps}"
    )
    train(benchmark, task_names, rollout_steps=args.rollout_steps)


if __name__ == "__main__":
    main()
