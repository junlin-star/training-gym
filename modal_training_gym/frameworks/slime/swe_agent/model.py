"""mini-swe-agent Model that calls sglang directly and records exact tokens — the model-API
seam. Keeps ONE running token sequence: each turn appends the new context (loss_mask 0) and the
exact generated ids (loss_mask 1), so the trajectory is token-in-token-out faithful (one training
Sample, no re-tokenization). One instance per episode; read tokens/loss_mask/logprobs/versions after.
"""

import json
import random
import re
import time
import urllib.error
import urllib.request

from minisweagent.exceptions import FormatError, InterruptAgentFlow

from modal_training_gym.common.agents.mini_swe import BASH_TOOL

# Qwen3.6 native tool-call is the qwen3_xml format (NOT JSON), e.g.:
#   <tool_call>
#   <function=bash>
#   <parameter=command>
#   ls -la
#   </parameter>
#   </function>
#   </tool_call>
# Count <tool_call> blocks, then pull the function name and the command parameter from inside.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_CMD_RE = re.compile(
    r"<parameter\s*=\s*command\s*>\s*(.*?)\s*</parameter>", re.DOTALL
)

# Per-turn generation budget against the served context window.
_CONTEXT_MARGIN = 256  # headroom for stop/special tokens
_MIN_GEN_TOKENS = 2048  # below this much room left, end the episode rather than spiral on truncated actions


class ContextExceeded(InterruptAgentFlow):
    """The growing trajectory left no room to generate."""


class RolloutAborted(InterruptAgentFlow):
    """The rollout was aborted (the step's batch is already full) — end this surplus episode at the turn
    boundary instead of running to completion. Engine-side aborts only kill in-flight generations; an
    episode that was mid-bash never sees one, so without this check it churns for up to a full episode
    while the sync drain waits for it. generate() sees model.aborted and recycles the sample."""


class RecordingModel:
    config = None  # mini-swe Model protocol

    def __init__(
        self,
        tokenizer,
        sampling_params,
        router_url,
        observation_template,
        session_id,
        query_timeout=600,
        max_context_len=131072,
        abort_check=None,
    ):
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self._abort_check = (
            abort_check  # per-turn rollout-abort probe (see RolloutAborted)
        )
        self.url = f"{router_url}/generate"
        self.query_timeout = (
            query_timeout  # cap per-turn sglang call; bounds hung/queued generations
        )
        self.max_context_len = (
            max_context_len  # served window; per-turn gen is capped to the remainder
        )
        self.observation_template = observation_template
        # consistent-hashing routing key → every turn hits the same worker (prefix cache across turns)
        self.headers = {
            "Content-Type": "application/json",
            "X-SMG-Routing-Key": session_id,
        }
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.versions: list[str] = []
        self.prompt_len: int | None = None
        self.aborted = False
        self.gen_time = 0.0
        self.cached_tokens = (
            0  # radix-cache hits, summed over turns (prefix_cache_hit_rate)
        )
        self.input_tokens = (
            0  # prompt tokens sent, summed over turns (hit-rate denominator)
        )
        self.n_calls = 0  # total query() calls incl. format-error retries (vs len(versions) = productive turns)
        self.n_format_errors = 0
        self.n_length_truncations = (
            0  # turns the model overran the per-turn cap (finish_reason=length)
        )
        self.reasoning_tokens = (
            0  # tokens spent inside <think>…</think>, summed over turns
        )
        self._consumed = ""  # rendered text of the conversation already in `tokens`
        self._think_kwargs = {"preserve_thinking": True}
        stop_ids = [
            tokenizer.convert_tokens_to_ids(token)
            for token in ("<|im_end|>", "<|endoftext|>")
        ]
        self._stop = {
            "stop_token_ids": [
                token_id
                for token_id in stop_ids
                if isinstance(token_id, int) and token_id >= 0
            ]
        }
        self._tools = [BASH_TOOL]

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        # Keep each turn's reasoning in-context so the render stays a stable prefix across turns — required by
        # the Sample→Sample append below. The kwarg is model-specific (detected in __init__); without it the
        # past <think> blocks get stripped on re-render, making the render non-monotonic and desyncing recording.
        # tools= renders a stable bash schema into the system section.
        return self.tokenizer.apply_chat_template(
            clean,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            tools=self._tools,
            **self._think_kwargs,
        )

    def _extend(self, text: str, mask: int) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.tokens += ids
        self.loss_mask += [mask] * len(ids)
        self.logprobs += [0.0] * len(ids)
        return len(ids)

    def _tool_call_format_error(
        self, n_found: int, segment: str, msg: str, finish: str
    ) -> FormatError:
        """Build a mini-swe FormatError (dict payload, matching parse_regex_actions) for a bad tool call —
        DefaultAgent appends `content` as the next user turn and counts it toward max_consecutive_format_errors."""
        if finish == "length":
            msg = "your response hit the output-token limit before a complete `bash` tool call"
        content = f"Format error: {msg}.\n\nProvide a THOUGHT, then make EXACTLY ONE call to the `bash` tool with a single command. To finish the task, call `bash` with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`."
        return FormatError(
            {
                "role": "user",
                "content": content,
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": n_found,
                    "model_response": segment,
                },
            }
        )

    def _parse_actions(self, segment: str, finish: str) -> list[dict]:
        """Parse one Qwen native bash tool call."""
        calls = _TOOL_CALL_RE.findall(segment)
        if len(calls) != 1:
            raise self._tool_call_format_error(
                len(calls),
                segment,
                f"expected exactly one tool call, found {len(calls)}",
                finish,
            )
        fn = _FUNCTION_RE.search(calls[0])
        if not fn:
            raise self._tool_call_format_error(
                1, segment, "tool call had no <function=...> block", finish
            )
        if fn.group(1) != "bash":
            raise self._tool_call_format_error(
                1,
                segment,
                f"called tool {fn.group(1)!r}; only 'bash' is available",
                finish,
            )
        cmd_match = _PARAM_CMD_RE.search(fn.group(2))
        cmd = cmd_match.group(1) if cmd_match else None
        if not isinstance(cmd, str) or not cmd.strip():
            raise self._tool_call_format_error(
                1,
                segment,
                "the bash tool call had no <parameter=command> value",
                finish,
            )
        return [{"command": cmd}]

    def query(self, messages: list[dict], **kwargs) -> dict:
        if self._abort_check is not None and self._abort_check():
            self.aborted = True
            raise RolloutAborted(
                {
                    "role": "exit",
                    "content": "RolloutAborted",
                    "extra": {"exit_status": "RolloutAborted", "submission": ""},
                }
            )
        self.n_calls += 1
        full = self._render(messages, add_generation_prompt=True)
        # The render must extend the prior verbatim or exact-token recording
        # would lose provenance.
        assert full.startswith(self._consumed), (
            "chat-template render not prefix-stable — token recording would desync"
        )
        n_ctx = self._extend(
            full[len(self._consumed) :], mask=0
        )  # new context since last turn, masked out
        if self.prompt_len is None:
            self.prompt_len = len(self.tokens)

        # Cap this turn's generation to the room left in the served window; if too little remains to
        # produce a useful turn, end the episode cleanly rather than spiral on truncated actions.
        remaining = self.max_context_len - len(self.tokens) - _CONTEXT_MARGIN
        if remaining < _MIN_GEN_TOKENS:
            raise ContextExceeded(
                {
                    "role": "exit",
                    "content": "ContextExceeded",
                    "extra": {"exit_status": "ContextExceeded", "submission": ""},
                }
            )
        sp = {
            **self.sampling_params,
            **self._stop,
            "max_new_tokens": min(
                self.sampling_params.get("max_new_tokens", remaining), remaining
            ),
        }

        n_in = len(self.tokens)
        payload = {
            "input_ids": self.tokens,
            "sampling_params": sp,
            "return_logprob": True,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=self.headers
        )
        t0 = time.perf_counter()
        # Retry transient overload — any connection error, or a 5xx (engine/router load-shed, worst during
        # the cold-start concurrency ramp). Jittered backoff disperses the thundering herd so retries don't
        # re-synchronize into another burst. HTTPError is a URLError subclass, so one handler covers both.
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.query_timeout) as resp:
                    out = json.loads(resp.read())
                break
            except urllib.error.URLError as e:
                retriable = not isinstance(e, urllib.error.HTTPError) or e.code >= 500
                if retriable and attempt < 4:
                    time.sleep(random.uniform(1.0, 2.0 ** (attempt + 1)))
                    continue
                raise
        self.gen_time += time.perf_counter() - t0
        meta = out["meta_info"]
        self.cached_tokens += meta.get("cached_tokens", 0)
        self.input_tokens += n_in
        token_logprobs = meta.get("output_token_logprobs") or []
        text = out.get("text", "")
        finish = meta.get("finish_reason", {}).get("type", "stop")
        self.tokens += [t[1] for t in token_logprobs]  # exact generated ids (trained)
        self.loss_mask += [1] * len(token_logprobs)
        self.logprobs += [t[0] for t in token_logprobs]
        self.versions.append(meta.get("weight_version"))
        if finish == "abort":
            # Engine aborted this generation (weight update / engine restart) — the turn is truncated garbage
            # and the sample will be recycled, so end the episode NOW rather than burn the remaining budget.
            self.aborted = True
            raise RolloutAborted(
                {
                    "role": "exit",
                    "content": "RolloutAborted",
                    "extra": {"exit_status": "RolloutAborted", "submission": ""},
                }
            )
        if finish == "length":
            self.n_length_truncations += 1
        if "</think>" in text:  # reasoning the policy spent before its answer
            self.reasoning_tokens += len(
                self.tokenizer.encode(
                    text.split("</think>")[0], add_special_tokens=False
                )
            )

        # Parse the action from the post-reasoning segment only — a tool call inside <think> is not an action.
        try:
            actions = self._parse_actions(text.split("</think>")[-1], finish)
        except FormatError:
            # mini-swe drops the malformed turn; drop BOTH this turn's generated ids AND the context
            # tokens appended above so `tokens` stays a faithful prefix of the next render (and
            # `_consumed`, left untouched, still matches `tokens`). The error feedback re-adds the context.
            drop = len(token_logprobs) + n_ctx
            if drop:
                del self.tokens[-drop:], self.loss_mask[-drop:], self.logprobs[-drop:]
            self.versions.pop()
            self.n_format_errors += 1
            raise
        self._consumed = self._render(
            messages + [{"role": "assistant", "content": text}],
            add_generation_prompt=False,
        )
        return {
            "role": "assistant",
            "content": text,
            "extra": {"actions": actions, "cost": 0.0},
        }

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ):
        from minisweagent.models.utils.actions_text import format_observation_messages

        return format_observation_messages(
            outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict:
        return {}

    def serialize(self) -> dict:
        return {}
