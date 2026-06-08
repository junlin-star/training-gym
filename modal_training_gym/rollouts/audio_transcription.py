"""Custom slime rollout that drives an ASR model via /v1/audio/transcriptions.

slime's default rollout posts chat-formatted prompts to /v1/chat/completions.
That works for Qwen-Omni (chat model with audio content blocks), but Qwen3-ASR
is served by SGLang as a Whisper-style model on a *different* endpoint:

    POST /v1/audio/transcriptions
      file=<wav bytes>,  model=<served>,  temperature=<T>,  prompt=<optional>

This module plugs into slime via `SlimeRecipe.custom_generate_function`:
the launcher cloudpickles the callable, slime calls it once per sample, we
POST the audio, set sample.response to the transcript, and return.

Wire-in:
    SlimeRecipe(
        custom_generate_function=transcription_rollout,
        custom_rm_function=wer_reward,
        ...
    )

slime calls this `n_samples_per_prompt` times per prompt (sampling stochastically
at sampling_params["temperature"]), giving GRPO its N samples to score.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any


# ── audio extraction (defensive — slime's Sample carries multimodal a few ways) ─


def _data_uri_to_bytes(data_uri: str) -> bytes:
    """Strip a `data:audio/<fmt>;base64,...` prefix and decode."""
    if data_uri.startswith("data:"):
        _, _, b64 = data_uri.partition(",")
    else:
        b64 = data_uri
    return base64.b64decode(b64)


def _coerce_audio(val: Any) -> bytes | None:
    """Coerce one audio reference (bytes or data-URI/base64 str) to bytes."""
    first = val[0] if isinstance(val, (list, tuple)) and val else val
    if isinstance(first, (bytes, bytearray)):
        return bytes(first)
    if isinstance(first, str) and first:
        return _data_uri_to_bytes(first)
    return None


def _extract_audio_bytes(sample: Any) -> bytes:
    """Pull raw audio bytes out of a slime Sample, regardless of how it's carried.

    slime's data pipeline puts audio in the *conversation-list* ``sample.prompt``
    as ``{"type": "audio", "audio": <data-uri>}`` (its ``process_vision_info`` only
    extracts images/videos, so ``multimodal_inputs`` carries no audio — hence we
    run with apply_chat_template=False to keep the prompt a message list). We also
    check a few other historical spots defensively.
    """
    # 1) conversation-list prompt: [{"role":..,"content":[{"type":"audio","audio":uri}, ...]}]
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, list):
        for msg in prompt:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and (
                        item.get("type") == "audio" or "audio" in item
                    ):
                        b = _coerce_audio(item.get("audio") or item.get("audio_url"))
                        if b:
                            return b

    # 2) multimodal_inputs / multi_modal_data dicts
    for attr in ("multimodal_inputs", "multi_modal_data", "multimodal_data"):
        mm = getattr(sample, attr, None)
        if isinstance(mm, dict):
            b = _coerce_audio(mm.get("audio") or mm.get("audios"))
            if b:
                return b

    # 3) raw dataset row, if exposed
    row = getattr(sample, "row", None) or getattr(sample, "raw", None) or {}
    if isinstance(row, dict):
        for key in ("audios", "audio", "audio_b64"):
            b = _coerce_audio(row.get(key))
            if b:
                return b

    raise RuntimeError(
        "transcription_rollout: could not find audio data on the slime Sample. "
        "Checked prompt message-list, multimodal_inputs/multi_modal_data, and row."
    )


# ── SGLang URL (defensive — slime exposes the rollout endpoint a few ways) ────


def _router_base(args: Any) -> str | None:
    ip = getattr(args, "sglang_router_ip", None)
    port = getattr(args, "sglang_router_port", None)
    if not ip:
        return None
    ip = str(ip)
    host = ip if ip.startswith("http") else f"http://{ip}"
    return f"{host}:{port}" if port else host


def _candidate_engine_bases(args: Any) -> list[str]:
    """SGLang *engine* base URLs that expose /v1/audio/transcriptions.

    The slime sglang_router (sglang_router_ip:port) does NOT proxy the
    transcription endpoint (404s), so we talk to the engines directly. Engines
    bind to the Ray node IP at base_port 15000, 15002, ... (one per engine).
    """
    import socket

    bases: list[str] = []
    # explicit overrides, if a future slime exposes them
    for attr in ("sglang_url", "rollout_url", "sglang_server_url"):
        v = getattr(args, attr, None)
        if v:
            v = str(v)
            bases.append(v if v.startswith("http") else f"http://{v}")

    base_port = int(getattr(args, "sglang_server_port", None) or 15000)
    n_engines = int(
        getattr(args, "rollout_num_gpus", 0)
        or getattr(args, "actor_num_gpus_per_node", 1)
        or 1
    )
    ports = [base_port + 2 * i for i in range(max(n_engines, 1) + 1)]

    hosts = []
    # The engines bind to the *Ray node IP* (not localhost), so this is the one
    # that actually works from the colocated rollout process.
    try:
        import ray

        ip = ray.util.get_node_ip_address()
        if ip:
            hosts.append(ip)
    except Exception:
        pass
    try:
        hosts.append(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass
    hosts.append("127.0.0.1")

    for h in hosts:
        for p in ports:
            bases.append(f"http://{h}:{p}")
    # de-dup, preserve order
    seen, out = set(), []
    for b in bases:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


# ── the rollout itself ─────────────────────────────────────────────────────────

# Engine base URLs for the WHOLE cluster (every node), discovered once. The single
# central RolloutManager must reach engines on all nodes, so we enumerate them via the
# router's worker registry rather than probing only the local node. Round-robin across
# them so load spreads across engines/nodes.
_ENGINE_BASES: list[str] | None = None
_RR_INDEX = 0


async def _discover_engine_bases(args: Any, sess) -> list[str]:
    """Every SGLang engine base URL across the cluster (all nodes).

    slime's router knows every engine — each self-registers its own-node URL at
    startup — so we read the registry via ``GET {router}/workers`` (the same endpoint
    slime uses to fan out aborts). This spans all nodes, which a local-node IP probe
    cannot (it would miss a second node's engines entirely). The router can't *proxy*
    ``/v1/audio/transcriptions`` (fixed-route Rust fork), so we still POST audio
    directly to each engine — we just use the router to discover them. Falls back to
    local-node probing when the registry is unavailable (single node / no router).
    """
    global _ENGINE_BASES
    if _ENGINE_BASES:
        return _ENGINE_BASES
    import aiohttp

    router = _router_base(args)
    if router:
        # /workers (sglang_router > 0.2.1) → {"workers": [{"url": ...}]};
        # /list_workers (older) → {"urls": [...]}.
        for path, extract in (
            (
                "/workers",
                lambda b: [w["url"] for w in b.get("workers", []) if w.get("url")],
            ),
            ("/list_workers", lambda b: list(b.get("urls", []))),
        ):
            try:
                async with sess.get(
                    router + path, timeout=aiohttp.ClientTimeout(total=15)
                ) as r:
                    if r.status != 200:
                        continue
                    urls = extract(await r.json())
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError):
                continue
            bases = [u if u.startswith("http") else f"http://{u}" for u in urls]
            if bases:
                _ENGINE_BASES = bases
                return _ENGINE_BASES
    # single node / router unavailable: probe the local node's engines.
    return _candidate_engine_bases(args)


_TOKENIZER = None


def _get_tokenizer(args):
    """Cached tokenizer (with Qwen3-ASR chat template injected by our slime patch)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        hf = getattr(args, "hf_checkpoint", None) or getattr(args, "model_name", None)
        try:
            from slime.utils.processing_utils import load_tokenizer

            _TOKENIZER = load_tokenizer(hf, trust_remote_code=True)
        except Exception:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    return _TOKENIZER


_PROCESSOR = None


def _get_processor(args):
    """Cached Qwen3-ASR processor (WhisperFeatureExtractor + tokenizer) — produces
    mel input_features AND the prompt with <audio_pad> expanded to the audio
    encoder's output length, so audio embeds align with token positions."""
    global _PROCESSOR
    if _PROCESSOR is None:
        hf = getattr(args, "hf_checkpoint", None) or getattr(args, "model_name", None)
        from sglang.srt.configs.qwen3_asr import Qwen3ASRProcessor

        _PROCESSOR = Qwen3ASRProcessor.from_pretrained(hf)
    return _PROCESSOR


# Qwen3-ASR audio placeholder: the processor expands the single <|audio_pad|> to N
# tokens (N = audio-encoder output length for the clip). Must appear in the prompt
# TEXT — and the raw audio data-URI must NOT, or it tokenizes into ~300K-900K text
# tokens (base64 length scales with clip duration) and the actor OOMs.
_AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


def _render_asr_prompt_text(prompt, tok) -> str:
    """Render the prompt to text with exactly one audio placeholder and no audio
    payload. apply_chat_template on the raw message list inlines the base64
    data-URI as literal text (n_audio_pad=0, ~10^5-10^6 tokens) — so we strip audio
    items / '<audio>' markers and inject the placeholder ourselves."""
    if not isinstance(prompt, list):
        txt = str(prompt).replace("<audio>", "").strip()
        return _AUDIO_PLACEHOLDER + "\n" + txt

    msgs = []
    for msg in prompt:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")
        parts = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    # drop audio/image items — never render their data payloads
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(content, str):
            parts.append(content)
        txt = " ".join(p for p in parts if p).replace("<audio>", "").strip()
        msgs.append({"role": role, "content": txt})

    # prepend the audio placeholder to the first user turn
    for m in msgs:
        if m["role"] == "user":
            m["content"] = (_AUDIO_PLACEHOLDER + "\n" + m["content"]).strip()
            break
    else:
        msgs.insert(0, {"role": "user", "content": _AUDIO_PLACEHOLDER})

    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # fallback: minimal Qwen chat format
        user = next(
            (m["content"] for m in msgs if m["role"] == "user"), _AUDIO_PLACEHOLDER
        )
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def _populate_tokens(args, sample, audio_bytes: bytes) -> None:
    """Set the token-level + multimodal fields the Megatron trainer needs:
    - sample.tokens: processor-tokenized prompt (with N expanded <audio_pad>) + response
    - sample.response_length / loss_mask: response only
    - sample.multimodal_train_inputs: {input_features, feature_attention_mask} so the
      actor's frozen audio tower can produce N embeds that scatter onto the N audio
      tokens → the actor's log-probs are AUDIO-CONDITIONED (valid GRPO)."""
    import io

    import soundfile as sf

    proc = _get_processor(args)
    tok = proc.tokenizer

    text = _render_asr_prompt_text(getattr(sample, "prompt", ""), tok)

    arr, sr = sf.read(io.BytesIO(audio_bytes))
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)  # mono
    tgt_sr = int(getattr(proc.feature_extractor, "sampling_rate", 16000))
    if sr != tgt_sr:
        import librosa

        arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=tgt_sr)

    out = proc(text=text, audio=arr, return_tensors="pt")
    prompt_ids = out["input_ids"][0].tolist()
    resp_ids = tok.encode(sample.response or "", add_special_tokens=False)
    if not resp_ids:  # never zero-length: prompt_length-1 must stay non-negative
        resp_ids = [int(getattr(tok, "eos_token_id", None) or 0)]
    sample.tokens = [int(t) for t in prompt_ids] + [int(t) for t in resp_ids]
    sample.response_length = len(resp_ids)
    sample.loss_mask = [1] * len(resp_ids)

    mm = {}
    if out.get("input_features") is not None:
        mm["input_features"] = out["input_features"]
    if out.get("feature_attention_mask") is not None:
        mm["feature_attention_mask"] = out["feature_attention_mask"]
    if mm:
        sample.multimodal_train_inputs = mm


def _build_form(audio_bytes: bytes, model: str, temperature: float, prompt: str | None):
    import aiohttp

    form = aiohttp.FormData()
    form.add_field(
        "file", io.BytesIO(audio_bytes), filename="clip.wav", content_type="audio/wav"
    )
    form.add_field("model", model)
    form.add_field("temperature", str(temperature))
    if prompt:
        form.add_field("prompt", prompt[:200])
    return form


async def transcription_rollout(args, sample, sampling_params: dict):
    """Drive Qwen3-ASR (served on /v1/audio/transcriptions) for one sample.

    GRPO sampling diversity comes from temperature; slime calls this function
    n_samples_per_prompt times per prompt, each producing one stochastic sample.
    """
    global _RR_INDEX
    import aiohttp

    audio_bytes = _extract_audio_bytes(sample)
    model = getattr(args, "served_model_name", None) or "qwen3-asr"
    temperature = float(sampling_params.get("temperature", 1.0))
    text_prompt = getattr(sample, "prompt_text", "") or ""
    if isinstance(text_prompt, str):
        text_prompt = text_prompt.replace("<audio>", "").strip()
    else:
        text_prompt = ""

    timeout = aiohttp.ClientTimeout(total=120)
    last_exc: Exception | None = None
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        # All engines across the cluster; round-robin the start so concurrent calls
        # spread across engines/nodes, then fall through the rest on failure (engines
        # are interchangeable).
        all_bases = await _discover_engine_bases(args, sess)
        if all_bases:
            start = _RR_INDEX % len(all_bases)
            _RR_INDEX += 1
            bases = all_bases[start:] + all_bases[:start]
        else:
            bases = []
        for base in bases:
            url = base.rstrip("/") + "/v1/audio/transcriptions"
            backoff = 1.0
            for _ in range(3):
                try:
                    form = _build_form(audio_bytes, model, temperature, text_prompt)
                    async with sess.post(url, data=form) as r:
                        if r.status == 404:
                            break  # wrong endpoint/host — try next candidate base
                        if r.status in (502, 503, 504):
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        r.raise_for_status()
                        body = await r.json()
                    sample.response = (body.get("text") or "").strip()
                    _populate_tokens(args, sample, audio_bytes)
                    return sample
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exc = e
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

    raise RuntimeError(
        f"transcription_rollout: no engine served /v1/audio/transcriptions. "
        f"Tried {bases}. Last error: {last_exc!r}"
    )
