"""slime rollout that drives an ASR model via /v1/audio/transcriptions.

slime's default rollout posts chat-formatted prompts to /v1/chat/completions.
That works for Qwen-Omni (a chat model with audio content blocks), but Qwen3-ASR
is served by SGLang as a Whisper-style model on a *different* endpoint:

    POST /v1/audio/transcriptions
      file=<wav bytes>, model=<served>, temperature=<T>, prompt=<optional>

This module is the slime glue layer: it extracts audio from the slime Sample,
discovers the SGLang engines, POSTs the clip, and writes the transcript +
training fields back onto the Sample. The generic audio decode lives in
:mod:`modal_training_gym.common.audio`; the Qwen3-ASR tokenization lives in
:mod:`modal_training_gym.common.models.qwen3_asr_1_7b`.

Wire it in via ``SlimeRecipe.custom_generate_function`` (``Qwen3_ASR_1_7b_Recipe``
already does). slime calls it ``n_samples_per_prompt`` times per prompt, sampling
at ``sampling_params["temperature"]`` to give GRPO its N samples to score.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

from modal_training_gym.common.audio import coerce_audio_bytes
from modal_training_gym.common.models.qwen3_asr_1_7b import encode_training_inputs

# ``args`` is slime's runtime arg namespace; like the sibling ``opd_reward`` rollout
# it has no public type, so we read attributes defensively and annotate it ``Any``.


def _extract_audio_bytes(sample: Any) -> bytes:
    """Pull raw audio bytes out of a slime Sample.

    Our ``MultimodalDataset`` runs with ``apply_chat_template=False``, so slime
    keeps ``sample.prompt`` a conversation list and the audio rides in the message
    content as ``{"type": "audio", "audio": <data-uri>}`` (slime's
    ``process_vision_info`` extracts only images/videos, so audio never reaches
    ``multimodal_inputs``). Fail loudly if it's missing — a silent miss would train
    on audio-free prompts.
    """
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, list):
        for msg in prompt:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and (
                    item.get("type") == "audio" or "audio" in item
                ):
                    audio = coerce_audio_bytes(
                        item.get("audio") or item.get("audio_url")
                    )
                    if audio:
                        return audio
    raise RuntimeError(
        "transcription_rollout: no audio on the slime Sample. Expected a "
        "conversation-list prompt with a {'type': 'audio', 'audio': <data-uri>} item."
    )


# ── SGLang engine discovery ──────────────────────────────────────────────────


def _router_base(args: Any) -> str | None:
    ip = getattr(args, "sglang_router_ip", None)
    if not ip:
        return None
    ip = str(ip)
    host = ip if ip.startswith("http") else f"http://{ip}"
    port = getattr(args, "sglang_router_port", None)
    return f"{host}:{port}" if port else host


def _local_engine_bases(args: Any) -> list[str]:
    """SGLang engine base URLs on the local Ray node.

    The slime router does NOT proxy /v1/audio/transcriptions (404s on its fixed
    Rust routes), so we POST audio to engines directly. Colocated engines bind to
    the Ray node IP at base_port 15000, 15002, ... (one per engine).
    """
    import socket

    base_port = int(getattr(args, "sglang_server_port", None) or 15000)
    n_engines = int(
        getattr(args, "rollout_num_gpus", 0)
        or getattr(args, "actor_num_gpus_per_node", 1)
        or 1
    )
    ports = [base_port + 2 * i for i in range(max(n_engines, 1) + 1)]

    hosts: list[str] = []
    # Engines bind to the Ray node IP (not localhost), so this is the host that
    # actually works from the colocated rollout process.
    try:
        import ray

        node_ip = ray.util.get_node_ip_address()
        if node_ip:
            hosts.append(node_ip)
    except (ImportError, RuntimeError):
        pass
    try:
        hosts.append(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    hosts.append("127.0.0.1")

    seen: set[str] = set()
    bases: list[str] = []
    for host in hosts:
        for port in ports:
            base = f"http://{host}:{port}"
            if base not in seen:
                seen.add(base)
                bases.append(base)
    return bases


async def _discover_engine_bases(args: Any, session: Any) -> list[str]:
    """Every SGLang engine base URL across the cluster, discovered once per run.

    slime's router knows every engine (each self-registers at startup), so
    ``GET {router}/workers`` spans all nodes — which a local-node probe can't. The
    router can't proxy the transcription endpoint, so we use it only for discovery
    and POST audio to each engine directly; we fall back to local-node probing when
    no router / registry is available (single node). Cached on ``args`` so the
    router is queried once, not per sample.
    """
    cached = getattr(args, "_audio_engine_bases", None)
    if cached is not None:
        return cached

    import aiohttp

    bases: list[str] = []
    router = _router_base(args)
    if router:
        # /workers (sglang_router > 0.2.1) -> {"workers": [{"url": ...}]};
        # /list_workers (older) -> {"urls": [...]}.
        for path, extract in (
            (
                "/workers",
                lambda b: [w["url"] for w in b.get("workers", []) if w.get("url")],
            ),
            ("/list_workers", lambda b: list(b.get("urls", []))),
        ):
            try:
                async with session.get(
                    router + path, timeout=aiohttp.ClientTimeout(total=15)
                ) as r:
                    if r.status != 200:
                        continue
                    payload = await r.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                continue
            urls = extract(payload)
            if urls:
                bases = [u if u.startswith("http") else f"http://{u}" for u in urls]
                break

    if not bases:
        bases = _local_engine_bases(args)

    try:
        args._audio_engine_bases = bases
    except (AttributeError, TypeError):
        pass
    return bases


def _next_round_robin(args: Any, n: int) -> int:
    """Index to start engine selection at, advancing a run-scoped counter on ``args``.

    Spreads concurrent rollouts across engines/nodes without module-global state.
    """
    if n <= 0:
        return 0
    index = getattr(args, "_audio_rr_index", 0)
    try:
        args._audio_rr_index = index + 1
    except (AttributeError, TypeError):
        pass
    return index % n


# ── the rollout ──────────────────────────────────────────────────────────────


def _build_form(audio_bytes: bytes, model: str, temperature: float, prompt: str) -> Any:
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


def _populate_training_fields(args: Any, sample: Any, audio_bytes: bytes) -> None:
    """Map the model's encoded inputs onto the slime Sample for training.

    Delegates Qwen3-ASR tokenization (prompt + expanded ``<audio_pad>``, mel
    features) to the model module, then writes slime's token / response-length /
    loss-mask / multimodal fields. ``response_length`` and ``loss_mask`` cover the
    response only.
    """
    checkpoint = getattr(args, "hf_checkpoint", None) or getattr(
        args, "model_name", None
    )
    encoded = encode_training_inputs(
        checkpoint, getattr(sample, "prompt", ""), sample.response or "", audio_bytes
    )
    response_ids = encoded["response_ids"]
    sample.tokens = encoded["prompt_ids"] + response_ids
    sample.response_length = len(response_ids)
    sample.loss_mask = [1] * len(response_ids)
    if encoded["multimodal_inputs"]:
        sample.multimodal_train_inputs = encoded["multimodal_inputs"]


async def transcription_rollout(args: Any, sample: Any, sampling_params: dict) -> Any:
    """Drive Qwen3-ASR (served on /v1/audio/transcriptions) for one sample.

    slime calls this ``n_samples_per_prompt`` times per prompt; GRPO's sampling
    diversity comes from temperature. We POST the clip, set ``sample.response`` to
    the transcript, then populate the trainer's token + multimodal fields so the
    actor's log-probs are audio-conditioned.
    """
    import aiohttp

    audio_bytes = _extract_audio_bytes(sample)
    model = getattr(args, "served_model_name", None) or "qwen3-asr"
    temperature = float(sampling_params.get("temperature", 1.0))
    prompt_text = getattr(sample, "prompt_text", "") or ""
    prompt_text = (
        prompt_text.replace("<audio>", "").strip()
        if isinstance(prompt_text, str)
        else ""
    )

    last_exc: Exception | None = None
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        bases = await _discover_engine_bases(args, session)
        # Round-robin the start so concurrent rollouts spread across engines/nodes,
        # then fall through the rest on failure (engines are interchangeable).
        start = _next_round_robin(args, len(bases))
        bases = bases[start:] + bases[:start]
        for base in bases:
            url = base.rstrip("/") + "/v1/audio/transcriptions"
            backoff = 1.0
            for _ in range(3):
                try:
                    form = _build_form(audio_bytes, model, temperature, prompt_text)
                    async with session.post(url, data=form) as r:
                        if r.status == 404:
                            break  # wrong endpoint/host — try the next engine
                        if r.status in (502, 503, 504):
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        r.raise_for_status()
                        body = await r.json()
                    sample.response = (body.get("text") or "").strip()
                    _populate_training_fields(args, sample, audio_bytes)
                    return sample
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_exc = exc
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

    raise RuntimeError(
        "transcription_rollout: no engine served /v1/audio/transcriptions "
        f"(tried {bases}). Last error: {last_exc!r}"
    )
