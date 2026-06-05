from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import datetime
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from modal_training_gym.common.dataset import DatasetRow
from modal_training_gym.common.ids import create_hash
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_list, vol_put

from modal_training_gym.common.models.base import ParsedResponse

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.deployment import ModelDeployment
    from modal_training_gym.common.models.base import ModelConfig

EVAL_SUMMARY_STORE = MetadataStore.EVALS
EVAL_SUMMARY_KEY = "summary"
EVAL_SUMMARY_PAYLOAD_KEY = "summaries"


def _callable_name(fn: Callable[..., Any]) -> str:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if name:
        return name
    return type(fn).__name__


class EvalConfigDurable(BaseModel):
    """JSON-serializable audit record for an :class:`EvalConfig`."""

    eval_config_id: str
    dataset_name: str
    eval_fn_name: str
    prompt_column: str | None = None
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)

    def save(self) -> None:
        vol_put(
            MetadataStore.EVAL_CONFIGS,
            self.eval_config_id,
            self.model_dump(mode="json"),
        )

    @classmethod
    def from_id(cls, eval_config_id: str) -> "EvalConfigDurable":
        return cls.model_validate(vol_get(MetadataStore.EVAL_CONFIGS, eval_config_id))

    @classmethod
    def list_configs(cls) -> list["EvalConfigDurable"]:
        return [cls.model_validate(v) for v in vol_list(MetadataStore.EVAL_CONFIGS)]


class EvalRowResult(BaseModel):
    """One evaluated row: score, response text, and optional metadata."""

    score: float
    response: str = ""  # TODO, this doesn't have to be a string
    prompt: str = ""
    parsed_response: ParsedResponse | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )  # metadata that user can inject about the evaluation result


class EvalSummary(BaseModel):
    eval_id: str
    eval_config_id: str
    created_at: datetime.datetime
    total: int
    mean: float

    @classmethod
    def list_summaries(cls) -> list["EvalSummary"]:
        try:
            payload = vol_get(EVAL_SUMMARY_STORE, EVAL_SUMMARY_KEY)
        except KeyError:
            return []
        summaries = (
            payload.get(EVAL_SUMMARY_PAYLOAD_KEY, [])
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(summaries, list):
            return []
        return [cls.model_validate(summary) for summary in summaries]

    @classmethod
    def save_summaries(cls, summaries: list["EvalSummary"]) -> None:
        vol_put(
            EVAL_SUMMARY_STORE,
            EVAL_SUMMARY_KEY,
            {
                EVAL_SUMMARY_PAYLOAD_KEY: [
                    summary.model_dump(mode="json") for summary in summaries
                ]
            },
        )


class EvalResult(BaseModel):
    """Saved results for one evaluation run across a dataset."""

    eval_id: str
    eval_config_id: str
    deployment_id: str
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    rows: list[EvalRowResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def mean(self) -> float:
        return sum(r.score for r in self.rows) / self.total if self.total else 0.0

    def to_summary(self) -> EvalSummary:
        return EvalSummary(
            eval_id=self.eval_id,
            eval_config_id=self.eval_config_id,
            created_at=self.created_at,
            total=self.total,
            mean=self.mean,
        )

    def save(self) -> None:
        vol_put(MetadataStore.EVAL_RESULTS, self.eval_id, self.model_dump(mode="json"))
        summaries = EvalSummary.list_summaries()
        summaries_by_id = {summary.eval_id: summary for summary in summaries}
        summaries_by_id[self.eval_id] = self.to_summary()
        EvalSummary.save_summaries(
            sorted(
                summaries_by_id.values(),
                key=lambda summary: summary.created_at,
                reverse=True,
            )
        )

    @classmethod
    def from_id(cls, eval_id: str) -> "EvalResult":
        return cls.model_validate(vol_get(MetadataStore.EVAL_RESULTS, eval_id))

    @classmethod
    def list_results(cls) -> list["EvalResult"]:
        return [cls.model_validate(v) for v in vol_list(MetadataStore.EVAL_RESULTS)]


Response = str
EvalResponseFn = Callable[[DatasetRow, Response], EvalRowResult]  # TOOD: bad name
EvalFn = Callable[["ModelDeployment", DatasetRow], EvalRowResult]


@dataclass
class EvalConfig:
    """Evaluate a deployed model on a dataset config.

    The dataset must expose ``load()`` and return iterable dict examples.
    """

    dataset: "DatasetConfig"
    eval_fn: EvalFn | None = None
    eval_response_fn: EvalResponseFn | None = None
    prompt_column: str | None = None
    eval_config_id: str | None = None
    generate_kwargs: dict[str, Any] = field(default_factory=dict)

    def _build_eval_fn(self, eval_response_fn: EvalResponseFn) -> EvalFn:
        def eval_fn(
            deployment: ModelDeployment,
            example: DatasetRow,
        ) -> EvalRowResult:
            prompt = self.build_prompt(example)
            text = deployment.generate(
                prompt,
                ensure_ready=False,
                **self.generate_kwargs,
            )
            result = eval_response_fn(example, text)
            return EvalRowResult(
                score=result.score,
                response=text,
                prompt=prompt,
                metadata=result.metadata,
            )

        return eval_fn

    def __post_init__(self):
        if self.eval_config_id is None:
            class_name = type(self).__name__
            dataset_name = type(self.dataset).__name__
            eval_fn_name = _callable_name(self.eval_fn or self.eval_response_fn)
            self.eval_config_id = create_hash(
                "eval-config",
                class_name,
                dataset_name,
                eval_fn_name,
                "",
            )
        if self.eval_fn is None:
            assert self.eval_response_fn is not None, (
                "eval_fn or eval_response_fn must be set"
            )
            self.eval_fn = self._build_eval_fn(self.eval_response_fn)

    def to_durable(self) -> EvalConfigDurable:
        eval_callable = (
            self.eval_response_fn if self.eval_response_fn is not None else self.eval_fn
        )
        return EvalConfigDurable(
            eval_config_id=self.eval_config_id,
            dataset_name=type(self.dataset).__name__,
            eval_fn_name=_callable_name(eval_callable),
            prompt_column=self.prompt_column,
            generate_kwargs=self.generate_kwargs,
        )

    def save(self) -> EvalConfigDurable:
        durable = self.to_durable()
        durable.save()
        return durable

    def build_prompt(self, row: DatasetRow) -> str:
        prompt_column = (self.prompt_column or "").strip()
        input_column = getattr(self.dataset, "input_column", "")
        dataset_column = input_column if isinstance(input_column, str) else ""

        preferred_columns: list[str] = []
        if prompt_column:
            preferred_columns.append(prompt_column)
        if dataset_column and dataset_column not in preferred_columns:
            preferred_columns.append(dataset_column)
        for fallback in ("prompt", "input", "instruction", "question"):
            if fallback not in preferred_columns:
                preferred_columns.append(fallback)

        for column in preferred_columns:
            if column not in row:
                continue
            raw = str(row[column])
            if column in {prompt_column, dataset_column}:
                template = (
                    getattr(self.dataset, "prompt_template", "{input}") or "{input}"
                )
                row_context = {
                    key: str(value)
                    for key, value in row.items()
                    if isinstance(key, str)
                }
                try:
                    return template.format(input=raw, **row_context)
                except (KeyError, ValueError):
                    return raw
            return raw

        raise ValueError(
            "EvalConfig.build_prompt() could not resolve a prompt column. "
            "Set EvalConfig.prompt_column or dataset.input_column, or include one of "
            "['prompt', 'input', 'instruction', 'question'] in dataset rows."
        )

    def evaluate(
        self,
        deployment: "ModelDeployment",
        debug: bool = False,
        max_concurrency: int = 1,
    ) -> EvalResult:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self.save()
        deployment.wait_until_ready()

        def _evaluate_indexed(
            item: tuple[int, DatasetRow],
        ) -> tuple[int, EvalRowResult]:
            idx, example = item
            return idx, self.eval_fn(deployment, example)

        results: list[EvalRowResult] = []
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            indexed_results = executor.map(
                _evaluate_indexed,
                enumerate(self.dataset.load(split="eval"), start=1),
            )
            for idx, result in indexed_results:
                if debug:
                    print(
                        f"Finished example {idx}: "
                        f"response={result.response!r} score={result.score}",
                        flush=True,
                    )
                results.append(result)

        created_at = datetime.datetime.now(datetime.UTC)
        eval_id = create_hash(
            "eval",
            self.eval_config_id,
            deployment.deployment_id,
            "",
            "",
        )
        result = EvalResult(
            eval_id=eval_id,
            deployment_id=deployment.deployment_id,
            eval_config_id=self.eval_config_id,
            created_at=created_at,
            rows=results,
        )
        result.save()
        return result


# ── Harbor evaluation helpers ────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, model: "ModelConfig | None" = None) -> str:
    """Extract Python code from an LLM response.

    When *model* is provided, uses ``model.parse_response`` to strip
    thinking tags and chat-template artifacts, and checks tool-call
    arguments for a ``code`` key.  Falls back to regex heuristics when
    *model* is ``None``.
    """
    if model is not None:
        parsed = model.parse_response(text)
        for tool_call in parsed.tool_calls:
            code = tool_call.arguments.get("code", "")
            if code:
                return code
        content = parsed.content
    else:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if "<|im_start|>assistant" in normalized:
            normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
        if "</think>" in normalized:
            normalized = normalized.split("</think>", 1)[-1]
        normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
        content = normalized

    if match := _CODE_FENCE_RE.search(content):
        return match.group(1).strip()
    return content


def score_in_sandbox(
    code: str,
    *,
    test_cases: list[dict[str, str]],
    timeout_sec: int = 60,
    sandbox_cpu: float = 1.0,
    sandbox_memory: int = 1024,
    python_version: str = "3.11",
) -> tuple[float, dict[str, Any]]:
    """Run *code* against *test_cases* in a Modal sandbox.

    Each test case is a dict with ``input`` and ``expected_output`` keys.
    The code is executed once per test case with the input piped to stdin.
    Returns ``(fraction_passed, metadata_dict)``.
    """
    import modal

    if not test_cases:
        return 0.0, {"error": "no test cases"}

    runner = (
        "import sys, json, io, contextlib\n"
        "cases = json.loads(sys.argv[1])\n"
        "results = []\n"
        "for case in cases:\n"
        "    old_stdin = sys.stdin\n"
        '    sys.stdin = io.StringIO(case["input"])\n'
        "    buf = io.StringIO()\n"
        "    ok = False\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(buf):\n"
        '            exec(compile(case["code"], "<solution>", "exec"))\n'
        '        ok = buf.getvalue().strip() == case["expected_output"].strip()\n'
        "    except Exception as exc:\n"
        '        buf.write(f"ERROR: {exc}")\n'
        "    finally:\n"
        "        sys.stdin = old_stdin\n"
        '    results.append({"passed": ok, "stdout": buf.getvalue()})\n'
        "print(json.dumps(results))\n"
    )

    cases_payload = json.dumps(
        [
            {
                "code": code,
                "input": tc.get("input", ""),
                "expected_output": tc.get("expected_output", ""),
            }
            for tc in test_cases
        ]
    )

    app = modal.App.lookup("training-gym-sandbox-rm", create_if_missing=True)
    image = modal.Image.debian_slim(python_version=python_version)
    sb = modal.Sandbox.create(
        "python",
        "-c",
        runner,
        cases_payload,
        image=image,
        cpu=sandbox_cpu,
        memory=sandbox_memory,
        timeout=timeout_sec,
        app=app,
    )
    sb.wait()

    stdout = sb.stdout.read()
    stderr = sb.stderr.read()

    metadata: dict[str, Any] = {"stderr": stderr}
    try:
        results = json.loads(stdout)
        passed = sum(1 for r in results if r.get("passed"))
        metadata["per_case"] = results
        return passed / len(test_cases), metadata
    except (json.JSONDecodeError, TypeError):
        metadata["raw_stdout"] = stdout
        return 0.0, metadata


@dataclass
class HarborEval(EvalConfig):
    """Evaluate a deployed model on a Harbor dataset using sandbox execution.

    Automates the common pattern of generating code from a Harbor task,
    extracting it from the LLM response, running it in a Modal sandbox,
    and comparing stdout against expected test-case outputs.

    When neither ``eval_fn`` nor ``eval_response_fn`` is provided, a
    default sandbox-backed scorer is used automatically.  Pass
    ``extract_code_fn`` to override how code is pulled from the model
    response, or supply your own ``eval_fn`` to take full control.
    """

    model: "ModelConfig | None" = None
    test_cases: list[dict[str, str]] | None = None
    sandbox_timeout: int = 60
    sandbox_cpu: float = 1.0
    sandbox_memory: int = 1024
    sandbox_python_version: str = "3.11"
    extract_code_fn: Callable[[str], str] | None = None

    def _resolve_test_cases(self, example: DatasetRow) -> list[dict[str, str]]:
        label = example.get("label", {})
        if isinstance(label, str):
            try:
                label = json.loads(label)
            except (json.JSONDecodeError, ValueError):
                label = {}
        if isinstance(label, dict):
            cases = label.get("test_cases")
            if isinstance(cases, list) and cases:
                return cases
        if self.test_cases is not None:
            return self.test_cases
        return []

    def _extract_code(self, text: str) -> str:
        if self.extract_code_fn is not None:
            return self.extract_code_fn(text)
        return extract_code(text, model=self.model)

    def _build_messages(self, example: DatasetRow, prompt: str) -> list[dict[str, str]]:
        messages = example.get("messages")
        if isinstance(messages, list) and messages:
            return messages
        msgs: list[dict[str, str]] = []
        sys_prompt = getattr(self.dataset, "system_prompt", "")
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _harbor_eval_fn(
        self,
        deployment: "ModelDeployment",
        example: DatasetRow,
    ) -> EvalRowResult:
        prompt = self.build_prompt(example)
        messages = self._build_messages(example, prompt)
        response = deployment.generate(
            prompt,
            ensure_ready=False,
            messages=messages,
            **self.generate_kwargs,
        )
        code = self._extract_code(response)
        test_cases = self._resolve_test_cases(example)
        score, metadata = score_in_sandbox(
            code,
            test_cases=test_cases,
            timeout_sec=self.sandbox_timeout,
            sandbox_cpu=self.sandbox_cpu,
            sandbox_memory=self.sandbox_memory,
            python_version=self.sandbox_python_version,
        )

        parsed = self.model.parse_response(response) if self.model is not None else None

        return EvalRowResult(
            score=score,
            response=response,
            prompt=prompt,
            parsed_response=parsed,
            metadata=metadata,
        )

    def __post_init__(self) -> None:
        if self.eval_fn is None and self.eval_response_fn is None:
            self.eval_fn = self._harbor_eval_fn
        super().__post_init__()


# ── ASR transcription eval ───────────────────────────────────────────────
#
# Post-training evaluator for ASR models (e.g. Qwen3-ASR): serves the trained
# HF checkpoint on SGLang's /v1/audio/transcriptions, transcribes the dataset's
# clips, and writes an EvalResult with the dashboard panel's contract
# ({audio, reference, wer, hyp}). It's the gym-owned plumbing behind the one-line
# eval in the audio tutorial, so the example file stays short:
#
#     result = TrainConfig(...).train()
#     evaluate_asr(result, LibriSpeechASRDataset(n_rows=8))
#
# Dashboard audio is downsampled to 8 kHz mono so eval payloads stay light;
# transcription uses the full-resolution clip, so WER is unaffected.

_ASR_DASHBOARD_SR = 8000  # stored (playable) dashboard audio sample rate


def _asr_serve_and_eval(dataset, hf_dir: str, n_clips: int) -> dict:
    import base64
    import datetime
    import io
    import subprocess
    import time

    import jiwer
    import librosa
    import requests
    import soundfile as sf

    print(f"=== serving {hf_dir} ===")
    port = 30000
    proc = subprocess.Popen([
        "python", "-m", "sglang.launch_server",
        "--model-path", hf_dir,
        "--served-model-name", "qwen3-asr",
        "--trust-remote-code",
        "--host", "127.0.0.1", "--port", str(port),
        "--mem-fraction-static", "0.80",
    ])
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 1800
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"sglang server exited early (code {proc.returncode})")
        try:
            if requests.get(f"{base}/health", timeout=5).status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(10)
    if not ready:
        proc.terminate()
        raise RuntimeError("sglang server never became healthy")
    print("=== sglang healthy; running ASR eval ===")

    rows = dataset._build_rows()[:n_clips]
    results: list[EvalRowResult] = []
    try:
        for i, row in enumerate(rows):
            data_uri = row["audios"][0]
            ref = (row["label"] or "").lower().strip()
            b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
            arr, sr = sf.read(io.BytesIO(base64.b64decode(b64)))

            # Transcribe the full-resolution clip (WER must reflect real audio).
            buf = io.BytesIO()
            sf.write(buf, arr, sr, format="WAV")
            buf.seek(0)
            r = requests.post(
                f"{base}/v1/audio/transcriptions",
                files={"file": ("clip.wav", buf, "audio/wav")},
                data={"model": "qwen3-asr", "temperature": "0.0"},
                timeout=120,
            )
            r.raise_for_status()
            hyp = (r.json().get("text") or "").lower().strip()
            w = float(jiwer.wer(ref, hyp)) if ref else 0.0
            print(f"[{i}] WER={w:.3f} ref={ref[:48]!r} hyp={hyp[:48]!r}")

            # Light, downsampled clip for the dashboard player.
            small = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=_ASR_DASHBOARD_SR)
            sbuf = io.BytesIO()
            sf.write(sbuf, small, _ASR_DASHBOARD_SR, format="WAV", subtype="PCM_16")
            small_uri = "data:audio/wav;base64," + base64.b64encode(sbuf.getvalue()).decode()

            # Score is word accuracy (1 − WER) in [0, 1] — higher is better, matching
            # the dashboard's score model/histogram. Raw WER stays in metadata.
            results.append(
                EvalRowResult(
                    score=max(0.0, 1.0 - w),
                    response=hyp,
                    prompt=row["prompt"],
                    metadata={"audio": small_uri, "reference": ref, "wer": w, "hyp": hyp},
                )
            )
    finally:
        proc.terminate()

    label = hf_dir.rstrip("/").split("/checkpoints/", 1)[-1].split("/")[0] or "trained"
    eval_config_id = "qwen3-asr-librispeech-wer"
    deployment_id = f"qwen3-asr-1.7b-{label}"
    eval_id = create_hash(
        "eval", eval_config_id, deployment_id, str(datetime.datetime.now(datetime.UTC)), ""
    )
    result = EvalResult(
        eval_id=eval_id,
        deployment_id=deployment_id,
        eval_config_id=eval_config_id,
        rows=results,
    )
    result.save()
    mean_wer = sum(r.metadata["wer"] for r in results) / len(results)
    print(f"saved EvalResult {eval_id}  mean WER={mean_wer:.3f}  ({len(results)} rows) -> dashboard")
    return {"eval_id": eval_id, "mean_wer": mean_wer, "rows": len(results)}


def evaluate_asr(result, dataset, *, n_clips: int = 8, gpu_type: str = "H100") -> dict:
    """Serve ``result``'s trained checkpoint, eval ``dataset``, publish to the dashboard.

    Spins up a short-lived 1×GPU Modal app (the native slime image serves Qwen3-ASR
    on ``/v1/audio/transcriptions`` directly), transcribes the clips, scores word
    accuracy (1 − WER), and writes a gym :class:`EvalResult` to the shared metadata
    volume the dashboard reads. Returns ``{"eval_id", "mean_wer", "rows"}``.
    """
    import modal

    from modal_training_gym.common.checkpoint import list_checkpoints
    from modal_training_gym.frameworks.slime.launcher import SLIME_IMAGE

    checkpoints = list_checkpoints(result.training_run_id)
    if not checkpoints:
        raise RuntimeError(f"no checkpoints for run {result.training_run_id}")
    hf = [c for c in checkpoints if c.path.rstrip("/").endswith("_hf")]
    ckpt = (hf or checkpoints)[-1]
    if not ckpt.path.rstrip("/").endswith("_hf"):
        raise RuntimeError(
            f"latest checkpoint {ckpt.path!r} is not an HF export; "
            "ensure the run exported (megatron_to_hf_mode='bridge')."
        )
    vol_name = ckpt.checkpoints_volume_name or f"{result.app_name}-checkpoints"
    mount_path = ckpt.checkpoints_mount_path or "/checkpoints"

    image = (
        modal.Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands("rm -rf /root/.cache/huggingface")  # let the hf-cache volume mount
        .uv_pip_install("jiwer", "librosa", "soundfile", "randomname")
        .add_local_python_source("modal_training_gym", copy=True)
    )

    app = modal.App("qwen3-asr-eval", image=image)
    remote = app.function(
        gpu=gpu_type,
        timeout=2400,
        volumes={
            mount_path: modal.Volume.from_name(vol_name, create_if_missing=True),
            "/root/.cache/huggingface": modal.Volume.from_name(
                "huggingface-cache", create_if_missing=True
            ),
        },
        secrets=[modal.Secret.from_name("huggingface-secret")],
        serialized=True,
    )(_asr_serve_and_eval)

    with app.run():
        out = remote.remote(dataset, ckpt.path, n_clips)
    print(f"eval published to dashboard: {out}")
    return out
