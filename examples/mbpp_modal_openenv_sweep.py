"""Fan out MBPP OpenEnv-backed Training Gym sweeps with a Modal controller.

Usage:

    # Preview the matrix without launching jobs:
    uv run modal run examples/mbpp_modal_openenv_sweep.py --dry-run

    # Launch the default 12-job OpenEnv matrix:
    uv run modal run examples/mbpp_modal_openenv_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
MBPP_TRAIN_OPENENV_GYM = REPO_ROOT / "examples" / "mbpp_train_openenv_gym.py"
MBPP_OPENENV_SANDBOX = REPO_ROOT / "examples" / "mbpp_openenv_sandbox.py"
MBPP_TRAIN_GYM = REPO_ROOT / "examples" / "mbpp_train_gym.py"

MODEL_KEYS = ("Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B")
DEFAULT_ROLLOUTS = (100, 200)
DEFAULT_BREVITY_WEIGHTS = (0.1, 0.0)

controller_app = modal.App("mbpp-openenv-training-sweep-controller")

controller_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "cloudpickle",
        "datasets",
        "fastapi",
        "huggingface_hub",
        "modal>=1.4.0",
        "msgspec",
        "openai",
        "openenv==0.3.1",
        "pydantic",
        "randomname",
    )
    .add_local_python_source("modal_training_gym", copy=True)
    .add_local_file(
        MBPP_TRAIN_OPENENV_GYM,
        remote_path="/root/mbpp_train_openenv_gym.py",
        copy=True,
    )
    .add_local_file(
        MBPP_OPENENV_SANDBOX,
        remote_path="/root/mbpp_openenv_sandbox.py",
        copy=True,
    )
    .add_local_file(MBPP_TRAIN_GYM, remote_path="/root/mbpp_train_gym.py", copy=True)
)


@dataclass(frozen=True)
class SweepCombo:
    model: str
    num_rollout: int
    brevity_weight: float
    run_label: str


@dataclass(frozen=True)
class SweepSettings:
    subset: str
    train_size: int
    test_size: int
    split_seed: int
    train_repeats: int
    save_interval: int
    rollout_batch_size: int
    global_batch_size: int
    n_samples_per_prompt: int
    n_samples_per_eval_prompt: int
    max_response_len: int
    temperature: float
    max_tokens_per_gpu: int
    gpu_type: str
    max_concurrency: int
    skip_base_eval: bool


def brevity_slug(weight: float) -> str:
    return f"bw{int(round(weight * 100)):03d}"


def build_combos(
    *,
    models: list[str],
    rollouts: list[int],
    brevity_weights: list[float],
    run_label_prefix: str,
) -> list[SweepCombo]:
    combos: list[SweepCombo] = []
    for model in models:
        for num_rollout in rollouts:
            for brevity_weight in brevity_weights:
                run_label = (
                    f"{run_label_prefix}-r{num_rollout}-{brevity_slug(brevity_weight)}"
                )
                combos.append(
                    SweepCombo(
                        model=model,
                        num_rollout=num_rollout,
                        brevity_weight=brevity_weight,
                        run_label=run_label,
                    )
                )
    return combos


def chunked(items: list[SweepCombo], chunk_size: int | None) -> list[list[SweepCombo]]:
    if chunk_size is None:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


@controller_app.function(
    image=controller_image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=60 * 60 * 24,
    name="run_combo",
)
def run_combo(combo_payload: dict[str, object], settings_payload: dict[str, object]):
    sys.path.append("/root")

    from argparse import Namespace

    from mbpp_train_openenv_gym import train_and_eval_model

    combo = SweepCombo(
        model=str(combo_payload["model"]),
        num_rollout=int(combo_payload["num_rollout"]),
        brevity_weight=float(combo_payload["brevity_weight"]),
        run_label=str(combo_payload["run_label"]),
    )
    settings = SweepSettings(
        subset=str(settings_payload["subset"]),
        train_size=int(settings_payload["train_size"]),
        test_size=int(settings_payload["test_size"]),
        split_seed=int(settings_payload["split_seed"]),
        train_repeats=int(settings_payload["train_repeats"]),
        save_interval=int(settings_payload["save_interval"]),
        rollout_batch_size=int(settings_payload["rollout_batch_size"]),
        global_batch_size=int(settings_payload["global_batch_size"]),
        n_samples_per_prompt=int(settings_payload["n_samples_per_prompt"]),
        n_samples_per_eval_prompt=int(settings_payload["n_samples_per_eval_prompt"]),
        max_response_len=int(settings_payload["max_response_len"]),
        temperature=float(settings_payload["temperature"]),
        max_tokens_per_gpu=int(settings_payload["max_tokens_per_gpu"]),
        gpu_type=str(settings_payload["gpu_type"]),
        max_concurrency=int(settings_payload["max_concurrency"]),
        skip_base_eval=bool(settings_payload["skip_base_eval"]),
    )

    started_at = time.time()
    args = Namespace(
        models=[combo.model],
        subset=settings.subset,
        train_size=settings.train_size,
        test_size=settings.test_size,
        split_seed=settings.split_seed,
        train_repeats=settings.train_repeats,
        num_rollout=combo.num_rollout,
        save_interval=settings.save_interval,
        rollout_batch_size=settings.rollout_batch_size,
        global_batch_size=settings.global_batch_size,
        n_samples_per_prompt=settings.n_samples_per_prompt,
        n_samples_per_eval_prompt=settings.n_samples_per_eval_prompt,
        max_response_len=settings.max_response_len,
        temperature=settings.temperature,
        brevity_weight=combo.brevity_weight,
        run_label=combo.run_label,
        max_tokens_per_gpu=settings.max_tokens_per_gpu,
        gpu_type=settings.gpu_type,
        max_concurrency=settings.max_concurrency,
        skip_base_eval=settings.skip_base_eval,
    )
    result = train_and_eval_model(combo.model, args)
    return {
        "combo": asdict(combo),
        "elapsed_sec": round(time.time() - started_at, 1),
        "result": result,
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


@controller_app.local_entrypoint()
def main(
    models: str = ",".join(MODEL_KEYS),
    rollouts: str = ",".join(str(item) for item in DEFAULT_ROLLOUTS),
    brevity_weights: str = ",".join(str(item) for item in DEFAULT_BREVITY_WEIGHTS),
    run_label_prefix: str = "openenv-long",
    max_parallel: int = 0,
    dry_run: bool = False,
    subset: str = "sanitized",
    train_size: int = 327,
    test_size: int = 100,
    split_seed: int = 20260608,
    train_repeats: int = 1,
    save_interval: int = 10,
    rollout_batch_size: int = 8,
    global_batch_size: int = 8,
    n_samples_per_prompt: int = 4,
    n_samples_per_eval_prompt: int = 4,
    max_response_len: int = 512,
    temperature: float = 0.9,
    max_tokens_per_gpu: int = 8192,
    gpu_type: str = "H100",
    max_concurrency: int = 4,
    include_base_eval: bool = False,
) -> None:
    combos = build_combos(
        models=parse_csv(models),
        rollouts=parse_int_csv(rollouts),
        brevity_weights=parse_float_csv(brevity_weights),
        run_label_prefix=run_label_prefix,
    )
    settings = SweepSettings(
        subset=subset,
        train_size=train_size,
        test_size=test_size,
        split_seed=split_seed,
        train_repeats=train_repeats,
        save_interval=save_interval,
        rollout_batch_size=rollout_batch_size,
        global_batch_size=global_batch_size,
        n_samples_per_prompt=n_samples_per_prompt,
        n_samples_per_eval_prompt=n_samples_per_eval_prompt,
        max_response_len=max_response_len,
        temperature=temperature,
        max_tokens_per_gpu=max_tokens_per_gpu,
        gpu_type=gpu_type,
        max_concurrency=max_concurrency,
        skip_base_eval=not include_base_eval,
    )

    print("MBPP OpenEnv sweep combos:")
    for combo in combos:
        print(json.dumps(asdict(combo), sort_keys=True))
    print(f"settings={json.dumps(asdict(settings), sort_keys=True)}")

    if dry_run:
        return

    settings_payload = asdict(settings)
    results = []
    chunk_size = max_parallel if max_parallel > 0 else None
    for wave_index, wave in enumerate(chunked(combos, chunk_size), 1):
        print(f"launching wave {wave_index}: {len(wave)} combo(s)")
        calls = [
            (
                combo,
                run_combo.spawn(asdict(combo), settings_payload),
            )
            for combo in wave
        ]
        for combo, call in calls:
            print(
                f"spawned {combo.model} {combo.run_label}: "
                f"function_call_id={call.object_id}"
            )
        for combo, call in calls:
            result = call.get()
            print(json.dumps(result, sort_keys=True))
            results.append(result)

    print("MBPP OpenEnv sweep completed:")
    print(json.dumps(results, sort_keys=True))
