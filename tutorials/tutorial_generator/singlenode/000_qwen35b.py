# pyright: reportUndefinedVariable=false
"""Tutorial source for `004_qwen35b` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "Train Qwen3.6-35B-A3B on DAPO-math with GRPO",
    "difficulty": "Advanced",
    "order": 25,
    "api_classes": [
        "HuggingFaceDataset",
        "Qwen3_6_35B",
        "Qwen3_6_35b_Recipe",
        "TrainConfig",
        "TrainResult",
        "ensure_endpoint",
        "endpoint_chat",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Training Qwen3.6-35B-A3B on DAPO-math

    This tutorial trains **Qwen3.6-35B-A3B** (a 35B-parameter MoE model
    with ~3B active) on grade-school math problems from
    [DAPO-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k).

    The loop:
    1. Load math problems from HuggingFace via `HuggingFaceDataset`.
    2. Score model outputs using slime's built-in `deepscaler` reward
       model, which extracts the final numerical answer and compares
       it to the ground truth.
    3. Feed that score back as a GRPO reward through SLIME.
    4. Point a [Modal Endpoint](https://modal.com/docs/guide/endpoints)
       at the trained checkpoint and run a custom math accuracy check.

    Qwen3.6-35B-A3B uses slime's mbridge conversion path:
    the HuggingFace checkpoint is pre-converted to torch_dist format
    before training, enabling fast batched weight sync during training steps.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run python tutorials/singlenode/000_qwen35b/000_qwen35b.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    import re

    from modal_training_gym import (
        HuggingFaceDataset,
        Qwen3_6_35B,
        TrainConfig,
        endpoint_chat,
        ensure_endpoint,
    )
    from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_35b_Recipe


@markdown
def _dataset_intro():
    """
    ## Load DAPO-math from HuggingFace

    [DAPO-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
    contains ~17k math problems with ground-truth answers. We use a
    small subset for this tutorial — 100 training samples and 20 held out.
    """


@code
def _dataset():
    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_column = "prompt"
        output_column = "label"
        output_format = "jsonl"
        apply_chat_template = True

    train_dataset = MathDataset(hf_split="train[:100]")
    check_dataset = MathDataset(hf_split="train[100:120]")


@notebook_only
@markdown
def _dataset_preview():
    """
    Let's take a quick look at the dataset.
    """


@notebook_only
@code
def _dataset_preview_code():
    rows = check_dataset.load()
    for row in rows.select(range(2)):
        prompt = row["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]["content"] if prompt else ""
        print(prompt[:200])
        print(f"  label: {row['label']}")
        print()


@markdown
def _train_intro():
    """
    ## Train with SLIME

    This MoE model runs on 1 × 8×H100 with TP2, PP2, CP1, EP4,
    and optimizer CPU offload, matching the native Slime parallelism
    that works for Qwen3.6-35B-A3B.

    Key points:
    - **`rm_type="deepscaler"`** — slime's built-in math reward that
      extracts and compares numerical answers. No custom reward function
      or sandbox needed.
    - The HF checkpoint is pre-converted to torch_dist format; slime's
      implicit mbridge mode handles fast weight sync during training steps.
    - Built-in slime model args come from
      `scripts/models/qwen3.5-35B-A3B.sh`; the tutorial does not patch slime.
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3_6_35B(),
        dataset=train_dataset,
        recipe=Qwen3_6_35b_Recipe(
            rm_type="deepscaler",
            num_rollout=10,
        ),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _serve_intro():
    """
    ## Serve and check the trained checkpoint

    Point a [Modal Endpoint](https://modal.com/docs/guide/endpoints) at the
    checkpoint via `ensure_endpoint` (`--custom-volume-*`). Endpoints load
    HuggingFace-format directories, so `train_result.hf_model()` converts the
    newest Megatron checkpoint on demand and hands back a config pointing at it.
    Then score yourself with `endpoint_chat`.
    """


@code
def _serve_trained():
    trained_model = train_result.hf_model()
    print(f"Checkpoint: {trained_model.model_path}")

    trained_url = ensure_endpoint(
        name=f"gym-qwen3-6-35b-math-trained-{train_result.training_run_id}",
        model=trained_model.model_name,
        custom_volume_name=train_result.checkpoints_volume,
        custom_volume_path=trained_model.model_path,
    )
    print(f"Trained model endpoint: {trained_url}")


@markdown
def _custom_check_intro():
    """
    ## Custom scoring loop

    Loop a held-out slice yourself: call the endpoint, extract the answer,
    compare to the label, aggregate accuracy.
    """


@code
def _custom_check():
    def _normalize_answer(answer: str) -> str:
        answer = str(answer).strip()
        answer = answer.split("=")[-1]
        for old, new in [
            ("$", ""),
            ("\\$", ""),
            (",", ""),
            (" ", ""),
            ("\\text{", ""),
            ("}", ""),
            ("\\boxed{", ""),
        ]:
            answer = answer.replace(old, new)
        return answer.strip()

    def _extract_answer(response: str) -> str:
        match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response)
        return match[-1].strip() if match else "[INVALID]"

    def _check_math(response: str, label: str) -> bool:
        pred = _normalize_answer(_extract_answer(response))
        gt = _normalize_answer(label)
        try:
            gt = str(int(float(gt)))
        except (ValueError, OverflowError):
            pass
        return pred == gt

    def run_custom_check(url: str, model_id: str) -> float:
        rows = check_dataset.load()
        scores = []
        for example in rows:
            prompt = example.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0]["content"] if prompt else ""
            label = example.get("label", "")
            response = endpoint_chat(
                url,
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
    )
            correct = _check_math(response, label)
            pred = _normalize_answer(_extract_answer(response))
            print(
                f"correct={correct} pred={pred!r} label={label!r}",
                flush=True,
            )
            scores.append(1.0 if correct else 0.0)
        return sum(scores) / len(scores) if scores else float("nan")

    print("——— Running trained model custom check... ———")
    trained_mean = run_custom_check(trained_url, trained_model.model_name)
    print(f"Trained accuracy: {trained_mean:.1%}")
    print("——— Trained model custom check complete ———")
