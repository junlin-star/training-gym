# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `007_computer_use` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "GUI grounding with Qwen3-VL-8B — predict click coordinates from screenshots",
    "difficulty": "Advanced",
    "order": 40,
    "api_classes": [
        "Qwen3VL_8B",
        "Qwen3VL_Recipe",
        "MultimodalDataset",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "ModelDeployment",
        "TrainConfig",
        "list_checkpoints",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # GUI Grounding with Qwen3-VL-8B

    This tutorial trains **Qwen3-VL-8B-Instruct** via GRPO to predict click
    coordinates given a screenshot and a natural-language instruction like
    "click the Submit button".

    The task is simple: given an image of a GUI and an instruction identifying
    a UI element, output the normalized `(x, y)` center coordinate of that
    element. This is a foundational capability for computer-use agents.

    We use the [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot)
    benchmark — a standard GUI grounding evaluation set covering iOS, Android,
    macOS, Windows, and Web screenshots with annotated bounding boxes.

    The reward is distance-based: +1 for a prediction within 5% of the target,
    linearly decaying to −1 for predictions 50%+ away.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run locally (your machine drives the Modal GPU workers):

    ```
    cd training-gym
    uv sync
    uv run python tutorials/rl/007_computer_use/007_computer_use.py
    ```

    To detach and watch it from the Modal dashboard instead:

    ```
    uv run modal run -d tutorials/rl/007_computer_use/007_computer_use.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@markdown
def _prereqs():
    """
    ## Prerequisites

    This tutorial requires a Modal Secret named `huggingface-secret` containing your
    `HF_TOKEN`. Create one at [modal.com/secrets](https://modal.com/secrets) if you
    haven't already — the cell below fails fast with instructions otherwise.
    """


@code
def _imports():
    import re
    from typing import Any

    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        ModelDeployment,
        MultimodalDataset,
        Qwen3VL_8B,
        Qwen3VL_Recipe,
        TrainConfig,
        list_checkpoints,
    )


@markdown
def _dataset_intro():
    """
    ## Dataset

    We use [rootsautomation/ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot)
    — ~1,200 GUI screenshots annotated with natural-language instructions and
    bounding boxes. Each row has:

    - `image` — a screenshot from iOS/Android/macOS/Windows/Web
    - `instruction` — e.g. "click the Submit button"
    - `bbox` — `[left, top, right, bottom]` in normalized [0, 1] coordinates

    We convert each bounding box to a center-point `(x, y)` as the training
    target. The model learns to output these coordinates.

    For this tutorial we train on 800 samples and hold out 200 for evaluation.
    """


@code
def _dataset():
    # One "<image>" placeholder per image: slime's _build_messages splits the
    # prompt on "<image>" to interleave the image column, and asserts the count
    # matches (one screenshot here). apply_chat_template then renders it into the
    # Qwen3-VL vision tokens.
    GROUNDING_PROMPT = (
        "<image>\n"
        "You are a GUI agent. Given the screenshot, click on the element "
        "described below.\n\n"
        "Instruction: {instruction}\n\n"
        "Respond with ONLY the normalized (x, y) coordinates of the click "
        "target, formatted as: (x, y)\n"
        "where x and y are decimals between 0 and 1 representing the "
        "horizontal and vertical position on the screen."
    )

    class ScreenSpotDataset(MultimodalDataset):
        """GUI grounding dataset from ScreenSpot."""

        modality = "image"
        hf_repo: str = "rootsautomation/ScreenSpot"
        hf_split: str = "test"
        n_rows: int = 800
        row_offset: int = 0
        always_prepare: bool = True
        # True so slime renders the prompt + image into a single chat-templated
        # string (with vision tokens) before the Qwen3-VL processor tokenizes it.
        # With False the rollout hands the processor a raw message list and crashes
        # ('dict' object has no attribute 'replace').
        apply_chat_template: bool = True

        def __init__(self, **kwargs):
            super().__init__(rows=[], **kwargs)

        def _build_rows(self) -> list[dict]:
            from datasets import load_dataset

            ds = load_dataset(self.hf_repo, split=self.hf_split)
            start = min(self.row_offset, len(ds))
            stop = min(start + self.n_rows, len(ds))
            rows = []
            for row in ds.select(range(start, stop)):
                bbox = row["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                instruction = row["instruction"]

                # Convert PIL image to a data URI for the multimodal pipeline
                import base64
                import io

                buf = io.BytesIO()
                row["image"].save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                data_uri = f"data:image/png;base64,{img_b64}"

                rows.append(
                    {
                        self.input_key: GROUNDING_PROMPT.format(
                            instruction=instruction
                        ),
                        self.media_column: [data_uri],
                        self.label_key: f"{cx:.4f},{cy:.4f}",
                    }
                )
            return rows

        def load(self, split: str = "all") -> list[dict]:
            return self._build_rows()

        def prepare(self, path, eval_paths=None):
            rows = self._build_rows()
            self._write_jsonl(rows, path)
            if eval_paths:
                for ep in eval_paths.values():
                    self._write_jsonl(rows, ep)


@code
def _make_datasets():
    train_dataset = ScreenSpotDataset(n_rows=800)
    eval_dataset = ScreenSpotDataset(n_rows=200, row_offset=800)


@notebook_only
@code
def _dataset_peek():
    rows = eval_dataset.load()
    for row in rows[:2]:
        print(f"prompt: {row['prompt'][:100]}...")
        print(f"  label (cx, cy): {row['label']}")
        print()


@markdown
def _reward_intro():
    """
    ## Reward function

    The reward is based on Euclidean distance between the predicted and target
    coordinates:

    ```text
    R = +1.0                                    if dist < 0.05  (within 5%)
      = 1.0 - 2.0 * (dist - 0.05) / 0.45      if 0.05 <= dist < 0.50
      = -1.0                                    if dist >= 0.50
    ```

    This gives a clear positive signal for accurate predictions (within ~5% of
    the screen), a linear gradient for near-misses, and a flat penalty for
    completely wrong answers. The model also gets −1 if it fails to output
    parseable coordinates.
    """


@code
def _reward():
    def _parse_coordinates(text: str) -> tuple[float, float] | None:
        """Extract (x, y) from model output like '(0.45, 0.32)' or '0.45, 0.32'."""
        nums = re.findall(r"([\d.]+)", text)
        if len(nums) < 2:
            return None
        try:
            x, y = float(nums[0]), float(nums[1])
            if 0 <= x <= 1 and 0 <= y <= 1:
                return (x, y)
        except (ValueError, IndexError):
            pass
        return None

    async def grounding_reward(args, sample, **kwargs) -> float:
        response = getattr(sample, "response", "") or ""
        label = getattr(sample, "label", "") or ""

        pred = _parse_coordinates(response)
        if pred is None:
            return -1.0

        gt_parts = label.split(",")
        gt_x, gt_y = float(gt_parts[0]), float(gt_parts[1])

        dist = ((pred[0] - gt_x) ** 2 + (pred[1] - gt_y) ** 2) ** 0.5

        if dist < 0.05:
            return 1.0
        elif dist < 0.50:
            return 1.0 - 2.0 * (dist - 0.05) / 0.45
        else:
            return -1.0


@markdown
def _eval_base_intro():
    """
    ## Baseline Eval

    Let's evaluate the base Qwen3-VL-8B model on our held-out set before
    training to see how well it grounds UI elements out of the box.
    """


@code
def _eval_helpers():
    def grounding_eval_fn(
        deployment: ModelDeployment, example: dict
    ) -> EvalRowResult:
        # Drop the slime "<image>" placeholder: the eval sends the screenshot as a
        # separate image_url part, so the marker would just be stray text here.
        prompt = example.get("prompt", "").replace("<image>", "").strip()
        label = example.get("label", "")
        images = example.get("images", [])

        # Pass the screenshot through — a text-only request would grade the model
        # without ever showing it the GUI.
        response = deployment.generate(prompt, images=images, ensure_ready=False)

        pred = _parse_coordinates(response)
        if pred is None:
            dist = 1.0
        else:
            gt_parts = label.split(",")
            gt_x, gt_y = float(gt_parts[0]), float(gt_parts[1])
            dist = ((pred[0] - gt_x) ** 2 + (pred[1] - gt_y) ** 2) ** 0.5

        return EvalRowResult(
            score=1.0 if dist < 0.10 else 0.0,
            response=response,
            metadata={
                "distance": round(dist, 4),
                "pred": f"{pred[0]:.4f},{pred[1]:.4f}" if pred else "PARSE_FAIL",
                "label": label,
                "hit_at_10pct": dist < 0.10,
            },
        )


@code
def _eval_base():
    base_model = Qwen3VL_8B()
    base_deployment = DeploymentConfig(model=base_model).serve()
    print(f"Base model URL: {base_deployment.url}")

    eval_config = EvalConfig(dataset=eval_dataset, eval_fn=grounding_eval_fn)
    print("--- Evaluating base model... ---")
    base_eval = eval_config.evaluate(base_deployment, debug=True)
    n_hits = sum(1 for r in base_eval.rows if r.metadata.get("hit_at_10pct"))
    print(
        f"Base accuracy (@10%): {n_hits}/{len(base_eval.rows)} "
        f"({base_eval.mean:.1%})"
    )


@notebook_only
@code
def _base_examples():
    for r in base_eval.rows[:3]:
        status = "HIT" if r.metadata["hit_at_10pct"] else "MISS"
        print(f"[{status}] label={r.metadata['label']}, pred={r.metadata['pred']}")
        print(f"  dist={r.metadata['distance']:.4f}")
        print(f"  ...{r.response[-100:]}")
        print()


@markdown
def _train_intro():
    """
    ## Training

    We use `Qwen3VL_Recipe` which carries VL-specific defaults:
    - Padded (bshd) batches for the vision encoder
    - TP=2 for the 8B model across 8 H100s
    - Short response cap (256 tokens — coordinates are brief)
    - Lower SGLang memory fraction (vision tower uses VRAM)

    This tutorial runs 15 rollouts as a quick demo. For a more meaningful
    accuracy gain, increase `num_rollout`.
    """


@code
def _train():
    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=Qwen3VL_Recipe(
            custom_rm_function=grounding_reward,
            num_rollout=15,
            rollout_batch_size=8,
            n_samples_per_prompt=4,
            rollout_max_response_len=256,
            global_batch_size=16,
            lr=1e-6,
            save_interval=10,
        ),
    )
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _eval_trained_intro():
    """
    ## Evaluate the trained model

    Let's run the same eval on the trained checkpoint and compare accuracy.
    """


@code
def _eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    trained_deployment = DeploymentConfig(
        model=Qwen3VL_8B(),
        checkpoint=checkpoint,
        app_name="qwen3-vl-8b-grounding-serve",
        served_model_name="qwen3-vl-8b-grounding",
    ).serve()
    print(f"Trained model URL: {trained_deployment.url}")

    print("--- Evaluating trained model... ---")
    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    n_hits = sum(1 for r in trained_eval.rows if r.metadata.get("hit_at_10pct"))
    print(
        f"Trained accuracy (@10%): {n_hits}/{len(trained_eval.rows)} "
        f"({trained_eval.mean:.1%})"
    )


@notebook_only
@code
def _trained_examples():
    for base_r, trained_r in zip(base_eval.rows[:3], trained_eval.rows[:3]):
        label = base_r.metadata["label"]
        b_status = "HIT" if base_r.metadata["hit_at_10pct"] else "MISS"
        t_status = "HIT" if trained_r.metadata["hit_at_10pct"] else "MISS"
        print(f"label={label}")
        print(f"  Base:    [{b_status}] pred={base_r.metadata['pred']} dist={base_r.metadata['distance']:.4f}")
        print(f"  Trained: [{t_status}] pred={trained_r.metadata['pred']} dist={trained_r.metadata['distance']:.4f}")
        print()


@markdown
def _compare_intro():
    """
    ## Results

    Let's compare base vs trained accuracy.
    """


@code
def _compare():
    base_hits = sum(1 for r in base_eval.rows if r.metadata.get("hit_at_10pct"))
    trained_hits = sum(
        1 for r in trained_eval.rows if r.metadata.get("hit_at_10pct")
    )
    total = len(base_eval.rows)
    print(f"Base model:    {base_hits}/{total} ({base_eval.mean:.1%})")
    print(f"Trained model: {trained_hits}/{total} ({trained_eval.mean:.1%})")
    print(f"Delta:         {trained_eval.mean - base_eval.mean:+.1%}")
