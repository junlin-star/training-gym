# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `001_toolathlon_privacy` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "Modal Sandbox",
    "cluster_shape": "1 × 1×H100",
    "summary": "Convert Toolathlon privacy-desensitization tasks to Harbor format, register agent tools, and evaluate Qwen3-8B",
    "difficulty": "Intermediate",
    "order": 15,
    "api_classes": [
        "DatasetConfig",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "ModelDeployment",
        "Qwen3_8B",
        "SglangRecipe",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Toolathlon privacy-desensitization → Harbor eval with Modal Sandboxes

    This tutorial shows how to take an existing agent benchmark —
    [Toolathlon](https://github.com/hkust-nlp/Toolathlon)'s
    **privacy-desensitization** task — convert it into
    [Harbor](https://hub.harborframework.com) format, wire up agent
    tools, and evaluate **Qwen3-8B** on executing the task inside a
    Modal Sandbox.

    The privacy-desensitization task asks an agent to scan a workspace
    of ~27 documents (CSV, JSON, TXT, MD, LOG) for sensitive PII —
    phone numbers, SSNs, emails, credit-card numbers, and IP
    addresses — and replace every occurrence with `/hidden/`.

    What you'll learn:
    1. Convert a Toolathlon task directory into a
       **Harbor-compatible dataset** (instruction + ground-truth
       files).
    2. Register **agent tools** (list directory, read file, write
       file) that run inside a Modal Sandbox.
    3. Build an **agent loop** that drives Qwen3-8B through the
       task using OpenAI-compatible tool calling.
    4. **Score** the agent's output against ground truth with a
       file-level comparison scorer.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run python tutorials/agent/001_toolathlon_privacy/001_toolathlon_privacy.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import json
    import os
    import re
    import tarfile
    import tempfile
    from pathlib import Path

    import modal
    import openai

    from modal_training_gym import (
        DatasetConfig,
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        ModelDeployment,
        Qwen3_8B,
    )
    from modal_training_gym.deploy_recipes import SglangRecipe


@markdown
def _harbor_section():
    """
    ## Step 1 — Convert the Toolathlon task to Harbor format

    Harbor tasks follow a simple convention: each task directory
    contains an `instruction.md` file (the prompt) and any
    supporting files.  We download the Toolathlon repo, extract
    the privacy-desensitization workspace files, and build a
    `DatasetConfig` that packages the instruction plus all 27
    source documents as a single task.

    The ground-truth desensitized files are bundled into the label
    so the scorer can compare against them later.
    """


@code
def _harbor_dataset():
    TOOLATHLON_ARCHIVE = (
        "https://github.com/hkust-nlp/Toolathlon/archive/refs/heads/main.tar.gz"
    )
    TASK_SUBDIR = "Toolathlon-main/tasks/finalpool/privacy-desensitization"

    INSTRUCTION = (
        "You are working in a directory that contains various documents "
        "(CSV, JSON, TXT, MD, LOG files). These files may contain "
        "sensitive personally identifiable information (PII).\n\n"
        "Your task:\n"
        "1. List all files in the /workspace directory.\n"
        "2. Read each file and identify all occurrences of:\n"
        "   - Phone/Fax numbers (any format)\n"
        "   - Social Security Numbers (SSN)\n"
        "   - Email addresses\n"
        "   - Credit card numbers\n"
        "   - IP addresses\n"
        "3. Create a directory called /workspace/desensitized_documents/\n"
        "4. For each file, create a desensitized copy named "
        "`<original_name>_desensitized.<ext>` in that directory.\n"
        "5. Replace every sensitive value with `/hidden/` — keep all "
        "surrounding text intact.\n\n"
        "Do NOT modify information that is not in the list above "
        "(e.g. names, addresses, policy numbers, dates). "
        "Do NOT add any files beyond the desensitized copies."
    )

    def _download_and_extract(tmp: str) -> Path:
        """Download the Toolathlon repo archive and extract the task files."""
        import urllib.request

        archive_path = os.path.join(tmp, "toolathlon.tar.gz")
        urllib.request.urlretrieve(TOOLATHLON_ARCHIVE, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        return Path(tmp) / TASK_SUBDIR

    class PrivacyDesensitizationDataset(DatasetConfig):
        """Single-task dataset wrapping the Toolathlon privacy task."""

        input_key = "messages"
        label_key = "label"
        apply_chat_template = True

        def load(self):
            with tempfile.TemporaryDirectory() as tmp:
                task_dir = _download_and_extract(tmp)
                workspace_dir = task_dir / "initial_workspace"

                # Extract the workspace tar inside initial_workspace/
                ws_tar = workspace_dir / "files.tar.gz"
                with tarfile.open(str(ws_tar), "r:gz") as tar:
                    tar.extractall(str(workspace_dir), filter="data")

                # Collect source file contents
                source_files = {}
                for f in sorted(workspace_dir.iterdir()):
                    if f.is_file() and f.name != "files.tar.gz":
                        source_files[f.name] = f.read_text(encoding="utf-8")

                # Collect ground-truth desensitized files
                gt_dir = task_dir / "groundtruth_workspace"
                gt_tar = gt_dir / "gt_files.tar.gz"
                with tarfile.open(str(gt_tar), "r:gz") as tar:
                    tar.extractall(str(gt_dir), filter="data")
                gt_files = {}
                gt_docs = gt_dir / "desensitized_documents"
                for f in sorted(gt_docs.iterdir()):
                    if f.is_file():
                        gt_files[f.name] = f.read_text(encoding="utf-8")

                return [
                    {
                        "instruction": INSTRUCTION,
                        "source_files": source_files,
                        "ground_truth_files": gt_files,
                    }
                ]

        def prepare(self, path: str, eval_paths: dict[str, str] | None = None):
            from datasets import Dataset

            rows = self.load()
            train_rows = [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a privacy compliance agent. Use the "
                                "provided tools to read, process, and write files. "
                                "Be thorough — scan every file in the workspace."
                            ),
                        },
                        {"role": "user", "content": rows[0]["instruction"]},
                    ],
                    "label": json.dumps(
                        {"ground_truth_files": rows[0]["ground_truth_files"]}
                    ),
                }
            ]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Dataset.from_list(train_rows).to_parquet(path)
            if eval_paths:
                for eval_path in eval_paths.values():
                    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                    Dataset.from_list(train_rows).to_parquet(eval_path)

    dataset = PrivacyDesensitizationDataset()
    print(f"Dataset has {len(dataset.load())} task(s)")
    task = dataset.load()[0]
    print(f"Source files: {sorted(task['source_files'].keys())[:5]}...")
    print(f"Ground-truth files: {len(task['ground_truth_files'])} desensitized documents")


@markdown
def _tools_section():
    """
    ## Step 2 — Register agent tools

    We register three tools that the agent can call:

    | Tool | Description |
    |------|-------------|
    | `list_directory` | `ls -1` inside the sandbox |
    | `read_file` | `cat` a file from the sandbox |
    | `write_file` | Write content to a file in the sandbox |

    Each tool runs a command (or writes via the filesystem API)
    inside a Modal Sandbox. The `dispatch_tool` function routes
    calls by name.
    """


@code
def _define_tools():
    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": (
                    "List the contents of a directory. Returns one "
                    "entry per line."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the directory.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the full contents of a text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write content to a file, creating parent "
                    "directories as needed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path for the file.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The text content to write.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]

    def dispatch_tool(sb, name: str, arguments: str) -> str:
        args = json.loads(arguments)
        if name == "list_directory":
            proc = sb.exec("ls", "-1", args["path"])
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
            return stdout if proc.returncode == 0 else f"Error: {stderr}"

        elif name == "read_file":
            proc = sb.exec("cat", args["path"])
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
            return stdout if proc.returncode == 0 else f"Error: {stderr}"

        elif name == "write_file":
            dir_path = os.path.dirname(args["path"])
            if dir_path:
                proc = sb.exec("mkdir", "-p", dir_path)
                proc.wait()
            sb.filesystem.write_text(args["content"], args["path"])
            return f"Wrote {len(args['content'])} chars to {args['path']}"

        return f"Unknown tool: {name}"


@markdown
def _deploy_section():
    """
    ## Step 3 — Deploy Qwen3-8B

    We deploy Qwen3-8B via `DeploymentConfig.serve()` with the
    `qwen25` tool-call parser so the model emits structured tool
    calls. We then point the OpenAI SDK at the self-hosted
    endpoint.
    """


@code
def _deploy_model():
    recipe = SglangRecipe(
        extra_server_args={"--tool-call-parser": "qwen25"},
    )
    deployment = DeploymentConfig(
        model=Qwen3_8B(),
        recipe=recipe,
    ).serve()
    deployment.wait_until_ready()
    print(f"Model URL: {deployment.url}")

    client = openai.OpenAI(
        base_url=f"{deployment.url}/v1",
        api_key="not-needed",
    )


@markdown
def _sandbox_section():
    """
    ## Step 4 — Create a sandbox and load workspace files

    We spin up a long-lived Sandbox and write all 27 source
    documents from the Toolathlon task into `/workspace/`.
    This mirrors the environment the original benchmark uses.
    """


@code
def _create_sandbox():
    sandbox_app = modal.App.lookup(
        "toolathlon-privacy-tutorial", create_if_missing=True
    )

    sandbox = modal.Sandbox.create(
        "sleep", "infinity",
        app=sandbox_app,
        image=modal.Image.debian_slim(python_version="3.12"),
        timeout=600,
    )

    for filename, content in task["source_files"].items():
        sandbox.filesystem.write_text(content, f"/workspace/{filename}")

    proc = sandbox.exec("ls", "-1", "/workspace")
    stdout = proc.stdout.read()
    proc.wait()
    print(f"Loaded {len(stdout.strip().splitlines())} files into sandbox /workspace/")


@markdown
def _agent_loop_section():
    """
    ## Step 5 — Run the agent loop

    The agent loop drives Qwen3-8B through the task. On each
    iteration the model either:
    - Calls a tool → we execute it in the sandbox and feed back
      the result.
    - Produces a final text response → we stop.

    We set `enable_thinking=False` so Qwen3 skips its internal
    chain-of-thought and produces clean tool calls. The agent
    has up to 100 iterations to process all 27 files.
    """


@code
def _agent_loop():
    MODEL = deployment.deployment_config.served_model_name
    MAX_ITERATIONS = 100

    messages = [
        {
            "role": "system",
            "content": (
                "You are a privacy compliance agent. Use the provided tools to "
                "scan documents in /workspace/ for PII (phone numbers, SSNs, "
                "emails, credit card numbers, IP addresses). Create desensitized "
                "copies in /workspace/desensitized_documents/ with each sensitive "
                "value replaced by /hidden/. Process every file thoroughly."
            ),
        },
        {
            "role": "user",
            "content": INSTRUCTION,
        },
    ]

    print("Starting agent loop...\n")
    tool_call_count = 0

    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=8192,
            tools=TOOL_DEFINITIONS,
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            print(f"\nAgent finished after {i+1} iterations, {tool_call_count} tool calls")
            print(f"Final response:\n{choice.message.content[:500]}")
            break

        messages.append(choice.message)

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_call_count += 1
                result = dispatch_tool(sandbox, tc.function.name, tc.function.arguments)
                if tc.function.name == "write_file":
                    print(f"  [{tool_call_count}] {tc.function.name}({json.loads(tc.function.arguments).get('path', '?')})")
                elif len(result) > 200:
                    print(f"  [{tool_call_count}] {tc.function.name}(...) → {len(result)} chars")
                else:
                    print(f"  [{tool_call_count}] {tc.function.name}(...) → {result[:80]}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
    else:
        print(f"Reached max iterations ({MAX_ITERATIONS}).")


@markdown
def _scoring_section():
    """
    ## Step 6 — Score the agent output

    We read back the files the agent wrote to
    `/workspace/desensitized_documents/` and compare them against
    the ground-truth files from Toolathlon. Scoring mirrors the
    original evaluation script: strip all whitespace, then check
    for an exact match. The final score is the fraction of files
    that match.
    """


@code
def _score():
    _WHITESPACE_RE = re.compile(r"\s+")

    def normalize(text: str) -> str:
        return _WHITESPACE_RE.sub("", text).strip()

    ground_truth = task["ground_truth_files"]

    proc = sandbox.exec("ls", "-1", "/workspace/desensitized_documents")
    stdout = proc.stdout.read()
    proc.wait()
    agent_files = [f for f in stdout.strip().splitlines() if f]

    print(f"Agent produced {len(agent_files)} files, expected {len(ground_truth)}")

    matched = 0
    mismatched = []
    missing = []

    for gt_name, gt_content in sorted(ground_truth.items()):
        if gt_name not in agent_files:
            missing.append(gt_name)
            continue
        proc = sandbox.exec("cat", f"/workspace/desensitized_documents/{gt_name}")
        agent_content = proc.stdout.read()
        proc.stderr.read()
        proc.wait()
        if normalize(agent_content) == normalize(gt_content):
            matched += 1
        else:
            mismatched.append(gt_name)

    total = len(ground_truth)
    score = matched / total if total else 0.0
    print(f"\nScore: {matched}/{total} = {score:.2%}")
    if missing:
        print(f"Missing files: {missing}")
    if mismatched:
        print(f"Mismatched files: {mismatched}")


@markdown
def _eval_section():
    """
    ## Step 7 — Structured evaluation with EvalConfig

    We can also plug the scoring into the training-gym
    `EvalConfig` framework so results are persisted to the
    metadata store and visible on the dashboard.

    The `eval_fn` spins up a fresh sandbox per example, runs
    the full agent loop, and returns a score.
    """


@code
def _eval_config():
    def agent_eval_fn(dep: ModelDeployment, example: dict) -> EvalRowResult:
        _model = dep.deployment_config.served_model_name
        _client = openai.OpenAI(
            base_url=f"{dep.url}/v1",
            api_key="not-needed",
        )

        source_files = example["source_files"]
        gt_files = example["ground_truth_files"]

        eval_app = modal.App.lookup(
            "toolathlon-privacy-eval", create_if_missing=True
        )
        sb = modal.Sandbox.create(
            "sleep", "infinity",
            app=eval_app,
            image=modal.Image.debian_slim(python_version="3.12"),
            timeout=600,
        )

        try:
            for name, content in source_files.items():
                sb.filesystem.write_text(content, f"/workspace/{name}")

            msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are a privacy compliance agent. Use tools to scan "
                        "/workspace/ for PII and write desensitized copies to "
                        "/workspace/desensitized_documents/."
                    ),
                },
                {"role": "user", "content": example["instruction"]},
            ]

            _tc_count = 0
            for _ in range(100):
                resp = _client.chat.completions.create(
                    model=_model,
                    max_tokens=8192,
                    tools=TOOL_DEFINITIONS,
                    messages=msgs,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                )
                ch = resp.choices[0]
                if ch.finish_reason == "stop":
                    break
                msgs.append(ch.message)
                if ch.message.tool_calls:
                    for tc in ch.message.tool_calls:
                        _tc_count += 1
                        result = dispatch_tool(
                            sb, tc.function.name, tc.function.arguments
                        )
                        msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result,
                            }
                        )

            ws_re = re.compile(r"\s+")
            matched = 0
            for gt_name, gt_content in gt_files.items():
                proc = sb.exec(
                    "cat",
                    f"/workspace/desensitized_documents/{gt_name}",
                )
                agent_content = proc.stdout.read()
                proc.stderr.read()
                proc.wait()
                if ws_re.sub("", agent_content).strip() == ws_re.sub(
                    "", gt_content
                ).strip():
                    matched += 1

            score = matched / len(gt_files) if gt_files else 0.0
        finally:
            sb.terminate()

        return EvalRowResult(
            score=score,
            response=f"{_tc_count} tool calls, {matched}/{len(gt_files)} files matched",
            metadata={"matched": matched, "total": len(gt_files)},
        )

    eval_config = EvalConfig(
        dataset=dataset,
        eval_fn=agent_eval_fn,
    )
    print("Running structured evaluation...")
    eval_result = eval_config.evaluate(deployment, debug=True)
    print(f"Mean score: {eval_result.mean:.2%}")


@markdown
def _cleanup_section():
    """
    ## Clean up

    Terminate the sandbox so it stops billing.
    """


@code
def _cleanup():
    sandbox.terminate()
    print("Sandbox terminated.")


@markdown
def _next_steps():
    """
    ## Next steps

    This tutorial demonstrated the full pipeline from an external
    benchmark to a scored agent evaluation on Modal:

    - **Converted** Toolathlon's privacy-desensitization task into
      a Harbor-compatible `DatasetConfig`.
    - **Registered tools** (list, read, write) backed by Modal
      Sandbox execution.
    - **Evaluated** Qwen3-8B's ability to perform the PII
      desensitization task autonomously.

    Ideas to extend this:
    - **Train with RL** — use the file-match score as a reward
      signal with `SlimeRecipe` and `custom_rm_function`.
    - **Add more tools** — `run_command` for regex testing,
      `search_file` for grep-like pattern search.
    - **Scale to more tasks** — Toolathlon has many task categories
      (code generation, data analysis, etc.) that can each be
      converted to Harbor format.
    - **Try larger models** — `Qwen3_32B` may handle the nuances
      of different PII formats more reliably.
    """
