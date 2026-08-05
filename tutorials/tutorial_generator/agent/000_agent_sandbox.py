# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `000_agent_sandbox` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "Modal Sandbox",
    "cluster_shape": "1 × 1×H100",
    "summary": "Build an LLM agent harness with a self-hosted model and Modal Sandbox tool execution",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
        "Qwen3_8B",
        "endpoint_chat_message",
        "wait_for_server_url",
    ],
    "required_modal_secrets": [],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Build an agent harness with a self-hosted model + Modal Sandboxes

    This tutorial builds an LLM agent loop from scratch using a
    **self-hosted model** served on Modal. The agent can use two
    tools — **list a directory** and **read a file** — and every
    tool call executes inside a
    [Modal Sandbox](https://modal.com/docs/guide/sandbox), an
    isolated container with its own filesystem.

    What you'll learn:
    1. Serve Qwen3-8B with a custom Modal `@app.server` and get an
       OpenAI-compatible endpoint.
    2. Call that endpoint with `endpoint_chat_message` (including tools).
    3. Create a Modal Sandbox with files pre-loaded via
       `filesystem.write_text`.
    4. Define tools that run shell commands inside the sandbox
       with `sandbox.exec`.
    5. Wire everything into an agent loop with tool calling.

    The entire stack runs on Modal — model serving, tool execution,
    and the sandbox — so you control cost, latency, and data privacy.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run tutorials/misc/000_agent_sandbox/000_agent_sandbox.py
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
    import json
    import subprocess
    import time
    import urllib.error
    import urllib.request

    import modal

    from modal_training_gym import (
        Qwen3_8B,
        endpoint_chat_message,
        wait_for_server_url,
    )


@markdown
def _tools_section():
    """
    ## Define the tools

    We define two tools and a dispatcher function. Each tool runs a
    command inside a sandbox via `sandbox.exec` and captures
    stdout/stderr. The tool definitions follow the OpenAI
    function-calling schema.

    The dispatcher takes the sandbox as an argument so it can be
    called from the agent loop after the sandbox is created.
    """


@code
def _define_tools():
    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": (
                    "List the contents of a directory. Returns one entry per "
                    "line. Directories have a trailing slash."
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
    ]

    def dispatch_tool(sb, name: str, arguments: str) -> str:
        args = json.loads(arguments)
        if name == "list_directory":
            proc = sb.exec("ls", "-1F", args["path"])
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

        return f"Unknown tool: {name}"


@markdown
def _serve_model_section():
    """
    ## Serve the model

    Qwen3-8B is not in the managed Endpoint catalog, so we launch SGLang
    directly with `@app.server`. The server exposes an **OpenAI-compatible**
    `/v1/chat/completions` endpoint used by `endpoint_chat_message`.

    We pass `--tool-call-parser qwen25` so the server parses Qwen3's tool-call
    format into structured `tool_calls` in the response. Without this, the
    model emits tool calls as raw text.

    Custom `@app.server` endpoints here use `unauthenticated=False`. Run
    `training-gym set-proxy-auth` or export `MODAL_KEY`/`MODAL_SECRET` before
    calling `wait_for_server_url` / `endpoint_chat_message` with
    `proxy_auth=True`.
    """


@code
def _serve_model():
    MODEL_ID = Qwen3_8B().model_name
    SERVER_APP_NAME = "gym-qwen3-8b-agent"
    SERVER_PORT = 8000
    SERVER_STARTUP_TIMEOUT = 20 * 60

    server_image = (
        modal.Image.from_registry("lmsysorg/sglang:v0.5.12")
        .entrypoint([])
        .run_commands("rm -rf /root/.cache/huggingface")
        .env({"HF_HUB_CACHE": "/root/.cache/huggingface"})
    )

    def serve_model() -> str:
        app = modal.App(SERVER_APP_NAME)

        @app.server(
            image=server_image,
            gpu="H100",
            volumes={
                "/root/.cache/huggingface": modal.Volume.from_name(
                    "huggingface-cache", create_if_missing=True
                )
            },
            port=SERVER_PORT,
            startup_timeout=SERVER_STARTUP_TIMEOUT,
            scaledown_window=10 * 60,
            exit_grace_period=25,
            target_concurrency=4,
            unauthenticated=False,
            serialized=True,
        )
        class ModelServer:
            @modal.enter()
            def start(self):
                command = [
                    "python",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    MODEL_ID,
                    "--served-model-name",
                    MODEL_ID,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(SERVER_PORT),
                    "--mem-fraction-static",
                    "0.82",
                    "--context-length",
                    "32768",
                    "--tool-call-parser",
                    "qwen25",
                    "--trust-remote-code",
                ]
                print(" ".join(command), flush=True)
                self.proc = subprocess.Popen(command)

                deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
                health = f"http://127.0.0.1:{SERVER_PORT}/health"
                while time.monotonic() < deadline:
                    if self.proc.poll() is not None:
                        raise RuntimeError(
                            f"SGLang exited with code {self.proc.returncode}"
                        )
                    try:
                        with urllib.request.urlopen(health, timeout=5) as response:
                            if response.status == 200:
                                return
                    except (urllib.error.URLError, TimeoutError, OSError):
                        pass
                    time.sleep(2)
                raise TimeoutError(f"SGLang not healthy at {health}")

            @modal.exit()
            def stop(self):
                process = getattr(self, "proc", None)
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)

        with modal.enable_output():
            app.deploy()
        return wait_for_server_url(
            ModelServer, label="Qwen3-8B agent server", proxy_auth=True
        )

    model_url = serve_model()
    print(f"Model URL: {model_url}")


@markdown
def _sandbox_section():
    """
    ## Create a sandbox with sample files

    We spin up a long-lived Sandbox running `sleep infinity` so it
    stays alive while the agent issues commands. After creation we
    write a small project tree into the sandbox's filesystem.
    """


@code
def _create_sandbox():
    sandbox_app = modal.App.lookup("agent-sandbox-tutorial", create_if_missing=True)

    sandbox = modal.Sandbox._experimental_create(
        "sleep", "infinity",
        app=sandbox_app,
        image=modal.Image.debian_slim(python_version="3.12"),
        timeout=600,
    )

    FILES = {
        "/repo/README.md": (
            "# My Project\n\n"
            "A small Python utility that computes Fibonacci numbers.\n\n"
            "## Usage\n\n"
            "```bash\n"
            "python fib.py 10\n"
            "```\n"
        ),
        "/repo/fib.py": (
            "import sys\n\n\n"
            "def fibonacci(n: int) -> list[int]:\n"
            '    """Return the first n Fibonacci numbers."""\n'
            "    if n <= 0:\n"
            "        return []\n"
            "    seq = [0, 1]\n"
            "    while len(seq) < n:\n"
            "        seq.append(seq[-1] + seq[-2])\n"
            "    return seq[:n]\n\n\n"
            'if __name__ == "__main__":\n'
            "    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10\n"
            "    print(fibonacci(count))\n"
        ),
        "/repo/tests/test_fib.py": (
            "from fib import fibonacci\n\n\n"
            "def test_empty():\n"
            "    assert fibonacci(0) == []\n\n\n"
            "def test_one():\n"
            "    assert fibonacci(1) == [0]\n\n\n"
            "def test_ten():\n"
            "    result = fibonacci(10)\n"
            "    assert len(result) == 10\n"
            "    assert result[-1] == 34\n"
        ),
        "/repo/pyproject.toml": (
            '[project]\nname = "fib"\nversion = "0.1.0"\n'
            'requires-python = ">=3.12"\n'
        ),
    }

    for path, content in FILES.items():
        sandbox.filesystem.write_text(content, path)

    print(f"Sandbox created: {sandbox.object_id}")


@markdown
def _agent_loop_section():
    """
    ## The agent loop

    The loop uses the OpenAI-compatible tool-calling protocol via
    `endpoint_chat_message`:
    1. Send messages + tool definitions to the self-hosted model.
    2. If the model returns `tool_calls`, execute each one in the
       sandbox and append the results as `tool` messages.
    3. Repeat until the model produces a final text response.

    We cap iterations at 10 to avoid runaway loops. We also pass
    `enable_thinking=False` in `chat_template_kwargs` so Qwen3
    skips its internal chain-of-thought block and responds
    directly — this keeps tool-call parsing clean.
    """


@code
def _agent_loop():
    MAX_ITERATIONS = 10

    messages = [
        {
            "role": "user",
            "content": (
                "Explore the /repo directory. List what files exist, read "
                "each one, and give me a summary of the project — what it "
                "does, how the code is structured, and whether the tests "
                "look correct."
            ),
        },
    ]

    print("Starting agent loop...\n")

    for i in range(MAX_ITERATIONS):
        message = endpoint_chat_message(
            model_url,
            model=MODEL_ID,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            proxy_auth=True,
        )
        messages.append(message)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            content = message.get("content") or message.get("reasoning_content") or ""
            print(f"Agent response:\n{content}")
            break

        for tool_call in tool_calls:
            function = tool_call["function"]
            print(
                f"  [{i + 1}] Calling {function['name']}"
                f"({function.get('arguments', '{}')})"
            )
            result = dispatch_tool(
                sandbox, function["name"], function.get("arguments") or "{}"
            )
            print(f"       → {len(result)} chars returned")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                }
            )

    else:
        print("Reached max iterations without a final response.")


@markdown
def _cleanup_section():
    """
    ## Clean up

    Terminate the sandbox so it doesn't keep running (and billing)
    after we're done.
    """


@code
def _cleanup():
    sandbox.terminate()
    print("Sandbox terminated.")


@markdown
def _next_steps():
    """
    ## Next steps

    This tutorial showed how to combine a self-hosted model with
    sandbox tool execution — no external API keys required. The
    model runs on Modal, the tools run on Modal, and everything
    is under your control.

    Ideas to extend this:
    - **Add a `run_command` tool** so the agent can execute
      arbitrary shell commands (run tests, install packages).
    - **Add a `write_file` tool** using
      `sandbox.filesystem.write_text` so the agent can modify
      code.
    - **Swap models** — try `Qwen3_8B` for harder tasks, or
      `Qwen3_4B` for lower cost.
    - **Snapshot the filesystem** with
      `sandbox.snapshot_filesystem()` to create a reusable
      `modal.Image` from the sandbox state.
    """
