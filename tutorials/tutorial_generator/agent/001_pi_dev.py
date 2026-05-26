# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `001_pi_dev` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "pi.dev + Modal Sandbox",
    "cluster_shape": "1 × 1×H100",
    "summary": "Wikipedia Speedrun — pi.dev agent with custom tools and self-hosted model evaluation",
    "difficulty": "Intermediate",
    "order": 20,
    "api_classes": [
        "DeploymentConfig",
        "Qwen3_8B",
        "SglangRecipe",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Wikipedia Speedrun with pi.dev

    This tutorial builds a **Wikipedia Speedrun** evaluator: an agent
    navigates from one Wikipedia article to another by clicking links,
    trying to reach the target in as few clicks as possible.

    It showcases how to use [pi.dev](https://pi.dev) — a coding-agent
    SDK — with the training gym:
    1. **Self-host a model** with `DeploymentConfig` (OpenAI-compatible
       endpoint, no external API keys).
    2. **Generate an eval dataset** of Wikipedia (start, target) pairs
       using the MediaWiki API.
    3. **Drive a pi.dev agent via RPC** from Python — pi.dev handles
       tool execution, context management, and retries while our
       code sends prompts and reads structured events.
    4. **Score the runs** — success rate and average clicks.

    The agent uses pi's built-in `bash` tool to call a Python helper
    script that wraps the Wikipedia API. The entire flow is
    orchestrated from Python via pi.dev's JSONL RPC protocol.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run modal run -d tutorials/agent/001_pi_dev/001_pi_dev.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import json
    import queue
    import threading
    import urllib.parse
    import urllib.request

    import modal

    from modal_training_gym import (
        DeploymentConfig,
        Qwen3_8B,
    )
    from modal_training_gym.deploy_recipes import SglangRecipe


@markdown
def _dataset_section():
    """
    ## Generate the eval dataset

    We build (start, target) pairs that are **exactly 2 clicks
    apart** so every pair is solvable. The algorithm:
    1. Pick well-known "seed" articles with many outgoing links.
    2. Follow one link from the seed to get a hop-1 article.
    3. Follow one link from hop-1 to get a hop-2 article.
    4. The pair is (seed → hop-2) with a known 2-click path
       through hop-1.

    The agent doesn't know the intermediate hop — it has to
    discover a route on its own.
    """


@code
def _generate_dataset():
    def get_wiki_links(title: str) -> list[str]:
        url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "titles": title,
                "prop": "links",
                "pllimit": "max",
                "plnamespace": "0",
                "format": "json",
            })
        )
        req = urllib.request.Request(url, headers={"User-Agent": "training-gym-tutorial/1.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        pages = list(data["query"]["pages"].values())
        return sorted(l["title"] for l in pages[0].get("links", []))

    SEEDS = [
        "Python_(programming_language)",
        "Machine_learning",
        "San_Francisco",
        "Albert_Einstein",
        "World_Wide_Web",
    ]

    EVAL_DATASET = []
    for seed in SEEDS:
        links_1 = get_wiki_links(seed)
        if len(links_1) < 20:
            continue
        hop1 = links_1[len(links_1) // 3]
        links_2 = get_wiki_links(hop1)
        if len(links_2) < 10:
            continue
        target = links_2[len(links_2) // 3]
        EVAL_DATASET.append({
            "start": seed,
            "target": target,
            "known_path": [seed, hop1, target],
        })

    print(f"Generated {len(EVAL_DATASET)} eval pairs:")
    for pair in EVAL_DATASET:
        print(f"  {pair['start']} → {pair['target']}")
        print(f"    known path: {' → '.join(pair['known_path'])}")


@markdown
def _deploy_section():
    """
    ## Deploy the model

    Same pattern as the previous tutorial: serve Qwen3-8B with
    sglang and enable the tool-call parser so the agent gets
    structured tool calls.  The `--reasoning-parser qwen3` flag
    strips Qwen3's `<think>` blocks from the API response — the
    model still reasons internally (which improves tool-call
    quality at 8B scale) but clients see clean output.
    """


@code
def _deploy_model():
    recipe = SglangRecipe(
        extra_server_args={
            "--tool-call-parser": "qwen25",
            "--reasoning-parser": "qwen3",
        },
    )
    deployment = DeploymentConfig(
        model=Qwen3_8B(),
        recipe=recipe,
    ).serve()
    deployment.wait_until_ready()
    print(f"Model URL: {deployment.url}")


@markdown
def _sandbox_section():
    """
    ## Create the sandbox

    The sandbox image includes Node.js and the pi.dev CLI
    (`@earendil-works/pi-coding-agent`), both installed at
    image build time.

    After the sandbox starts we write three things:
    - **`models.json`** — points pi.dev at our self-hosted model
      as an OpenAI-compatible provider.
    - **`auth.json`** — pi.dev requires auth config even when the
      endpoint doesn't need a real key.
    - **`wiki-tools.ts`** — a pi.dev extension that registers
      `wiki_links` and `wiki_navigate` as proper tools using
      `pi.registerTool()`. pi auto-discovers extensions from
      `~/.pi/agent/extensions/`.
    """


@code
def _create_sandbox():
    sandbox_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl", "ca-certificates")
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "apt-get install -y nodejs",
            "npm install -g @earendil-works/pi-coding-agent",
        )
    )

    sandbox_app = modal.App.lookup("wiki-speedrun-tutorial", create_if_missing=True)
    sandbox = modal.Sandbox.create(
        "sleep", "infinity",
        app=sandbox_app,
        image=sandbox_image,
        timeout=3600,
    )

    served_model = deployment.deployment_config.served_model_name
    models_json = json.dumps({
        "providers": {
            "gym": {
                "baseUrl": f"{deployment.url}/v1",
                "api": "openai-completions",
                "apiKey": "not-needed",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsTools": True,
                    "maxTokensField": "max_tokens",
                },
                "models": [{
                    "id": served_model,
                    "name": "Qwen3-8B (self-hosted)",
                    "contextWindow": 32768,
                    "maxTokens": 16384,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                }],
            },
        },
    })
    sandbox.exec("mkdir", "-p", "/root/.pi/agent").wait()
    sandbox.filesystem.write_text(models_json, "/root/.pi/agent/models.json")
    sandbox.filesystem.write_text(
        json.dumps({"gym": {"type": "api_key", "key": "not-needed"}}),
        "/root/.pi/agent/auth.json",
    )

    WIKI_EXTENSION = "\n".join([
        'import { Type } from "typebox";',
        "",
        "async function getWikiLinks(title) {",
        "  const params = new URLSearchParams({",
        '    action: "query", titles: title, prop: "links",',
        '    pllimit: "max", plnamespace: "0", format: "json",',
        "  });",
        '  const url = "https://en.wikipedia.org/w/api.php?" + params.toString();',
        "  const res = await fetch(url, {",
        '    headers: { "User-Agent": "training-gym-tutorial/1.0" },',
        "  });",
        "  const data = await res.json();",
        "  const pages = Object.values(data.query.pages);",
        "  const links = (pages[0].links || []).map((l) => l.title).sort();",
        "  return links.slice(0, 30);",
        "}",
        "",
        "export default function (pi) {",
        "  pi.registerTool({",
        '    name: "wiki_links",',
        '    label: "Wiki Links",',
        "    description:",
        '      "Get the list of outgoing links on a Wikipedia article. " +',
        '      "This is FREE and does not cost a click. Use it to scout pages.",',
        "    parameters: Type.Object({",
        '      title: Type.String({ description: "Wikipedia article title, e.g. Machine_learning" }),',
        "    }),",
        "    async execute(_id, params) {",
        "      const links = await getWikiLinks(params.title);",
        "      return {",
        '        content: [{ type: "text", text: JSON.stringify(links) }],',
        "        details: {},",
        "      };",
        "    },",
        "  });",
        "",
        "  pi.registerTool({",
        '    name: "wiki_navigate",',
        '    label: "Wiki Navigate",',
        "    description:",
        '      "Navigate to a Wikipedia article. COSTS 1 CLICK. " +',
        '      "Returns whether you reached the target and the links on the page.",',
        "    parameters: Type.Object({",
        '      title: Type.String({ description: "Wikipedia article title to navigate to" }),',
        '      target: Type.String({ description: "The target article you are trying to reach" }),',
        "    }),",
        "    async execute(_id, params) {",
        "      const links = await getWikiLinks(params.title);",
        '      const normalize = (s) => s.toLowerCase().replace(/_/g, " ");',
        "      const reached = normalize(params.title) === normalize(params.target);",
        "      const result = { on: params.title, reached, links };",
        "      return {",
        '        content: [{ type: "text", text: JSON.stringify(result) }],',
        "        details: {},",
        "      };",
        "    },",
        "  });",
        "}",
    ])
    sandbox.exec("mkdir", "-p", "/root/.pi/agent/extensions").wait()
    sandbox.filesystem.write_text(WIKI_EXTENSION, "/root/.pi/agent/extensions/wiki-tools.ts")

    print(f"Sandbox created: {sandbox.object_id}")


@markdown
def _rpc_section():
    """
    ## Drive the agent via RPC

    pi.dev's RPC mode (`pi --mode rpc`) exposes a JSONL protocol
    over stdin/stdout. Our Python wrapper:

    1. Starts `pi` inside the sandbox with `sandbox.exec`.
    2. Sends a `prompt` command via stdin.
    3. Reads events from stdout — auto-approving tool execution
       dialogs and counting `wiki_navigate` calls.
    4. Stops when the agent finishes (`agent_end` event).

    The agent calls our custom `wiki_links` and `wiki_navigate`
    tools (registered via the extension). pi handles the tool
    execution loop, retries, and context management.
    """


@code
def _define_rpc_runner():
    def send_rpc(proc, msg: dict) -> None:
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        proc.stdin.drain()

    HANDLED_EVENTS = {
        "turn_start", "message_start", "message_update", "message_end",
        "extension_error", "extension_ui_request",
        "tool_execution_start", "tool_execution_end", "agent_end",
    }
    STALL_TIMEOUT = 120

    def run_wiki_game(
        sb, model_provider: str, model_id: str,
        start: str, target: str, max_clicks: int = 8,
    ) -> dict:
        proc = sb.exec(
            "pi", "--mode", "rpc", "--no-session",
            "--provider", model_provider, "--model", model_id,
        )

        event_q: queue.Queue = queue.Queue()
        def _reader():
            for line in proc.stdout:
                event_q.put(line)
            event_q.put(None)
        threading.Thread(target=_reader, daemon=True).start()

        prompt = (
            f'Wikipedia Speedrun: navigate from "{start}" to "{target}".\n\n'
            f"You have two tools:\n"
            f"  wiki_links(title) → returns JSON array of links on that page (FREE)\n"
            f'  wiki_navigate(title, target="{target}") → navigates to title '
            f'(COSTS 1 CLICK), returns JSON with "reached" (bool) and "links" (array)\n\n'
            f"Rules:\n"
            f"- Each wiki_navigate call costs 1 click (max {max_clicks})\n"
            f"- wiki_links calls are free — use them to scout\n"
            f"- Minimize total clicks\n"
            f"- Stop immediately when wiki_navigate returns reached=true\n\n"
            f'Start by calling: wiki_links(title="{start}")'
        )
        send_rpc(proc, {"id": "1", "type": "prompt", "message": prompt})

        clicks = 0
        reached = False
        turn = 0
        timed_out = False

        while True:
            try:
                raw = event_q.get(timeout=STALL_TIMEOUT)
            except queue.Empty:
                print(f"    ⚠ no events for {STALL_TIMEOUT}s — aborting")
                timed_out = True
                break
            if raw is None:
                break

            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "turn_start":
                turn += 1
                print(f"  Turn {turn}")

            if etype == "message_start":
                msg = event.get("message", {})
                if msg.get("role") == "assistant":
                    stop = msg.get("stopReason", "")
                    model = msg.get("model", "")
                    if model:
                        print(f"    model: {model}")
                    if stop and stop != "stop":
                        print(f"    ⚠ stopReason: {stop}")

            if etype == "message_update":
                pass

            if etype == "message_end":
                msg = event.get("message", {})
                stop = msg.get("stopReason", "")
                usage = msg.get("usage", {})
                tokens_in = usage.get("input", 0)
                tokens_out = usage.get("output", 0)
                if tokens_in or tokens_out:
                    print(f"    tokens: {tokens_in} in / {tokens_out} out")
                if stop == "length":
                    print(f"    ⚠ hit token limit — response truncated")

            if etype == "extension_error":
                print(f"    ✗ error: {event.get('message', event)}")

            if etype == "extension_ui_request":
                send_rpc(proc, {
                    "type": "extension_ui_response",
                    "id": event["id"],
                    "value": (
                        event.get("options", ["Allow"])[0]
                        if event.get("method") == "select"
                        else True
                    ),
                })

            if etype == "tool_execution_start":
                tool_name = event.get("toolName", "")
                args = event.get("args", {})
                if tool_name == "wiki_navigate":
                    clicks += 1
                    print(f"    → navigate (click {clicks}/{max_clicks}): {args.get('title', '?')}")
                elif tool_name == "wiki_links":
                    print(f"    → links: {args.get('title', '?')}")
                else:
                    cmd = str(event.get("input", event.get("command", "")))
                    print(f"    → {tool_name or 'tool'}: {cmd[:120]}")

            if etype == "tool_execution_end":
                tool_name = event.get("toolName", "")
                result = event.get("result", {})
                content_parts = result.get("content", []) if isinstance(result, dict) else []
                output = content_parts[0].get("text", "") if content_parts else str(result)
                if tool_name == "wiki_navigate":
                    try:
                        parsed = json.loads(output)
                        if parsed.get("reached"):
                            reached = True
                            print(f"    ✓ reached target!")
                        else:
                            on_page = parsed.get("on", "?")
                            n_links = len(parsed.get("links", []))
                            print(f"    ✗ on: {on_page} ({n_links} links)")
                    except (json.JSONDecodeError, AttributeError):
                        if '"reached": true' in output or '"reached":true' in output:
                            reached = True
                            print(f"    ✓ reached target!")
                        else:
                            print(f"    ✗ not reached")

            if etype not in HANDLED_EVENTS:
                print(f"    [{etype}] {json.dumps(event)[:200]}")

            if clicks >= max_clicks and etype == "tool_execution_end":
                print(f"    ⚠ max clicks reached, aborting")
                send_rpc(proc, {"id": "2", "type": "abort"})
                break

            if etype == "agent_end":
                break

        try:
            proc.stdin.write_eof()
            proc.stdin.drain()
        except Exception:
            pass

        for err_line in proc.stderr:
            err_line = err_line.strip()
            if err_line:
                print(f"  [stderr] {err_line}")

        proc.wait()

        return {"success": reached, "clicks": clicks, "timed_out": timed_out}


@markdown
def _eval_section():
    """
    ## Run the evaluation

    For each (start, target) pair we launch a fresh pi.dev RPC
    session and measure clicks to reach the target.
    """


@code
def _run_eval():
    served_model = deployment.deployment_config.served_model_name
    results = []

    for pair in EVAL_DATASET:
        print(f"\n{'='*60}")
        print(f"Start:  {pair['start']}")
        print(f"Target: {pair['target']}")
        print(f"Known:  {' → '.join(pair['known_path'])} (2 clicks)")

        result = run_wiki_game(
            sandbox, "gym", served_model,
            pair["start"], pair["target"],
        )
        results.append(result)

        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"Agent:  {status} in {result['clicks']} clicks")


@markdown
def _results_section():
    """
    ## Results
    """


@code
def _print_results():
    successes = [r for r in results if r["success"]]
    print(f"\nSuccess rate: {len(successes)}/{len(results)}")
    if successes:
        avg = sum(r["clicks"] for r in successes) / len(successes)
        print(f"Avg clicks (successful): {avg:.1f}")
    avg_all = sum(r["clicks"] for r in results) / max(len(results), 1)
    print(f"Avg clicks (all):        {avg_all:.1f}")
    print(f"Known optimal:           2.0")


@markdown
def _cleanup_section():
    """
    ## Clean up
    """


@code
def _cleanup():
    sandbox.terminate()
    print("Sandbox terminated.")


@markdown
def _next_steps():
    """
    ## Next steps

    This tutorial showed how to drive pi.dev's agent loop from
    Python via RPC — the model ran on Modal, tools ran in a
    sandbox, and pi.dev handled the agentic orchestration
    (tool calls, retries, context).

    Ideas to extend this:
    - **Harder pairs** — generate 3-hop or 4-hop pairs for more
      challenging navigation.
    - **Compare models** — run the same dataset against Qwen3-4B
      vs Qwen3-32B to measure how model size affects navigation.
    - **Add a summary tool** — let the agent read a page's
      opening paragraph for more informed routing.
    - **Train with RL** — use click count as a reward signal with
      `SlimeRecipe` to fine-tune a model on Wikipedia navigation.
    """
