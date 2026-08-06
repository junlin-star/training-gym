"""Give the rollout router longer than 30s to bind its port.

``miles/ray/rollout/router_manager.py`` launches the router with
``wait_for_server_ready(..., timeout=30)`` at both call sites, and there is no
CLI flag for it. The router is spawned from ``RolloutManager.__init__`` while
the Megatron actors are concurrently loading the checkpoint, so on a large model
the child is alive but starved and has not bound its port when the 30s expires:

    RuntimeError: Server at 127.0.0.1:4077 not ready after 30s

Small models never hit this -- a Qwen3-0.6B control run on the same image logs
"Router launched at 127.0.0.1:4077" well inside the window -- so this is a
model-size problem, not a broken router.

Raising the bound is safe: ``wait_for_server_ready`` returns as soon as the port
accepts a connection, and it still fails fast (``Server process died before port
... became ready``) if the child actually crashes. The only behaviour change is
how long we are willing to wait before giving up.

Report upstream; the timeout should be an argument.

Idempotent. Run at image build:  python patch_router_startup_timeout.py
"""

import pathlib

MARKER = "PATCHED_ROUTER_STARTUP_TIMEOUT"
TIMEOUT = 600

TARGET = pathlib.Path("/root/miles/miles/ray/rollout/router_manager.py")

OLD = "wait_for_server_ready(router_ip, router_port, process, timeout=30)"
NEW = (
    f"wait_for_server_ready(router_ip, router_port, process, timeout={TIMEOUT})"
    f"  # {MARKER}"
)
OLD_SESSION = "wait_for_server_ready(ip, port, process, timeout=30)"
NEW_SESSION = f"wait_for_server_ready(ip, port, process, timeout={TIMEOUT})  # {MARKER}"

if not TARGET.exists():
    print(f"{TARGET} not found; skipping router startup timeout patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("Router startup timeout patch already applied")
    raise SystemExit(0)

replacements = 0
for old, new in ((OLD, NEW), (OLD_SESSION, NEW_SESSION)):
    if old in src:
        src = src.replace(old, new)
        replacements += 1

if not replacements:
    raise SystemExit(
        "Router startup timeout patch did not match; miles' router_manager.py "
        "has changed. Re-check wait_for_server_ready call sites before shipping."
    )

TARGET.write_text(src)
print(f"Patched {replacements} router wait_for_server_ready call site(s) -> {TIMEOUT}s")
