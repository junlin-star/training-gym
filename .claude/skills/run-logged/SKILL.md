---
name: run-logged
description: Launch a long-running command (training runs, servers, builds) in a tmux session with unbuffered stdout so it streams live to the pane AND is teed to a log file Claude can tail. Use when the user wants to watch a long job live while Claude also monitors it, or when a Python process's output appears "frozen" because stdout is block-buffered to a pipe.
---

# run-logged

Run a long-lived command so that **both** a human (watching the tmux pane) and
Claude (reading a log file) see output in real time.

The core problem this solves: when a Python process's stdout is a pipe (e.g.
`python ... | tee log`) instead of a TTY, Python **block-buffers** stdout. During
quiet phases (image builds, model loading) the pane and log look frozen for
minutes even though work is happening. `PYTHONUNBUFFERED=1` forces line/stream
flushing so output appears immediately.

## Launch pattern

Pick a tmux session name and an absolute log path (prefer the scratchpad dir, not
`/tmp`). Then:

```bash
SESSION=opd                       # tmux session to run in (create if missing)
LOG=/abs/path/to/scratchpad/run.log

# Create the session if it doesn't exist (detached).
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"

# Fresh log, then launch UNBUFFERED, merging stderr, teeing to the log.
: > "$LOG"
tmux send-keys -t "$SESSION" \
  "PYTHONUNBUFFERED=1 <your command here> 2>&1 | tee $LOG" Enter
```

Key requirements:
- **`PYTHONUNBUFFERED=1`** in front of the command (for Python jobs). For other
  runtimes, use the equivalent (`stdbuf -oL -eL <cmd>` is a general fallback).
- **`2>&1`** so stderr (tracebacks!) is captured in the log, not just the pane.
- **`tee`** (not `>`) so the pane stays live while the log persists.
- Keep any required env (`MODAL_ENVIRONMENT`, `MODAL_KEY`, …) on the same line, or
  rely on vars already exported in that pane's shell.

## Monitoring (Claude side)

Read the log directly, or use a background Monitor that filters for progress +
**all** failure signatures (never grep only the happy path — a crash must emit a
line, or silence reads as "still running"):

```bash
tail -n0 -f "$LOG" | grep -E --line-buffered \
  "Traceback|Error|FAILED|Exception|Killed|OOM|assert| <progress markers> "
```

Tips:
- Use `tail -n0 -f` (not `-n +1`) when re-attaching so you don't re-emit old lines
  after a log truncation.
- Drop high-frequency polling lines (e.g. "Waiting for…") from the filter or the
  monitor floods and auto-stops.
- The buffered driver log can still lag during image builds / model loads where
  output genuinely hasn't been written yet — cross-check the underlying service
  (e.g. `modal app logs <id>`) for true progress in those windows.

## Stopping / restarting

```bash
tmux send-keys -t "$SESSION" C-c    # interrupt the running command
```

Then re-launch with the pattern above (truncating the log first). Note that
restarting a job that rebuilds container images or reloads large models pays that
cost again — restart during cheap phases when possible.
