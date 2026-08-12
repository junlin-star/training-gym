---
title: CLI Reference
description: Command-line interface for the Training Gym SDK.
---

The `training-gym` CLI is installed automatically with the package.

```bash
pip install git+https://github.com/modal-projects/training-gym.git@main
```

## `training-gym setup`

Deploy the observability dashboard to your Modal account.

```bash
training-gym setup
```

The command builds a Modal image (Node.js + Svelte frontend + FastAPI backend),
deploys it as a persistent web app, and prints the dashboard URL.

**Prerequisites:** a Modal account with `modal token set` already configured.

## `training-gym skills install`

Install the bundled `agent-driven-training` skill into the current project:

```bash
training-gym skills install
```

The command finds the nearest Git repository and copies the skill to
`.agents/skills/agent-driven-training`. Cursor and other agents that support
the Agent Skills standard discover project skills from `.agents/skills`. It
also creates `.claude/skills/agent-driven-training` as a relative symbolic link
to the canonical copy for Claude compatibility.

Options:

- `--project-dir DIR`: install into a specific project instead of discovering
  the nearest Git repository.
- `--force`: replace an existing skill or Claude link. Without this option,
  locally modified files are preserved.

## `training-gym --help`

```bash
training-gym --help
training-gym skills --help
training-gym skills install --help
```
