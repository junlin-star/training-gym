"""Patch slime train entrypoints to report rollout-engine startup status."""

from __future__ import annotations

import re
from pathlib import Path


PREAMBLE_MARKER = "PATCHED_TRAINING_GYM_PREAMBLE"
ROLLOUT_MARKER = "PATCHED_TRAINING_GYM_ROLLOUT_STATUS"
WEIGHT_SYNC_MARKER = "PATCHED_TRAINING_GYM_WEIGHT_SYNC_STATUS"
STEP_START_MARKER = "PATCHED_TRAINING_GYM_STEP_START"
STEP_FINISH_MARKER = "PATCHED_TRAINING_GYM_STEP_FINISH"
GENERATE_ROLLOUT_MARKER = "PATCHED_TRAINING_GYM_GENERATE_ROLLOUT_STATUS"

PREAMBLE = (
    f"# {PREAMBLE_MARKER}: bootstrap phase reporter (runs once per process)\n"
    "import sys as _tg_sys\n"
    "if '/root' not in _tg_sys.path:\n"
    "    _tg_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_training_gym.frameworks.slime.phase_reporting import (\n"
    "        report_generate_rollouts as _tg_report_generate_rollouts,\n"
    "        report_rollout_initializing as _tg_report_rollout_initializing,\n"
    "        report_step_start as _tg_report_step_start,\n"
    "        report_step_complete as _tg_report_step_complete,\n"
    "        report_weight_sync as _tg_report_weight_sync,\n"
    "    )\n"
    "except ImportError:\n"
    "    def _tg_report_generate_rollouts(args): pass\n"
    "    def _tg_report_rollout_initializing(args): pass\n"
    "    def _tg_report_step_start(args, rollout_id=None): pass\n"
    "    def _tg_report_step_complete(args, rollout_id=None): pass\n"
    "    def _tg_report_weight_sync(args): pass\n"
    "\n"
)


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping rollout-status patch")
        return

    src = path.read_text()
    needs_preamble = PREAMBLE_MARKER not in src
    needs_rollout = ROLLOUT_MARKER not in src
    needs_weight_sync = WEIGHT_SYNC_MARKER not in src
    needs_step_start = STEP_START_MARKER not in src
    needs_step_finish = STEP_FINISH_MARKER not in src
    needs_generate_rollout = GENERATE_ROLLOUT_MARKER not in src

    if not (
        needs_preamble
        or needs_rollout
        or needs_weight_sync
        or needs_step_start
        or needs_step_finish
        or needs_generate_rollout
    ):
        print(f"{path.name} already patched for rollout status reporting")
        return

    if needs_preamble:
        src = PREAMBLE + src
    elif "_tg_report_generate_rollouts" not in src:
        src = src.replace(
            "    from modal_training_gym.frameworks.slime.phase_reporting import (\n",
            "    from modal_training_gym.frameworks.slime.phase_reporting import (\n"
            "        report_generate_rollouts as _tg_report_generate_rollouts,\n",
            1,
        )
        src = src.replace(
            "except ImportError:\n",
            "except ImportError:\n    def _tg_report_generate_rollouts(args): pass\n",
            1,
        )
    if "_tg_report_step_start" not in src:
        src = src.replace(
            "    from modal_training_gym.frameworks.slime.phase_reporting import (\n",
            "    from modal_training_gym.frameworks.slime.phase_reporting import (\n"
            "        report_step_start as _tg_report_step_start,\n",
            1,
        )
        src = src.replace(
            "except ImportError:\n",
            "except ImportError:\n    def _tg_report_step_start(args, rollout_id=None): pass\n",
            1,
        )
    if "_tg_report_step_complete" not in src:
        src = src.replace(
            "    from modal_training_gym.frameworks.slime.phase_reporting import (\n",
            "    from modal_training_gym.frameworks.slime.phase_reporting import (\n"
            "        report_step_complete as _tg_report_step_complete,\n",
            1,
        )
        src = src.replace(
            "except ImportError:\n",
            "except ImportError:\n    def _tg_report_step_complete(args, rollout_id=None): pass\n",
            1,
        )

    rollout_count = 0
    if needs_rollout:
        rollout_pattern = re.compile(
            r"^(?P<indent>[ \t]*)rollout_manager, num_rollout_per_epoch = create_rollout_manager\(args, pgs\[\"rollout\"\]\)",
            re.M,
        )

        def _rollout_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{indent}# {ROLLOUT_MARKER}: rollout engine startup state\n"
                f"{indent}_tg_report_rollout_initializing(args)\n"
                f'{indent}rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])'
            )

        src, rollout_count = rollout_pattern.subn(_rollout_replacement, src, count=1)

    step_start_count = 0
    if needs_step_start:
        step_start_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<line>.*rollout_manager\.generate\.remote\((?P<rollout_id>[^)\n]+)\).*)",
            re.M,
        )

        def _step_start_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            line = match.group("line")
            rollout_id = match.group("rollout_id").strip()
            return "\n".join(
                [
                    f"{indent}# {STEP_START_MARKER}: training step start",
                    f"{indent}_tg_report_step_start(args, {rollout_id})",
                    f"{indent}{line}",
                ]
            )

        src, step_start_count = step_start_pattern.subn(_step_start_replacement, src)

    weight_sync_count = 0
    if needs_weight_sync or needs_generate_rollout:
        weight_sync_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<call>(?:await[ \t]+)?(?:[A-Za-z_][A-Za-z0-9_]*\.)?update_weights\(\))",
            re.M,
        )

        def _weight_sync_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            call = match.group("call")
            lines = []
            if needs_weight_sync:
                lines.extend(
                    [
                        f"{indent}# {WEIGHT_SYNC_MARKER}: weight sync state",
                        f"{indent}_tg_report_weight_sync(args)",
                    ]
                )
            lines.append(f"{indent}{call}")
            if needs_generate_rollout:
                lines.extend(
                    [
                        f"{indent}# {GENERATE_ROLLOUT_MARKER}: rollout generation state",
                        f"{indent}_tg_report_generate_rollouts(args)",
                    ]
                )
            return "\n".join(lines)

        src, weight_sync_count = weight_sync_pattern.subn(
            _weight_sync_replacement, src, count=1
        )

    step_finish_count = 0
    if needs_step_finish:
        step_finish_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<call>(?:await[ \t]+)?(?:[A-Za-z_][A-Za-z0-9_]*\.)?update_weights\(\))[ \t]*(?P<newline>\r?\n?)$"
        )
        lines = src.splitlines(keepends=True)
        patched_lines = []
        in_rollout_loop = False
        loop_indent = ""
        for line in lines:
            patched_lines.append(line)
            loop_match = re.match(
                r"^(?P<indent>[ \t]*)for[ \t]+rollout_id[ \t]+in[ \t]+.*:",
                line,
            )
            if loop_match:
                in_rollout_loop = True
                loop_indent = loop_match.group("indent")
                continue
            if not in_rollout_loop or not line.strip():
                continue
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            if len(indent) <= len(loop_indent):
                in_rollout_loop = False
                continue
            if step_finish_count != 0:
                continue
            step_finish_match = step_finish_pattern.match(line)
            if step_finish_match:
                newline = step_finish_match.group("newline") or "\n"
                patched_lines.extend(
                    [
                        f"{indent}# {STEP_FINISH_MARKER}: training step finish{newline}",
                        f"{indent}_tg_report_step_complete(args, rollout_id){newline}",
                    ]
                )
                step_finish_count += 1
        src = "".join(patched_lines)

    failed = []
    if needs_rollout and rollout_count != 1:
        failed.append("rollout init")
    if (needs_weight_sync or needs_generate_rollout) and weight_sync_count != 1:
        failed.append("weight sync")
    if needs_step_finish and step_finish_count != 1:
        failed.append("step finish")
    if needs_step_start and step_start_count == 0:
        failed.append("step start")
    if failed:
        print(f"WARNING: Could not patch {path.name} for: {', '.join(failed)}")

    path.write_text(src)
    print(f"Patched {path.name} with rollout status reporting")


_patch_file(Path("/root/slime/train.py"))
_patch_file(Path("/root/slime/train_async.py"))
