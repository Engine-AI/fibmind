"""Automated verification loop: execute physical commands and bind outcomes to reflection."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fibmind.brain import BrainState, FibBrain
from fibmind.service import MemoryService


def run_command_and_reflect(
    command: list[str] | str,
    *,
    node_id: str | None = None,
    goal: str | None = None,
    source: str | None = None,
    store_path: str | Path | None = None,
    owner: str | None = None,
    workspace_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute a verification command and reflect its outcome into FibBrain.

    Exit code 0 -> verdict="confirmed"
    Exit code != 0 -> verdict="refuted"
    """
    if isinstance(command, str):
        cmd_args = shlex.split(command)
        cmd_display = command
    else:
        cmd_args = list(command)
        cmd_display = " ".join(shlex.quote(arg) for arg in cmd_args)

    if not cmd_args:
        raise ValueError("Cannot execute empty command")

    inferred_source = source or Path(cmd_args[0]).name

    start_time = time.monotonic()
    proc = subprocess.run(
        cmd_args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    duration = round(time.monotonic() - start_time, 2)
    exit_code = proc.returncode
    verdict = "confirmed" if exit_code == 0 else "refuted"

    status_label = "passed" if exit_code == 0 else f"failed (code {exit_code})"
    note_lines = [f"{cmd_display} ({status_label} in {duration}s)"]
    if proc.stderr.strip():
        stderr_tail = [line.strip() for line in proc.stderr.strip().splitlines() if line.strip()][-5:]
        note_lines.append("stderr:\n" + "\n".join(stderr_tail))
    elif proc.stdout.strip() and exit_code != 0:
        stdout_tail = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()][-5:]
        note_lines.append("stdout:\n" + "\n".join(stdout_tail))
    note = "\n".join(note_lines)

    resolved_store = Path(store_path or os.environ.get("FIBMIND_STORE") or ".fibmind/memory.db").expanduser().resolve()
    service = MemoryService(resolved_store)
    brain = FibBrain(service)

    state = BrainState(
        owner=owner or os.environ.get("FIBMIND_OWNER"),
        workspace_id=workspace_id or os.environ.get("FIBMIND_WORKSPACE_ID"),
        project_id=project_id or os.environ.get("FIBMIND_PROJECT_ID"),
        session_id=session_id or os.environ.get("FIBMIND_SESSION_ID"),
        task_id=task_id or os.environ.get("FIBMIND_TASK_ID"),
    )

    target_goal = goal
    target_node = node_id

    # If neither node_id nor goal is provided, try resolving from active task goal
    if target_node is None and not target_goal:
        hot = brain.hot_snapshot(state=state)
        active_goal = hot.get("goal")
        if active_goal:
            target_goal = active_goal
        else:
            raise ValueError("Verification requires node_id or goal (or an active task goal in working context)")

    reflection = brain.reflect(
        verdict=verdict,
        source=inferred_source,
        node_id=target_node,
        goal=target_goal,
        note=note,
        state=state,
    )

    return {
        "command": cmd_display,
        "exit_code": exit_code,
        "verdict": verdict,
        "source": inferred_source,
        "duration_sec": duration,
        "note": note,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "reflection": reflection,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fibmind-verify",
        description="Run a verification command and automatically reflect outcome into FibBrain",
    )
    parser.add_argument("--node-id", help="Memory node ID to verify")
    parser.add_argument("--goal", help="Task goal or search query to locate memory node")
    parser.add_argument("--source", help="Source identifier (default: command name)")
    parser.add_argument(
        "--store",
        default=os.environ.get("FIBMIND_STORE") or ".fibmind/memory.db",
        help="Path to memory store",
    )
    parser.add_argument("--owner", help="Owner identity (default: FIBMIND_OWNER)")
    parser.add_argument("--workspace-id", help="Workspace ID (default: FIBMIND_WORKSPACE_ID)")
    parser.add_argument("--project-id", help="Project ID (default: FIBMIND_PROJECT_ID)")
    parser.add_argument("--session-id", help="Session ID (default: FIBMIND_SESSION_ID)")
    parser.add_argument("--task-id", help="Task ID (default: FIBMIND_TASK_ID)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Verification command and arguments")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("fibmind-verify: error: missing command to execute", file=sys.stderr)
        return 1

    try:
        res = run_command_and_reflect(
            cmd,
            node_id=args.node_id,
            goal=args.goal,
            source=args.source,
            store_path=args.store,
            owner=args.owner,
            workspace_id=args.workspace_id,
            project_id=args.project_id,
            session_id=args.session_id,
            task_id=args.task_id,
        )
        if res["stdout"]:
            sys.stdout.write(res["stdout"])
        if res["stderr"]:
            sys.stderr.write(res["stderr"])

        node_title = res["reflection"].get("title", res["reflection"].get("node_id", "target"))
        print(
            f"\nfibmind-verify: {res['source']} -> {res['verdict']} on '{node_title}' "
            f"(confidence={res['reflection'].get('confidence', 0.0):.2f})",
            file=sys.stderr,
        )
        return int(res["exit_code"])
    except Exception as exc:
        print(f"fibmind-verify: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
