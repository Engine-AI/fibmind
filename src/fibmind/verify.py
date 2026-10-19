"""Automated verification loop: execute physical commands and bind outcomes to reflection."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fibmind.brain import BrainState, FibBrain
from fibmind.service import MemoryService

ENV_FAILURE_PATTERNS = (
    "modulenotfounderror",
    "importerror",
    "syntaxerror",
    "indentationerror",
    "command not found",
    "no such file or directory",
    "permission denied",
    "errors during collection",
    "collection error",
    "internalerror",
)

ZERO_TEST_PATTERNS = (
    "collected 0 items",
    "no tests ran",
    "ran 0 tests",
    "0 passed, 0 total",
    "[no test files]",
    "no test suite found",
    "no tests found",
)

GENERIC_STEMS = frozenset(
    {
        "__init__",
        "main",
        "test",
        "tests",
        "index",
        "utils",
        "common",
        "base",
        "conftest",
        "config",
        "app",
        "cli",
    }
)


def _target_touched(target: str, combined_lower: str) -> bool:
    """Return True if ``target`` file or non-generic module stem appears in ``combined_lower``."""
    raw = str(target).strip().lower().replace("\\", "/")
    if not raw:
        return False
    if raw in combined_lower:
        return True
    p = Path(raw)
    if p.name and p.name in combined_lower:
        return True
    stem = p.stem
    if stem and len(stem) >= 4 and stem not in GENERIC_STEMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(stem)}(?![a-z0-9])", combined_lower):
            return True
    return False


def diagnose_verification_outcome(
    exit_code: int,
    stdout: str,
    stderr: str,
    *,
    target_files: list[str] | tuple[str, ...] | None = None,
    assertion_patterns: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Diagnose execution outcome into (verdict, explanation).

    Returns:
        verdict: 'confirmed', 'refuted', or 'inconclusive'
        explanation: reason describing the decision
    """
    combined = (stderr + "\n" + stdout).lower()

    # 1. Environment / Infrastructure Failures
    if exit_code != 0:
        if exit_code in (126, 127):
            return "inconclusive", f"command failed to start (exit code {exit_code})"
        for pattern in ENV_FAILURE_PATTERNS:
            if pattern in combined:
                return "inconclusive", f"environment or infrastructure failure ({pattern})"

    # 2. Zero tests run
    for zpat in ZERO_TEST_PATTERNS:
        if zpat in combined:
            return "inconclusive", f"no tests collected or executed ({zpat})"

    # 3. Target Scope / Files Binding Check (symmetrical check on both success and failure)
    if target_files:
        non_empty_targets = [f for f in target_files if f and str(f).strip()]
        touched = any(_target_touched(f, combined) for f in non_empty_targets) if non_empty_targets else True
        if not touched:
            if exit_code != 0:
                return "inconclusive", f"failure occurred in unrelated modules; did not touch {list(target_files)}"
            return "inconclusive", f"verification passed but did not touch target files: {list(target_files)}"

    # 4. Assertion Patterns Check
    if assertion_patterns:
        matched = any(re.search(pat, combined, re.IGNORECASE) for pat in assertion_patterns)
        if not matched:
            return (
                "inconclusive",
                f"command succeeded but missed required assertion pattern: {list(assertion_patterns)}",
            )

    # 5. Deterministic outcome
    if exit_code == 0:
        return (
            "confirmed",
            "verification command passed with confirmed target coverage"
            if target_files
            else "verification command passed",
        )
    return "refuted", f"verification command failed with exit code {exit_code}"


def run_command_and_reflect(
    command: list[str] | str,
    *,
    node_id: str | None = None,
    goal: str | None = None,
    source: str | None = None,
    target_files: list[str] | tuple[str, ...] | None = None,
    assertion_patterns: list[str] | tuple[str, ...] | None = None,
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

    Evaluates test attribution:
    - Exit 0 with touched targets -> 'confirmed'
    - Exit 0 but missing targets -> 'inconclusive'
    - Exit != 0 with environment or unrelated failure -> 'inconclusive' (no confidence penalty)
    - Exit != 0 with verified target failure -> 'refuted'
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

    # Resolve target files from node metadata if not explicitly provided
    resolved_files = list(target_files) if target_files else []
    if not resolved_files and target_node:
        node_meta = service.read(lambda m: m.nodes.get(target_node).metadata if target_node in m.nodes else {})
        extracted = node_meta.get("files") or node_meta.get("changed_files")
        if isinstance(extracted, (list, tuple)):
            resolved_files.extend(str(f) for f in extracted)

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

    verdict, reason = diagnose_verification_outcome(
        exit_code,
        proc.stdout,
        proc.stderr,
        target_files=resolved_files,
        assertion_patterns=assertion_patterns,
    )

    note_lines = [f"{cmd_display} ({verdict}: {reason} in {duration}s)"]
    if proc.stderr.strip():
        stderr_tail = [line.strip() for line in proc.stderr.strip().splitlines() if line.strip()][-5:]
        note_lines.append("stderr:\n" + "\n".join(stderr_tail))
    elif proc.stdout.strip() and exit_code != 0:
        stdout_tail = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()][-5:]
        note_lines.append("stdout:\n" + "\n".join(stdout_tail))
    note = "\n".join(note_lines)

    reflection = brain.reflect(
        verdict=verdict,
        source=inferred_source,
        node_id=target_node,
        goal=target_goal,
        note=note,
        state=state,
    )

    touched_targets = []
    if resolved_files:
        combined_lower = (proc.stderr + "\n" + proc.stdout).lower()
        for f in resolved_files:
            if _target_touched(f, combined_lower):
                touched_targets.append(f)

    attribution = {
        "command": cmd_display,
        "exit_code": exit_code,
        "verdict": verdict,
        "target_files": resolved_files,
        "touched_targets": touched_targets,
        "duration_sec": duration,
        "source": inferred_source,
    }

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
        "attribution": attribution,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fibmind-verify",
        description="Run a verification command and automatically reflect outcome into FibBrain",
    )
    parser.add_argument("--node-id", help="Memory node ID to verify")
    parser.add_argument("--goal", help="Task goal or search query to locate memory node")
    parser.add_argument("--target-files", nargs="*", default=None, help="Target file paths expected to be affected")
    parser.add_argument(
        "--assertion-patterns",
        nargs="*",
        default=None,
        help="Regex patterns that must appear in command output",
    )
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
            target_files=args.target_files,
            assertion_patterns=args.assertion_patterns,
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
