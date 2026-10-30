"""Automated verification loop: execute physical commands and bind outcomes to reflection."""

from __future__ import annotations

import argparse
import inspect
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
    service: MemoryService | None = None,
    brain: FibBrain | None = None,
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
    effective_service = service or getattr(brain, "service", None) or MemoryService(resolved_store)
    effective_brain = brain or FibBrain(effective_service)

    state = BrainState(
        owner=owner or os.environ.get("FIBMIND_OWNER"),
        workspace_id=workspace_id or os.environ.get("FIBMIND_WORKSPACE_ID"),
        project_id=project_id or os.environ.get("FIBMIND_PROJECT_ID"),
        session_id=session_id or os.environ.get("FIBMIND_SESSION_ID"),
        task_id=task_id or os.environ.get("FIBMIND_TASK_ID"),
    )

    target_goal = goal
    target_node = node_id

    # If target_node is provided, inherit missing identity fields from target node
    if target_node:
        target_info = effective_service.read(
            lambda m: (
                (
                    m.nodes[target_node].owner,
                    m.nodes[target_node].workspace_id,
                    m.nodes[target_node].project_id,
                    m.nodes[target_node].session_id,
                    m.nodes[target_node].task_id,
                )
                if target_node in m.nodes
                else None
            )
        )
        if target_info:
            n_owner, n_ws, n_proj, n_sess, n_task = target_info
            state = BrainState(
                owner=state.owner or n_owner,
                workspace_id=state.workspace_id or n_ws,
                project_id=state.project_id or n_proj,
                session_id=state.session_id or n_sess,
                task_id=state.task_id or n_task,
            )

    # If neither node_id nor goal is provided, try resolving from active task goal
    if target_node is None and not target_goal:
        hot = effective_brain.hot_snapshot(state=state)
        active_goal = hot.get("goal")
        if active_goal:
            target_goal = active_goal
        else:
            raise ValueError("Verification requires node_id or goal (or an active task goal in working context)")

    # Resolve target files from node metadata if not explicitly provided
    resolved_files = list(target_files) if target_files else []
    if not resolved_files and target_node:
        node_meta = effective_service.read(
            lambda m: m.nodes.get(target_node).metadata if target_node in m.nodes else {}
        )
        candidates: list[str] = []
        for key in ("files", "changed_files"):
            extracted = node_meta.get(key)
            if isinstance(extracted, (list, tuple, set)):
                candidates.extend(str(f).strip() for f in extracted if f and str(f).strip())
            elif isinstance(extracted, str) and extracted.strip():
                candidates.append(extracted.strip())

        file_hashes = node_meta.get("file_hashes")
        if isinstance(file_hashes, dict):
            for k in file_hashes:
                if k and str(k).strip():
                    candidates.append(str(k).strip())

        sym_hashes = node_meta.get("symbol_hashes")
        if isinstance(sym_hashes, dict):
            for k in sym_hashes:
                if k and "::" in str(k):
                    f_part = str(k).split("::", 1)[0].strip()
                    if f_part:
                        candidates.append(f_part)

        symbols = node_meta.get("symbols")
        if isinstance(symbols, (list, tuple, set)):
            for s in symbols:
                if s and "::" in str(s):
                    f_part = str(s).split("::", 1)[0].strip()
                    if f_part:
                        candidates.append(f_part)

        seen_resolved: set[str] = set()
        for c in candidates:
            if c not in seen_resolved:
                seen_resolved.add(c)
                resolved_files.append(c)

    start_time = time.monotonic()
    stdout = ""
    stderr = ""
    try:
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
        stdout = proc.stdout
        stderr = proc.stderr
        verdict, reason = diagnose_verification_outcome(
            exit_code,
            stdout,
            stderr,
            target_files=resolved_files,
            assertion_patterns=assertion_patterns,
        )
    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - start_time, 2)
        exit_code = 124
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        verdict = "inconclusive"
        reason = f"command timed out after {timeout}s"
    except (FileNotFoundError, PermissionError, OSError) as exc:
        duration = round(time.monotonic() - start_time, 2)
        exit_code = 127 if isinstance(exc, FileNotFoundError) else 126
        stderr = str(exc)
        verdict = "inconclusive"
        reason = f"command failed to start ({type(exc).__name__}: {exc})"

    note_lines = [f"{cmd_display} ({verdict}: {reason} in {duration}s)"]
    if stderr.strip():
        stderr_tail = [line.strip() for line in stderr.strip().splitlines() if line.strip()][-5:]
        note_lines.append("stderr:\n" + "\n".join(stderr_tail))
    elif stdout.strip() and exit_code != 0:
        stdout_tail = [line.strip() for line in stdout.strip().splitlines() if line.strip()][-5:]
        note_lines.append("stdout:\n" + "\n".join(stdout_tail))
    note = "\n".join(note_lines)

    reflect_kwargs: dict[str, Any] = {
        "verdict": verdict,
        "source": inferred_source,
        "node_id": target_node,
        "goal": target_goal,
        "note": note,
        "state": state,
    }
    sig = inspect.signature(effective_brain.reflect)
    if "trust_ring" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        reflect_kwargs["trust_ring"] = "ring0"

    try:
        reflection = effective_brain.reflect(**reflect_kwargs)
    except TypeError:
        reflect_kwargs.pop("trust_ring", None)
        reflection = effective_brain.reflect(**reflect_kwargs)

    regrounded = False
    if verdict == "confirmed" and reflection.get("node_id"):
        from fibmind.reflex import compute_file_hashes, compute_symbol_hashes

        confirmed_nid = str(reflection["node_id"])
        effective_cwd = Path(cwd or ".").resolve()

        def _reground_node(memory: Any) -> tuple[bool, dict[str, Any]]:
            target = memory.nodes.get(confirmed_nid)
            if target is None:
                return False, {}
            meta = target.metadata if isinstance(target.metadata, dict) else {}

            tracked_files: list[str] = []
            seen_f: set[str] = set()
            for k in ("files", "changed_files"):
                val = meta.get(k)
                if isinstance(val, (list, tuple, set)):
                    for f in val:
                        s = str(f).strip()
                        if s and s not in seen_f:
                            seen_f.add(s)
                            tracked_files.append(s)
                elif isinstance(val, str) and val.strip():
                    s = val.strip()
                    if s not in seen_f:
                        seen_f.add(s)
                        tracked_files.append(s)
            fh = meta.get("file_hashes")
            if isinstance(fh, dict):
                for k in fh:
                    s = str(k).strip()
                    if s and s not in seen_f:
                        seen_f.add(s)
                        tracked_files.append(s)

            tracked_specs: list[str] = []
            seen_s: set[str] = set()
            syms = meta.get("symbols")
            if isinstance(syms, (list, tuple, set)):
                for s in syms:
                    spec = str(s).strip()
                    if spec and spec not in seen_s:
                        seen_s.add(spec)
                        tracked_specs.append(spec)
            sh = meta.get("symbol_hashes")
            if isinstance(sh, dict):
                for k in sh:
                    spec = str(k).strip()
                    if spec and spec not in seen_s:
                        seen_s.add(spec)
                        tracked_specs.append(spec)

            refreshed = False
            if tracked_files:
                new_file_hashes = compute_file_hashes(tracked_files, cwd=effective_cwd)
                if new_file_hashes:
                    meta["file_hashes"] = new_file_hashes
                    refreshed = True

            if tracked_specs:
                new_sym_hashes = compute_symbol_hashes(tracked_specs, cwd=effective_cwd)
                if new_sym_hashes:
                    meta["symbol_hashes"] = new_sym_hashes
                    refreshed = True

            if "drift_detected" in meta:
                meta.pop("drift_detected", None)
                refreshed = True
            if "upstream_refuted" in meta:
                meta.pop("upstream_refuted", None)
                refreshed = True

            target.metadata = meta
            return refreshed, dict(meta)

        regrounded, updated_meta = effective_service.run(_reground_node)
        if updated_meta:
            reflection["metadata"] = updated_meta

    touched_targets = []
    if resolved_files:
        combined_lower = (stderr + "\n" + stdout).lower()
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
        "trust_ring": "ring0",
        "regrounded": regrounded,
    }

    return {
        "command": cmd_display,
        "exit_code": exit_code,
        "verdict": verdict,
        "source": inferred_source,
        "duration_sec": duration,
        "note": note,
        "stdout": stdout,
        "stderr": stderr,
        "reflection": reflection,
        "attribution": attribution,
    }


def revalidate_memories(
    brain: FibBrain,
    *,
    state: BrainState | None = None,
    cwd: str | Path | None = None,
    execute_commands: bool = False,
    older_than_days: int = 90,
) -> dict[str, Any]:
    """Active revalidation and drift recovery runner for long-term memory."""
    from fibmind.models import EventOp, MemoryStatus, utc_now
    from fibmind.reflex import check_node_preconditions

    effective_state = state or BrainState()
    identity = effective_state.identity()
    effective_cwd = cwd or getattr(effective_state, "workspace_path", None)

    # 1. Collect candidate node IDs from two sources:
    # A. Visible ACTIVE nodes in brain.memory that have node.metadata.get("drift_detected")
    def _collect_drifted(m: Any) -> list[str]:
        drifted: list[str] = []
        for n in m.nodes.values():
            if n.status == MemoryStatus.ACTIVE and n.folded_into is None:
                if (n.metadata or {}).get("drift_detected"):
                    if m.is_visible(
                        n,
                        owner=identity.get("owner"),
                        statuses={n.status},
                        workspace_id=identity.get("workspace_id"),
                        project_id=identity.get("project_id"),
                        session_id=identity.get("session_id"),
                        task_id=identity.get("task_id"),
                    ):
                        drifted.append(n.id)
        return drifted

    if hasattr(brain.memory, "read"):
        drifted_ids = brain.memory.read(_collect_drifted)
    elif hasattr(brain.memory, "nodes"):
        drifted_ids = _collect_drifted(brain.memory)
    else:
        drifted_ids = []

    # B. Nodes returned by brain.revalidation_candidates(state=effective_state, older_than_days=older_than_days)
    try:
        rev_candidates = brain.revalidation_candidates(state=effective_state, older_than_days=older_than_days)
        rev_candidate_ids = [c["node_id"] for c in rev_candidates if isinstance(c, dict) and "node_id" in c]
    except (AttributeError, Exception):
        rev_candidate_ids = []

    # Deduplicate candidate node IDs while preserving order
    candidate_ids: list[str] = list(dict.fromkeys(drifted_ids + rev_candidate_ids))

    recovered: list[str] = []
    still_drifted: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []

    def _get_node(m: Any, nid: str) -> Any:
        return m.nodes.get(nid)

    def _recover_node(m: Any, nid: str) -> None:
        n = m.nodes.get(nid)
        if n is None:
            return
        meta = dict(n.metadata or {})
        meta.pop("drift_detected", None)
        n.metadata = meta
        n.updated_at = utc_now()
        m._record(
            EventOp.OBSERVE,
            {
                "node_id": n.id,
                "verdict": "inconclusive",
                "source": "revalidate_recovered",
                "note": "physical preconditions restored",
                "confidence": n.confidence,
                "status": n.status.value,
                "status_reason": n.status_reason,
                "updated_at": n.updated_at.isoformat(),
                "metadata_remove": ["drift_detected"],
            },
        )

    for node_id in candidate_ids:
        if hasattr(brain.memory, "read"):
            node = brain.memory.read(lambda m, nid=node_id: _get_node(m, nid))
        elif hasattr(brain.memory, "nodes"):
            node = _get_node(brain.memory, node_id)
        else:
            node = None

        if node is None:
            continue

        meta = node.metadata or {}
        has_precond = (
            bool(meta.get("file_hashes")) or bool(meta.get("symbol_hashes")) or bool(meta.get("require_files_exist"))
        )

        if has_precond or meta.get("drift_detected"):
            ok, reason = check_node_preconditions(node, cwd=effective_cwd)
            if ok:
                if meta.get("drift_detected"):
                    if hasattr(brain.memory, "run"):
                        brain.memory.run(lambda m, nid=node_id: _recover_node(m, nid))
                    elif hasattr(brain, "service") and brain.service is not None:
                        brain.service.run(lambda m, nid=node_id: _recover_node(m, nid))
                    else:
                        _recover_node(brain.memory, node_id)
                    recovered.append(node_id)
            else:
                still_drifted.append({"node_id": node_id, "title": node.title, "reason": reason})

        if execute_commands:
            cmd = meta.get("verify") or meta.get("command")
            if not cmd:
                try:
                    from fibmind.procedure import Procedure

                    proc = Procedure.parse(node.content)
                    if proc is not None and proc.verify:
                        cmd = proc.verify
                except Exception:
                    pass

            if cmd:
                res = run_command_and_reflect(
                    cmd,
                    node_id=node_id,
                    brain=brain,
                    owner=identity.get("owner"),
                    workspace_id=identity.get("workspace_id"),
                    project_id=identity.get("project_id"),
                    session_id=identity.get("session_id"),
                    task_id=identity.get("task_id"),
                    cwd=effective_cwd,
                )
                executed.append(
                    {
                        "node_id": node_id,
                        "command": cmd,
                        "verdict": res.get("verdict"),
                        "exit_code": res.get("exit_code"),
                        "source": res.get("source"),
                        "note": res.get("note"),
                    }
                )

    return {
        "checked": len(candidate_ids),
        "recovered": recovered,
        "still_drifted": still_drifted,
        "executed": executed,
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
