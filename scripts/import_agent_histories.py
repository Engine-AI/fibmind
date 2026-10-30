"""Multi-agent conversation harvester: import and summarize sessions from Codex, Claude Code, and AGY into FibMind."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@dataclass
class HarvestedSession:
    """Normalized representation of a single conversation session from any agent."""

    host: str  # 'codex', 'claude-code', 'agy'
    session_id: str
    workspace_name: str
    project_name: str
    user_goal: str
    timestamp: str
    key_decisions: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    summary_markdown: str = ""
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. AGY Harvester (~/.gemini/antigravity/brain)
# ---------------------------------------------------------------------------
def harvest_agy_sessions(max_sessions: int = 10) -> list[HarvestedSession]:
    brain_root = Path.home() / ".gemini/antigravity/brain"
    if not brain_root.exists():
        return []

    sessions = []
    dirs = [d for d in brain_root.iterdir() if d.is_dir() and (d / ".system_generated/logs/transcript.jsonl").exists()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    for conv_dir in dirs[:max_sessions]:
        conv_id = conv_dir.name
        log_file = conv_dir / ".system_generated/logs/transcript.jsonl"

        user_goal = ""
        files_touched = set()
        created_at = datetime.fromtimestamp(conv_dir.stat().st_mtime).isoformat()

        try:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    etype = event.get("type")
                    if etype == "USER_INPUT" and not user_goal:
                        raw = event.get("content", "")
                        # strip tags
                        cleaned = re.sub(r"<USER_REQUEST>([\s\S]*?)</USER_REQUEST>", r"\1", raw).strip()
                        cleaned = re.sub(r"<ADDITIONAL_METADATA>[\s\S]*?</ADDITIONAL_METADATA>", "", cleaned).strip()
                        user_goal = cleaned[:200]

                    tool_calls = event.get("tool_calls", [])
                    for tc in tool_calls:
                        name = tc.get("name")
                        args = tc.get("args") or tc.get("arguments") or {}
                        if name in ("edit_file", "write_file", "read_file"):
                            p = args.get("Path") or args.get("path")
                            if p:
                                files_touched.add(Path(p).name)

            if user_goal:
                sessions.append(
                    HarvestedSession(
                        host="agy",
                        session_id=conv_id,
                        workspace_name="antigravity",
                        project_name="fibmind",
                        user_goal=user_goal,
                        timestamp=created_at,
                        files_touched=sorted(files_touched)[:10],
                        tags=["agy", "antigravity"],
                    )
                )
        except Exception:
            continue

    return sessions


# ---------------------------------------------------------------------------
# 2. Claude Code Harvester (~/.claude/projects)
# ---------------------------------------------------------------------------
def harvest_claude_sessions(max_sessions: int = 10) -> list[HarvestedSession]:
    claude_projects = Path.home() / ".claude/projects"
    if not claude_projects.exists():
        return []

    sessions = []
    jsonl_files = list(claude_projects.glob("**/*.jsonl"))
    # Filter out subagents
    jsonl_files = [f for f in jsonl_files if "subagents" not in str(f)]
    jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    for jf in jsonl_files[:max_sessions]:
        session_id = jf.stem
        project_dir_name = jf.parent.name
        # clean project name from -Users-abbila-...
        clean_project = project_dir_name.split("-")[-1] or "claude-session"
        created_at = datetime.fromtimestamp(jf.stat().st_mtime).isoformat()

        user_goal = ""
        files_touched = set()

        try:
            with open(jf, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    t = event.get("type")
                    if t == "user" and not user_goal:
                        msg = event.get("message", {})
                        content = msg.get("content")
                        if isinstance(content, str) and not content.startswith("<local-command-caveat>"):
                            user_goal = content.strip()[:200]
                    elif t == "tool_use":
                        name = event.get("name")
                        inp = event.get("input", {})
                        if name in ("Edit", "Write", "Read"):
                            fp = inp.get("file_path") or inp.get("path")
                            if fp:
                                files_touched.add(Path(fp).name)

            if user_goal:
                sessions.append(
                    HarvestedSession(
                        host="claude-code",
                        session_id=session_id,
                        workspace_name=clean_project,
                        project_name=clean_project,
                        user_goal=user_goal,
                        timestamp=created_at,
                        files_touched=sorted(files_touched)[:10],
                        tags=["claude-code", clean_project],
                    )
                )
        except Exception:
            continue

    return sessions


# ---------------------------------------------------------------------------
# 3. Codex Harvester (~/.codex/thread_history_1.sqlite)
# ---------------------------------------------------------------------------
def harvest_codex_sessions(max_sessions: int = 10) -> list[HarvestedSession]:
    codex_db = Path.home() / ".codex/thread_history_1.sqlite"
    if not codex_db.exists():
        return []

    sessions = []
    con = sqlite3.connect(codex_db)
    con.row_factory = sqlite3.Row

    try:
        # Query distinct threads
        threads = con.execute(
            """
            SELECT thread_id, max(started_at) as last_active 
            FROM thread_turns 
            GROUP BY thread_id 
            ORDER BY last_active DESC 
            LIMIT ?
        """,
            (max_sessions,),
        ).fetchall()

        for t in threads:
            tid = t["thread_id"]
            # get user messages
            items = con.execute(
                """
                SELECT item_type, item_json 
                FROM thread_items 
                WHERE thread_id = ? 
                ORDER BY rollout_ordinal ASC
            """,
                (tid,),
            ).fetchall()

            user_goal = ""
            files_touched = set()

            for it in items:
                itype = it["item_type"]
                try:
                    data = json.loads(it["item_json"])
                    if itype == "userMessage" and not user_goal:
                        content = data.get("content", "")
                        if isinstance(content, str):
                            user_goal = content.strip()[:200]
                        elif isinstance(content, list):
                            for sub in content:
                                if isinstance(sub, dict) and sub.get("text"):
                                    user_goal = sub.get("text").strip()[:200]
                                    break
                    elif itype == "commandExecution":
                        cmd = data.get("command", "")
                        # simple regex for file targets
                        found = re.findall(r"[\w\./-]+\.(?:py|ts|tsx|java|json|md)", cmd)
                        for f in found:
                            files_touched.add(Path(f).name)
                except Exception:
                    continue

            if user_goal:
                sessions.append(
                    HarvestedSession(
                        host="codex",
                        session_id=tid,
                        workspace_name="codex-workspace",
                        project_name="codex-project",
                        user_goal=user_goal,
                        timestamp=t["last_active"] or datetime.now().isoformat(),
                        files_touched=sorted(files_touched)[:10],
                        tags=["codex"],
                    )
                )
    finally:
        con.close()

    return sessions


def preview_harvest():
    print("=== Scanning Codex, Claude Code, and AGY sessions ===\n")
    agy = harvest_agy_sessions(max_sessions=5)
    claude = harvest_claude_sessions(max_sessions=5)
    codex = harvest_codex_sessions(max_sessions=5)

    print(f"✅ Found {len(agy)} recent AGY sessions:")
    for s in agy[:3]:
        print(f"  [{s.host}] ID: {s.session_id[:12]}... | Goal: {s.user_goal[:60]}... | Files: {s.files_touched[:3]}")

    print(f"\n✅ Found {len(claude)} recent Claude Code sessions:")
    for s in claude[:3]:
        print(
            f"  [{s.host}] ID: {s.session_id[:12]}... | Project: {s.project_name} | Goal: {s.user_goal[:60]}... | Files: {s.files_touched[:3]}"
        )

    print(f"\n✅ Found {len(codex)} recent Codex sessions:")
    for s in codex[:3]:
        print(f"  [{s.host}] ID: {s.session_id[:12]}... | Goal: {s.user_goal[:60]}... | Files: {s.files_touched[:3]}")


if __name__ == "__main__":
    preview_harvest()
