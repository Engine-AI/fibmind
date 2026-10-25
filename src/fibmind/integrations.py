"""Generate portable MCP launch configuration for independent agent hosts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fibmind.migrate import DEFAULT_HOST_OWNERS

HOSTS = ("codex", "claude-code", "agy")


def server_config(
    host: str,
    *,
    owner: str,
    store: str | Path,
    python: str | Path = sys.executable,
    workspace_id: str | None = None,
    project_id: str | None = None,
    profile: str = "memory",
) -> dict[str, Any]:
    """All hosts use one absolute store and pin the human, never the host."""
    if host not in HOSTS:
        raise ValueError(f"host must be one of {HOSTS}")
    if profile not in {"memory", "full"}:
        raise ValueError("profile must be memory or full")
    owner = owner.strip()
    if not owner or owner in DEFAULT_HOST_OWNERS:
        raise ValueError("owner must name the human, not an AI application")
    # Keep the venv symlink: resolving it loses that environment's packages.
    interpreter = os.path.abspath(os.path.expanduser(str(python)))
    if not Path(interpreter).is_file():
        raise ValueError(f"Python executable does not exist: {interpreter}")
    env = {"FIBMIND_OWNER": owner, "FIBMIND_APPLICATION": host}
    if workspace_id:
        env["FIBMIND_WORKSPACE_ID"] = workspace_id
    if project_id:
        env["FIBMIND_PROJECT_ID"] = project_id
    return {
        "command": interpreter,
        "args": [
            "-m",
            "fibmind.mcp_server",
            "--store",
            str(Path(store).expanduser().resolve()),
            "--owner",
            owner,
            "--profile",
            profile,
        ],
        "env": env,
    }


def render_config(host: str, server: dict[str, Any]) -> str:
    if host != "codex":
        return json.dumps({"mcpServers": {"fibmind": server}}, ensure_ascii=False, indent=2) + "\n"

    # JSON string/array literals are valid TOML basic strings/arrays. Env keys
    # are generated constants; no arbitrary TOML fragments are interpolated.
    def encode(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    lines = [
        "[mcp_servers.fibmind]",
        f"command = {encode(server['command'])}",
        f"args = {encode(server['args'])}",
        "",
        "[mcp_servers.fibmind.env]",
    ]
    lines.extend(f"{key} = {encode(value)}" for key, value in server["env"].items())
    return "\n".join(lines) + "\n"


def instructions(owner: str) -> str:
    return f"""# Shared FibMind memory

The human owner is {json.dumps(owner, ensure_ascii=False)}. Application names
belong in `metadata.host`; never use them as memory owners.

Before answering each user request, call `fibbrain_context` with the current
request as `goal`, the absolute current workspace directory as `workspace_path`,
and this host's native conversation ID as `session_id`. Supply `owner` above.
If the host does not expose that ID, omit it and avoid session-scope memories;
never invent a conversation ID or reuse one from a different conversation.
If the user named a task, pass it as `task_id`; otherwise the server derives
the git branch. Existing explicit workspace/project labels override inference.
The server also reads `.fibmind-project.json` at the workspace root for stable
labels when a project already has stored memories under those labels.
Use the returned `state` for subsequent memory operations across all hosts.

Use `text` as historical background. Current user instructions take precedence.
`unchecked` facts need verification; `confirmed` means past evidence, not a new
check. When `has_task_context` is false, profile may still apply, but do not
invent task-specific knowledge from it. The memory service supplies context;
your harness controls planning and tool execution.

Save durable user background with `fibbrain_remember`, category `profile` or
`preference`, and explicit `context_level="user"`. Use `context_level="project"`
for project decisions/errors and `"workspace"` only for facts applying to every
project in that workspace. Keep temporary scratch in `scope="session"` with
the returned session/task labels. Never store credentials or raw logs.

When a fact changes, revise or retire its existing node rather than saving a
contradiction. Attach actual tests or user corrections with `fibbrain_reflect`.
Never confirm a memory merely because it was recalled or another model agreed.
"""


def write_bundle(
    output: Path,
    *,
    owner: str,
    store: str | Path,
    python: str | Path = sys.executable,
    workspace_id: str | None = None,
    project_id: str | None = None,
    profile: str = "memory",
) -> list[Path]:
    prepared = {}
    for host in HOSTS:
        config = server_config(
            host,
            owner=owner,
            store=store,
            python=python,
            workspace_id=workspace_id,
            project_id=project_id,
            profile=profile,
        )
        filename = "codex.toml" if host == "codex" else f"{host}.json"
        prepared[filename] = render_config(host, config)
    prepared["memory-instructions.md"] = instructions(owner)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, content in prepared.items():
        path = output / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate shared FibMind MCP configuration for Codex, Claude Code, and AGY"
    )
    parser.add_argument("--owner", required=True, help="The human who owns these memories")
    parser.add_argument(
        "--store",
        default=str(Path.home() / ".fibmind" / "memory.db"),
        help="One shared SQLite store; use an existing store path to keep its memories",
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Interpreter with fibmind installed (venv symlink is preserved)"
    )
    parser.add_argument("--workspace-id", help="Optional established workspace label shared by all hosts")
    parser.add_argument("--project-id", help="Optional established project label shared by all hosts")
    parser.add_argument(
        "--profile",
        choices=["memory", "full"],
        default="memory",
        help="Use full for existing AGENTS workflows requiring planning/cognitive tools",
    )
    parser.add_argument("--output-dir", type=Path, help="Write all three configs and context-use instructions")
    parser.add_argument("--host", choices=HOSTS, help="Print one host's configuration")
    args = parser.parse_args(argv)
    values = {
        "owner": args.owner,
        "store": args.store,
        "python": args.python,
        "workspace_id": args.workspace_id,
        "project_id": args.project_id,
        "profile": args.profile,
    }
    try:
        if args.output_dir:
            paths = write_bundle(args.output_dir, **values)
            print(
                json.dumps(
                    {
                        "files": [str(path.resolve()) for path in paths],
                        "store": str(Path(args.store).expanduser().resolve()),
                    },
                    indent=2,
                )
            )
        elif args.host:
            print(render_config(args.host, server_config(args.host, **values)), end="")
        else:
            parser.error("supply --host or --output-dir")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
