"""One filesystem-based working identity for every local MCP client."""

from __future__ import annotations

import json
from pathlib import Path


def working_context(workspace_path: str) -> dict[str, str | None]:
    """Resolve a local directory to stable workspace/project/task labels.

    Canonical absolute workspace paths avoid basename collisions and symlink
    differences between harnesses. Explicit caller/server labels can override
    these defaults when a project already uses established labels.
    """
    path = Path(workspace_path).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("workspace_path must be an existing absolute directory on the memory server")
    path = path.resolve()
    root = next(
        (p for p in (path, *path.parents) if (p / ".git").exists() or (p / ".fibmind-project.json").is_file()), path
    )
    labels = {"workspace_id": str(root), "project_id": root.name}
    config = root / ".fibmind-project.json"
    if config.is_file():
        declared = json.loads(config.read_text(encoding="utf-8"))
        if not isinstance(declared, dict):
            raise ValueError(".fibmind-project.json must be an object")
        for key in labels:
            if key in declared:
                value = declared[key]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f".fibmind-project.json {key} must be a nonempty string")
                labels[key] = value.strip()
    git = root / ".git"
    if git.is_file():
        value = git.read_text(encoding="utf-8").strip()
        git = (root / value.removeprefix("gitdir:").strip()).resolve() if value.startswith("gitdir:") else git
    head = git / "HEAD"
    task_id = None
    if head.is_file():
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: refs/heads/"):
            task_id = value.removeprefix("ref: refs/heads/")
        elif value and all(c in "0123456789abcdef" for c in value.lower()):
            task_id = f"detached:{value[:12]}"
    return {**labels, "task_id": task_id}
