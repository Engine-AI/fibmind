"""Persistence helpers for FibMind."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fibmind.graph import FibMind
from fibmind.models import Edge, MemoryNode, MemoryTree


class JsonStore:
    """Simple JSON persistence for prototypes and tests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, memory: FibMind) -> None:
        payload: dict[str, Any] = {
            "nodes": [node.to_dict() for node in memory.nodes.values()],
            "edges": [edge.to_dict() for edge in memory.edges.values()],
            "trees": [tree.to_dict() for tree in memory.trees.values()],
            "category_roots": memory.category_roots,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        self._atomic_write(serialized)

    def _atomic_write(self, serialized: str) -> None:
        """Write to a temp file in the same directory, then atomically replace.

        This prevents readers (or a crashing writer) from ever observing a
        partially written store, which matters for a long-running MCP server.
        """
        fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def load(self) -> FibMind:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        memory = FibMind()
        memory.nodes = {item["id"]: MemoryNode.from_dict(item) for item in payload.get("nodes", [])}
        memory.edges = {item["id"]: Edge.from_dict(item) for item in payload.get("edges", [])}
        memory.trees = {item["id"]: MemoryTree.from_dict(item) for item in payload.get("trees", [])}
        memory.category_roots = dict(payload.get("category_roots", {}))
        memory.rebuild_indices()
        return memory
