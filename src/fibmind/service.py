"""Persistence and orchestration layer for FibMind.

``MemoryService`` owns a single :class:`FibMind` instance backed by a JSON
store. It loads existing state on construction, guards all access with a lock
(the MCP server may dispatch tool calls from a worker thread), and persists
after every mutation. Every method returns plain JSON-serializable dicts so the
MCP tool layer can stay a thin wrapper.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fibmind.context import build_context
from fibmind.graph import FibMind, SearchHit
from fibmind.models import EdgeDirection, MemoryNode, RelationType
from fibmind.ranking import ScoredHit
from fibmind.storage import JsonStore


def _node_summary(node: MemoryNode) -> dict[str, Any]:
    return {
        "node_id": node.id,
        "title": node.title,
        "category": node.category,
        "layer": node.layer,
        "node_type": node.node_type.value,
        "tags": sorted(node.tags),
        "importance": round(node.importance, 4),
        "access_count": node.access_count,
    }


def _scored_hit_dict(hit: ScoredHit) -> dict[str, Any]:
    data = _node_summary(hit.node)
    data["score"] = round(hit.score, 4)
    data["matched_terms"] = list(hit.matched_terms)
    return data


def _search_hit_dict(hit: SearchHit) -> dict[str, Any]:
    data = _node_summary(hit.node)
    data["depth"] = hit.depth
    data["via_relation"] = hit.via_relation.value if hit.via_relation else None
    return data


def _require_text(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _parse_relation_types(relation_types: list[str] | None) -> set[RelationType] | None:
    if not relation_types:
        return None
    try:
        return {RelationType(value) for value in relation_types}
    except ValueError as exc:
        valid = ", ".join(sorted(rt.value for rt in RelationType))
        raise ValueError(f"Unknown relation type. Valid values: {valid}") from exc


class MemoryService:
    """Thread-safe facade over a persisted FibMind memory forest."""

    def __init__(self, store_path: str | Path) -> None:
        self._store = JsonStore(store_path)
        self._lock = threading.Lock()
        path = Path(store_path)
        self._memory = self._store.load() if path.exists() else FibMind()

    def _save(self) -> None:
        self._store.save(self._memory)

    def append(
        self,
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new memory and persist it. Returns the created node summary."""
        _require_text(category, "category")
        _require_text(title, "title")
        _require_text(content, "content")

        with self._lock:
            node_id = self._memory.append(
                category=category,
                title=title,
                content=content,
                tags=tags,
                metadata=metadata,
            )
            tree_id = self._memory.category_roots[category]
            self._save()
            return {**_node_summary(self._memory.nodes[node_id]), "tree_id": tree_id}

    def link(
        self,
        from_node_id: str,
        to_node_id: str,
        relation_type: str,
        weight: float = 1.0,
        bidirectional: bool = False,
    ) -> dict[str, Any]:
        """Create a typed edge between two existing nodes and persist it."""
        relation = _parse_relation_types([relation_type])
        assert relation is not None  # relation_type is required here
        direction = EdgeDirection.BIDIRECTIONAL if bidirectional else EdgeDirection.DIRECTED
        with self._lock:
            try:
                edge_id = self._memory.link_nodes(
                    from_node_id,
                    to_node_id,
                    next(iter(relation)),
                    weight=weight,
                    direction=direction,
                )
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            self._save()
            return {"edge_id": edge_id}

    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: list[str] | None = None,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Read-only keyword search over the whole forest."""
        _require_text(query, "query")
        category_set = set(categories) if categories else None
        with self._lock:
            hits = self._memory.search(
                query,
                top_k=top_k,
                categories=category_set,
                min_score=min_score,
            )
            return {"query": query, "results": [_scored_hit_dict(hit) for hit in hits]}

    def search_from(
        self,
        node_id: str,
        depth: int = 2,
        relation_types: list[str] | None = None,
        reinforce: bool = False,
    ) -> dict[str, Any]:
        """Expand the association tree rooted at ``node_id``."""
        relations = _parse_relation_types(relation_types)
        with self._lock:
            try:
                hits = self._memory.search_from(
                    node_id,
                    depth=depth,
                    relation_types=relations,
                    reinforce=reinforce,
                )
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            if reinforce:
                self._save()
            return {"root": node_id, "hits": [_search_hit_dict(hit) for hit in hits]}

    def context(
        self,
        goal: str,
        top_k: int = 5,
        depth: int = 1,
        max_chars: int = 2000,
        reinforce: bool = False,
    ) -> dict[str, Any]:
        """Build a context-window-ready memory pack for a task goal."""
        _require_text(goal, "goal")
        with self._lock:
            pack = build_context(
                self._memory,
                goal,
                top_k=top_k,
                depth=depth,
                max_chars=max_chars,
                reinforce=reinforce,
            )
            if reinforce:
                self._save()
            return pack.to_dict()
