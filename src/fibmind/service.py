"""Persistence and orchestration layer for FibMind.

``MemoryService`` coordinates access to a JSON or SQLite store, guards calls
with a process-local lock, and uses store transactions for every mutation.
Every method returns plain JSON-serializable dicts so the MCP layer stays thin.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

from fibmind.admission import AdmitVerdict, MemoryCandidate, decide_admission
from fibmind.context import build_context
from fibmind.graph import FibMind, SearchHit
from fibmind.models import (
    EdgeDirection,
    MemoryNode,
    MemoryScope,
    MemoryStatus,
    RelationType,
    TraversalDirection,
    Verdict,
)
from fibmind.ranking import ScoredHit
from fibmind.storage import MemoryStore, open_store


def _node_summary(node: MemoryNode) -> dict[str, Any]:
    return {
        "node_id": node.id,
        "title": node.title,
        "category": node.category,
        "layer": node.layer,
        "node_type": node.node_type.value,
        "tags": sorted(node.tags),
        "scope": node.scope.value,
        "owner": node.owner,
        "workspace_id": node.workspace_id,
        "project_id": node.project_id,
        "session_id": node.session_id,
        "task_id": node.task_id,
        "status": node.status.value,
        "status_reason": node.status_reason,
        "memory_kind": node.memory_kind.value if node.memory_kind else None,
        "confidence": round(node.confidence, 4),
        "confidence_source": node.confidence_source,
        "familiarity": round(node.familiarity, 4),
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
    data["via_direction"] = hit.via_direction.value if hit.via_direction else None
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


def _parse_traversal_direction(direction: str) -> TraversalDirection:
    try:
        return TraversalDirection(direction)
    except ValueError as exc:
        valid = ", ".join(item.value for item in TraversalDirection)
        raise ValueError(f"Unknown traversal direction. Valid values: {valid}") from exc


def _parse_scopes(scopes: list[str] | None) -> set[MemoryScope] | None:
    if not scopes:
        return None
    try:
        return {MemoryScope(value) for value in scopes}
    except ValueError as exc:
        valid = ", ".join(item.value for item in MemoryScope)
        raise ValueError(f"Unknown scope. Valid values: {valid}") from exc


def _parse_statuses(statuses: list[str] | None) -> set[MemoryStatus] | None:
    if not statuses:
        return None
    try:
        return {MemoryStatus(value) for value in statuses}
    except ValueError as exc:
        valid = ", ".join(item.value for item in MemoryStatus)
        raise ValueError(f"Unknown status. Valid values: {valid}") from exc


def _parse_status(status: str) -> MemoryStatus:
    try:
        return MemoryStatus(status)
    except ValueError as exc:
        valid = ", ".join(item.value for item in MemoryStatus)
        raise ValueError(f"Unknown status. Valid values: {valid}") from exc


def _parse_scope(scope: str) -> MemoryScope:
    try:
        return MemoryScope(scope)
    except ValueError as exc:
        valid = ", ".join(item.value for item in MemoryScope)
        raise ValueError(f"Unknown scope. Valid values: {valid}") from exc


def _parse_verdict(verdict: str) -> Verdict:
    try:
        return Verdict(verdict)
    except ValueError as exc:
        valid = ", ".join(item.value for item in Verdict)
        raise ValueError(f"Unknown verdict. Valid values: {valid}") from exc


def _require_weight(weight: float) -> float:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0.0 and 1.0")
    return weight


class MemoryService:
    """Thread-safe facade over a persisted FibMind memory forest."""

    def __init__(self, store_path: str | Path) -> None:
        self._store: MemoryStore = open_store(store_path)
        self._lock = threading.RLock()

    def read(self, fn: Callable[[FibMind], T]) -> T:
        """Run ``fn`` against a loaded forest without persisting anything."""
        with self._lock:
            return fn(self._store.load())

    def run(self, fn: Callable[[FibMind], T]) -> T:
        """Run ``fn`` inside one store transaction; changes persist on return."""
        with self._lock, self._store.transaction() as memory:
            return fn(memory)

    def append(
        self,
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        scope: str = MemoryScope.PERSONAL.value,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Store a new memory and persist it. Returns the created node summary."""
        _require_text(category, "category")
        _require_text(title, "title")
        _require_text(content, "content")
        parsed_scope = _parse_scope(scope)
        if parsed_scope == MemoryScope.KNOWLEDGE:
            raise ValueError(
                "knowledge scope is reached through promote_knowledge, which requires "
                "supporting observations; append writes personal or session memories"
            )

        with self._lock, self._store.transaction() as memory:
            node_id = memory.append(
                category=category,
                title=title,
                content=content,
                tags=tags,
                metadata=metadata,
                scope=parsed_scope,
                owner=owner,
                workspace_id=workspace_id,
                project_id=project_id,
                session_id=session_id,
                task_id=task_id,
            )
            tree_id = memory.category_roots[category]
            return {**_node_summary(memory.nodes[node_id]), "tree_id": tree_id}

    def admit(self, candidate: MemoryCandidate) -> dict[str, Any]:
        """Preview whether ``candidate`` would be written. Does not persist."""
        with self._lock:
            return decide_admission(self._store.load(), candidate).to_dict()

    def remember(
        self,
        candidate: MemoryCandidate,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Admit then write inside one transaction. Skips are not persisted."""
        with self._lock, self._store.transaction() as memory:
            decision = decide_admission(memory, candidate)
            payload = decision.to_dict()
            if decision.verdict != AdmitVerdict.WRITE:
                return payload
            node_id = memory.append(
                category=candidate.category,
                title=candidate.title,
                content=candidate.content,
                tags=candidate.tags,
                metadata=metadata,
                scope=candidate.scope,
                owner=candidate.owner,
                workspace_id=candidate.workspace_id,
                project_id=candidate.project_id,
                session_id=candidate.session_id,
                task_id=candidate.task_id,
            )
            payload.update(_node_summary(memory.nodes[node_id]))
            payload["tree_id"] = memory.category_roots[candidate.category]
            return payload

    def inspect(self, node_id: str) -> dict[str, Any]:
        """Return one node including content and metadata."""
        _require_text(node_id, "node_id")
        with self._lock:
            memory = self._store.load()
            node = memory.nodes.get(node_id)
            if node is None:
                raise ValueError(f"unknown node: {node_id}")
            return {
                **_node_summary(node),
                "content": node.content,
                "metadata": dict(node.metadata),
            }

    def record_outcome(
        self,
        node_id: str,
        verdict: str,
        source: str,
        note: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Attach external evidence for or against a memory.

        The only way ``confidence`` — the single trust signal ranking uses —
        ever moves.
        """
        parsed = _parse_verdict(verdict)
        _require_text(source, "source")
        with self._lock, self._store.transaction() as memory:
            try:
                node = memory.record_outcome(
                    node_id, parsed, source=source, note=note, session_id=session_id
                )
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            return _node_summary(node)

    def revise(
        self,
        node_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Correct a memory. Its accumulated confidence is reset."""
        if title is None and content is None and tags is None:
            raise ValueError("revise needs at least one of title, content, or tags")
        with self._lock, self._store.transaction() as memory:
            try:
                node = memory.revise(node_id, title=title, content=content, tags=tags)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            return _node_summary(node)

    def set_status(self, node_id: str, status: str, reason: str) -> dict[str, Any]:
        """Move a memory between lifecycle states (approve / reject a pending one)."""
        parsed = _parse_status(status)
        _require_text(reason, "reason")
        with self._lock, self._store.transaction() as memory:
            try:
                return _node_summary(memory.set_status(node_id, parsed, reason))
            except KeyError as exc:
                raise ValueError(str(exc)) from exc

    def mark_stale(self, node_id: str, reason: str) -> dict[str, Any]:
        """Retire an outdated memory while preserving its provenance."""
        _require_text(reason, "reason")
        with self._lock, self._store.transaction() as memory:
            try:
                return _node_summary(memory.mark_stale(node_id, reason))
            except KeyError as exc:
                raise ValueError(str(exc)) from exc

    def forget(self, node_id: str, reason: str) -> dict[str, Any]:
        """Delete a memory and redact it from the event log."""
        _require_text(reason, "reason")
        with self._lock, self._store.transaction() as memory:
            try:
                return memory.forget(node_id, reason=reason)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc

    def promote_knowledge(
        self,
        title: str,
        content: str,
        supporting_node_ids: list[str],
        category: str = "knowledge",
        tags: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Distil several observations into one shareable claim."""
        _require_text(title, "title")
        _require_text(content, "content")
        if not supporting_node_ids:
            raise ValueError("supporting_node_ids must list the observations behind the claim")
        with self._lock, self._store.transaction() as memory:
            try:
                node_id = memory.promote_to_knowledge(
                    title=title,
                    content=content,
                    supporting_node_ids=supporting_node_ids,
                    category=category,
                    tags=tags,
                    workspace_id=workspace_id,
                    project_id=project_id,
                    session_id=session_id,
                    task_id=task_id,
                )
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            return _node_summary(memory.nodes[node_id])

    def link(
        self,
        from_node_id: str,
        to_node_id: str,
        relation_type: str,
        weight: float = 1.0,
        bidirectional: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a typed edge between two existing nodes and persist it."""
        relation = _parse_relation_types([relation_type])
        assert relation is not None
        weight = _require_weight(weight)
        direction = EdgeDirection.BIDIRECTIONAL if bidirectional else EdgeDirection.DIRECTED
        with self._lock, self._store.transaction() as memory:
            try:
                edge_id = memory.link_nodes(
                    from_node_id,
                    to_node_id,
                    next(iter(relation)),
                    weight=weight,
                    direction=direction,
                    metadata=metadata,
                )
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            edge = memory.edges[edge_id]
            return {
                "edge_id": edge.id,
                "from_node_id": edge.from_node_id,
                "to_node_id": edge.to_node_id,
                "relation_type": edge.relation_type.value,
                "weight": edge.weight,
                "direction": edge.direction.value,
            }

    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: list[str] | None = None,
        min_score: float = 0.0,
        scopes: list[str] | None = None,
        owner: str | None = None,
        statuses: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Read-only keyword search over the whole forest."""
        _require_text(query, "query")
        category_set = set(categories) if categories else None
        scope_set = _parse_scopes(scopes)
        status_set = _parse_statuses(statuses)
        with self._lock:
            memory = self._store.load()
            hits = memory.search(
                query,
                top_k=top_k,
                categories=category_set,
                min_score=min_score,
                scopes=scope_set,
                owner=owner,
                statuses=status_set,
                workspace_id=workspace_id,
                project_id=project_id,
                session_id=session_id,
            )
            return {"query": query, "results": [_scored_hit_dict(hit) for hit in hits]}

    def search_from(
        self,
        node_id: str,
        depth: int = 2,
        relation_types: list[str] | None = None,
        reinforce: bool = False,
        direction: str = TraversalDirection.BOTH.value,
        scopes: list[str] | None = None,
        owner: str | None = None,
        statuses: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Expand the association graph rooted at ``node_id``."""
        relations = _parse_relation_types(relation_types)
        traversal = _parse_traversal_direction(direction)
        scope_set = _parse_scopes(scopes)
        status_set = _parse_statuses(statuses)

        def execute(memory: FibMind) -> dict[str, Any]:
            try:
                hits = memory.search_from(
                    node_id,
                    depth=depth,
                    relation_types=relations,
                    reinforce=reinforce,
                    direction=traversal,
                    scopes=scope_set,
                    owner=owner,
                    statuses=status_set,
                    workspace_id=workspace_id,
                    project_id=project_id,
                    session_id=session_id,
                )
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            return {
                "root": node_id,
                "direction": traversal.value,
                "hits": [_search_hit_dict(hit) for hit in hits],
            }

        with self._lock:
            if reinforce:
                with self._store.transaction() as memory:
                    return execute(memory)
            return execute(self._store.load())

    def context(
        self,
        goal: str,
        top_k: int = 5,
        depth: int = 1,
        max_chars: int = 2000,
        reinforce: bool = False,
        scopes: list[str] | None = None,
        owner: str | None = None,
        statuses: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        exclude_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a context-window-ready memory pack for a task goal."""
        _require_text(goal, "goal")
        scope_set = _parse_scopes(scopes)
        status_set = _parse_statuses(statuses)
        excluded = set(exclude_categories) if exclude_categories else None
        with self._lock:
            if reinforce:
                with self._store.transaction() as memory:
                    return build_context(
                        memory,
                        goal,
                        top_k=top_k,
                        depth=depth,
                        max_chars=max_chars,
                        reinforce=True,
                        scopes=scope_set,
                        owner=owner,
                        statuses=status_set,
                        workspace_id=workspace_id,
                        project_id=project_id,
                        session_id=session_id,
                        exclude_categories=excluded,
                    ).to_dict()
            return build_context(
                self._store.load(),
                goal,
                top_k=top_k,
                depth=depth,
                max_chars=max_chars,
                reinforce=False,
                scopes=scope_set,
                owner=owner,
                statuses=status_set,
                workspace_id=workspace_id,
                project_id=project_id,
                session_id=session_id,
                exclude_categories=excluded,
            ).to_dict()
