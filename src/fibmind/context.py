"""Build a compact, context-window-ready memory pack for a task goal.

The pack is produced by keyword-searching for the goal and then expanding the
top hits along graph edges, so the caller receives both directly relevant
memories and their immediate neighbours. Rendering is bounded by a character
budget so the result can be dropped straight into a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fibmind.graph import FibMind
from fibmind.models import MemoryNode, MemoryScope, MemoryStatus, NodeType

DEFAULT_TOP_K = 5
DEFAULT_DEPTH = 1
DEFAULT_MAX_CHARS = 2000
EXCERPT_CHARS = 160

SOURCE_SEARCH = "search"
SOURCE_EXPAND = "expand"


@dataclass(frozen=True, slots=True)
class ContextHit:
    """A single memory selected for the context pack."""

    node: MemoryNode
    score: float
    source: str  # SOURCE_SEARCH or SOURCE_EXPAND
    depth: int = 0
    matched_terms: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node.id,
            "title": self.node.title,
            "category": self.node.category,
            "layer": self.node.layer,
            "node_type": self.node.node_type.value,
            "workspace_id": self.node.workspace_id,
            "project_id": self.node.project_id,
            "session_id": self.node.session_id,
            "task_id": self.node.task_id,
            "score": round(self.score, 4),
            "source": self.source,
            "depth": self.depth,
            "matched_terms": list(self.matched_terms),
            "excerpt": _excerpt(self.node.content),
        }


@dataclass(frozen=True, slots=True)
class ContextPack:
    """The retrieved memories plus a rendered, budget-bounded text block."""

    goal: str
    hits: list[ContextHit]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "text": self.text,
            "hits": [hit.to_dict() for hit in self.hits],
        }


def _excerpt(content: str, limit: int = EXCERPT_CHARS) -> str:
    flat = " ".join(content.split())
    if len(flat) <= limit:
        return flat
    return f"{flat[: limit - 3]}..."


def _render(hits: list[ContextHit], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for hit in hits:
        line = f"- [{hit.node.category}] {hit.node.title}: {_excerpt(hit.node.content)}"
        separator = 1 if parts else 0
        remaining = max_chars - used - separator
        if remaining <= 0:
            break
        parts.append(line[:remaining])
        used += len(parts[-1]) + separator
    return "\n".join(parts)


def build_context(
    memory: FibMind,
    goal: str,
    top_k: int = DEFAULT_TOP_K,
    depth: int = DEFAULT_DEPTH,
    max_chars: int = DEFAULT_MAX_CHARS,
    reinforce: bool = False,
    scopes: set[MemoryScope] | None = None,
    owner: str | None = None,
    statuses: set[MemoryStatus] | None = None,
    workspace_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
) -> ContextPack:
    """Assemble a context pack for ``goal``.

    Seeds come from keyword search; each seed is expanded ``depth`` hops along
    graph edges. Nodes are de-duplicated, keeping the strongest provenance
    (a direct search hit outranks an expansion hit). Read-only unless
    ``reinforce`` is set, in which case traversed nodes are activated.
    """
    seeds = memory.search(
        goal,
        top_k=top_k,
        scopes=scopes,
        owner=owner,
        statuses=statuses,
        workspace_id=workspace_id,
        project_id=project_id,
        session_id=session_id,
    )
    if not seeds:
        return ContextPack(goal=goal, hits=[], text="")

    hits: list[ContextHit] = []
    seen: set[str] = set()

    for seed in seeds:
        hits.append(
            ContextHit(
                node=seed.node,
                score=seed.score,
                source=SOURCE_SEARCH,
                depth=0,
                matched_terms=seed.matched_terms,
            )
        )
        seen.add(seed.node.id)

    if depth > 0:
        for seed in seeds:
            for expansion in memory.search_from(
                seed.node.id,
                depth=depth,
                reinforce=reinforce,
                scopes=scopes,
                owner=owner,
                statuses=statuses,
                workspace_id=workspace_id,
                project_id=project_id,
                session_id=session_id,
            ):
                if expansion.node.id in seen:
                    continue
                seen.add(expansion.node.id)
                if expansion.node.node_type == NodeType.ROOT:
                    continue
                if not memory.is_visible(
                    expansion.node,
                    scopes=scopes,
                    owner=owner,
                    statuses=statuses,
                    workspace_id=workspace_id,
                    project_id=project_id,
                    session_id=session_id,
                ):
                    continue
                hits.append(
                    ContextHit(
                        node=expansion.node,
                        score=0.0,
                        source=SOURCE_EXPAND,
                        depth=expansion.depth,
                    )
                )
    elif reinforce:
        for seed in seeds:
            memory.search_from(
                seed.node.id,
                depth=0,
                reinforce=True,
                scopes=scopes,
                owner=owner,
                statuses=statuses,
                workspace_id=workspace_id,
                project_id=project_id,
                session_id=session_id,
            )

    text = _render(hits, max_chars)
    return ContextPack(goal=goal, hits=hits, text=text)
