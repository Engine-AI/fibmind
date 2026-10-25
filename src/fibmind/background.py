"""Read-only, host-neutral background about a human and their current work."""

from __future__ import annotations

from typing import Any

from fibmind.context import ContextHit, _render_sections
from fibmind.embedding import HashingEmbeddingProvider
from fibmind.graph import FibMind
from fibmind.models import MemoryKind, MemoryNode, MemoryScope, NodeType, optional_id
from fibmind.ranking import tokenize
from fibmind.retrieval import retrieve
from fibmind.safety import scan_candidate, scan_text
from fibmind.tokens import DEFAULT_COUNTER, TokenCounter
from fibmind.visibility import context_level

INTERNAL_CATEGORIES = {"episode", "goal", "hot"}
PROFILE_KINDS = {MemoryKind.PREFERENCE, MemoryKind.PROFILE}


def _applicable(node: MemoryNode, model_id: str | None) -> bool:
    if node.node_type == NodeType.ROOT or node.category in INTERNAL_CATEGORIES or node.metadata.get("conflicted"):
        return False
    if node.metadata.get("grounding") == "model_heuristic":
        if not model_id or node.metadata.get("model_id") != model_id:
            return False
    return not scan_candidate(node.title, node.content, node.tags).blocked and not scan_text(
        _line(ContextHit(node, 0, "check")), "background"
    )


def _evidence(node: MemoryNode) -> str:
    return "confirmed" if node.confidence > 0 else "unchecked"


def _line(hit: ContextHit) -> str:
    node = hit.node
    level = context_level(node).value if node.scope != MemoryScope.KNOWLEDGE else "knowledge"
    parts = [f"- [{level}; {_evidence(node)}] {node.title}:"]
    # Constraints precede the body so they survive a shortened excerpt.
    for label, keys in (("Conditions", ("conditions", "preconditions")), ("Exceptions", ("exceptions", "risks"))):
        value = next((node.metadata.get(key) for key in keys if node.metadata.get(key)), None)
        if isinstance(value, (tuple, list)):
            value = "; ".join(str(item) for item in value)
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}: {value.strip()}.")
    parts.append(" ".join(node.content.split()))
    return " ".join(parts)


def _entry(hit: ContextHit) -> dict[str, Any]:
    node = hit.node
    sources = node.metadata.get("source_hosts", [])
    hosts = {str(host)[:80] for host in sources if host} if isinstance(sources, (list, tuple)) else set()
    if node.metadata.get("host"):
        hosts.add(str(node.metadata["host"])[:80])
    hosts = {host for host in hosts if not scan_text(host, "host")}
    return {
        "node_id": node.id,
        "section": hit.source,
        "context_level": context_level(node).value if node.scope != MemoryScope.KNOWLEDGE else "knowledge",
        "scope": node.scope.value,
        "memory_kind": node.memory_kind.value if node.memory_kind else None,
        "confidence": round(node.confidence, 4),
        "evidence": _evidence(node),
        "hosts": sorted(hosts),
    }


def build_background(
    memory: FibMind,
    goal: str = "",
    *,
    owner: str,
    workspace_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    model_id: str | None = None,
    top_k: int = 5,
    profile_top_k: int = 3,
    budget_tokens: int = 500,
    min_relevance: float = 0.12,
    counter: TokenCounter = DEFAULT_COUNTER,
) -> dict[str, Any]:
    """Combine applicable profile with relevant facts, without mutating memory.

    Confidence describes historical evidence, never fresh verification. The
    budget applies to ``text``; ``memories`` is a compact provenance manifest
    containing only IDs rendered in that text. Empty task context is valid.
    """
    owner = optional_id(owner)
    if owner is None:
        raise ValueError("context requires the human owner's identity")
    if top_k < 1 or profile_top_k < 0 or budget_tokens < 1:
        raise ValueError("top_k and budget_tokens must be positive; profile_top_k must be non-negative")
    if not 0 <= min_relevance <= 1:
        raise ValueError("min_relevance must be between 0 and 1")
    identity = {
        "owner": owner,
        "workspace_id": optional_id(workspace_id),
        "project_id": optional_id(project_id),
        "session_id": optional_id(session_id),
        "task_id": optional_id(task_id),
    }
    profiles = sorted(
        (
            node
            for node in memory.nodes.values()
            if node.scope == MemoryScope.PERSONAL
            and node.memory_kind in PROFILE_KINDS
            and memory.is_visible(node, **identity)
            and _applicable(node, model_id)
        ),
        key=lambda node: (node.confidence, node.updated_at, node.id),
        reverse=True,
    )[:profile_top_k]
    profile_hits = [ContextHit(node, node.confidence, "profile") for node in profiles]
    profile_ids = {node.id for node in profiles}
    task_hits = []
    terms = set(tokenize(goal))
    total_idf = sum(memory.lexical.idf(term) for term in terms)
    if terms:
        candidates, _ = retrieve(
            memory,
            goal,
            top_k=len(memory.nodes) or 1,
            exclude_categories=INTERNAL_CATEGORIES,
            embeddings=memory.embeddings,
            **identity,
        )
        provider = memory.embeddings.provider if memory.embeddings is not None else None
        for candidate in candidates:
            node = candidate.node
            if node.id in profile_ids or not _applicable(node, model_id):
                continue
            coverage = sum(memory.lexical.idf(term) for term in candidate.matched_terms) / total_idf if total_idf else 0
            semantic = (
                provider is not None
                and not isinstance(provider, HashingEmbeddingProvider)
                and candidate.vector is not None
                and candidate.vector >= provider.min_similarity
            )
            if coverage < min_relevance and not semantic:
                continue
            task_hits.append(ContextHit(node, candidate.score, "task", matched_terms=candidate.matched_terms))
            if len(task_hits) >= top_k:
                break
    profile_budget = budget_tokens // 4 if task_hits else budget_tokens
    text, budget = _render_sections(
        [("profile", profile_hits, profile_budget), ("task", task_hits, budget_tokens - profile_budget)],
        counter,
        budget_tokens,
        line_formatter=_line,
    )
    by_id = {hit.node.id: hit for hit in [*profile_hits, *task_hits]}
    rendered = [by_id[node_id] for node_id in budget.rendered_node_ids]
    return {
        "goal": goal,
        "text": text,
        "memories": [_entry(hit) for hit in rendered],
        "has_task_context": any(hit.source == "task" for hit in rendered),
        "budget": budget.to_dict(),
        "state": identity,
        "usage": "Historical background only. Current user instructions take precedence; unchecked memories need verification.",
    }
