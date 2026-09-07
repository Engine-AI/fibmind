"""Build a compact, context-window-ready memory pack for a task goal.

The pack is produced by searching for the goal and then expanding the top hits
along graph edges, so the caller receives both directly relevant memories and
their immediate neighbours. Rendering is bounded by a **token budget** split
into sections — hot memory, direct hits, graph expansion — so the result can
be dropped straight into a prompt, and the pack reports what it spent, what
was left, and why anything was cut.

``max_chars`` is kept as a compatibility knob; when only it is given the
budget is derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fibmind.graph import FibMind
from fibmind.models import MemoryNode, MemoryScope, MemoryStatus, NodeType
from fibmind.tokens import DEFAULT_COUNTER, TokenCounter

DEFAULT_TOP_K = 5
DEFAULT_DEPTH = 1
DEFAULT_MAX_CHARS = 2000
EXCERPT_CHARS = 160

# Roughly four characters per token for the compatibility conversion.
CHARS_PER_TOKEN = 4
# How the total budget splits when no explicit section budgets are given.
# Direct hits are what the caller asked for; expansion is context around them;
# hot memory is the standing preamble. Unused share flows down to the next.
SHARE_HOT = 0.30
SHARE_SEARCH = 0.50
SHARE_EXPAND = 0.20

SOURCE_HOT = "hot"
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
class BudgetReport:
    """What the pack spent, section by section, and why anything was cut."""

    budget_tokens: int
    used_tokens: int
    sections: dict[str, dict[str, int]]
    truncated: list[dict[str, Any]]
    counter: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "remaining_tokens": max(0, self.budget_tokens - self.used_tokens),
            "sections": self.sections,
            "truncated": self.truncated,
            "counter": self.counter,
        }


@dataclass(frozen=True, slots=True)
class ContextPack:
    """The retrieved memories plus a rendered, budget-bounded text block."""

    goal: str
    hits: list[ContextHit]
    text: str
    budget: BudgetReport | None = None
    hot: list[ContextHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "goal": self.goal,
            "text": self.text,
            "hits": [hit.to_dict() for hit in self.hits],
            "hot": [hit.to_dict() for hit in self.hot],
        }
        if self.budget is not None:
            data["budget"] = self.budget.to_dict()
        return data


def _excerpt(content: str, limit: int = EXCERPT_CHARS) -> str:
    flat = " ".join(content.split())
    if len(flat) <= limit:
        return flat
    return f"{flat[: limit - 3]}..."


def _line(hit: ContextHit) -> str:
    return f"- [{hit.node.category}] {hit.node.title}: {_excerpt(hit.node.content)}"


def _fit(line: str, budget: int, counter: TokenCounter) -> tuple[str, int, bool]:
    """Trim ``line`` until it fits ``budget`` tokens. Returns (text, cost, cut)."""
    cost = counter.count(line)
    if cost <= budget:
        return line, cost, False
    if budget <= 0:
        return "", 0, True
    # Shrink by the token overshoot ratio, then verify; two passes is plenty.
    for _ in range(4):
        keep = max(0, int(len(line) * budget / max(cost, 1)) - 3)
        if keep <= 0:
            return "", 0, True
        line = line[:keep].rstrip() + "..."
        cost = counter.count(line)
        if cost <= budget:
            return line, cost, True
    return "", 0, True


def _render_sections(
    sections: list[tuple[str, list[ContextHit], int]],
    counter: TokenCounter,
    total: int,
) -> tuple[str, BudgetReport]:
    """Render each section inside its budget; unused budget flows downward.

    A hit is cut (excerpt shortened) only when it is the first item that does
    not fit; later items in that section are dropped rather than crammed. Every
    cut and drop is listed in ``truncated`` with the reason.
    """
    lines: list[str] = []
    used = 0
    carry = 0
    report_sections: dict[str, dict[str, int]] = {}
    truncated: list[dict[str, Any]] = []
    for name, hits, allotted in sections:
        available = allotted + carry
        spent = 0
        kept = 0
        for hit in hits:
            separator = 1 if lines else 0
            line, cost, cut = _fit(_line(hit), available - spent - separator, counter)
            if not line:
                truncated.append({"node_id": hit.node.id, "section": name, "reason": "over_budget"})
                continue
            if cut:
                truncated.append({"node_id": hit.node.id, "section": name, "reason": "excerpt_shortened"})
            lines.append(line)
            spent += cost + separator
            kept += 1
            if cut:
                # The rest of this section will not fit either; record them.
                remaining = hits[hits.index(hit) + 1 :]
                for later in remaining:
                    truncated.append({"node_id": later.node.id, "section": name, "reason": "over_budget"})
                break
        report_sections[name] = {"budget": allotted, "used": spent, "kept": kept, "offered": len(hits)}
        used += spent
        carry = max(0, available - spent)
    return "\n".join(lines), BudgetReport(
        budget_tokens=total,
        used_tokens=used,
        sections=report_sections,
        truncated=truncated,
        counter=counter.name,
    )


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
    exclude_categories: set[str] | None = None,
    budget_tokens: int | None = None,
    hot: list[MemoryNode] | None = None,
    hot_budget_tokens: int | None = None,
    counter: TokenCounter = DEFAULT_COUNTER,
) -> ContextPack:
    """Assemble a context pack for ``goal`` inside a token budget.

    Seeds come from search; each seed is expanded ``depth`` hops along graph
    edges. Nodes are de-duplicated, keeping the strongest provenance (a direct
    search hit outranks an expansion hit). ``hot`` memories, if given, render
    first inside ``hot_budget_tokens``. The total ``budget_tokens`` defaults to
    ``max_chars / 4`` so existing callers keep their bound. Read-only unless
    ``reinforce`` is set.
    """
    total_budget = budget_tokens if budget_tokens is not None else max(1, max_chars // CHARS_PER_TOKEN)
    hot_hits = [
        ContextHit(node=node, score=node.confidence, source=SOURCE_HOT, depth=0)
        for node in (hot or [])
    ]
    hot_ids = {hit.node.id for hit in hot_hits}
    seeds = memory.search(
        goal,
        top_k=top_k,
        scopes=scopes,
        owner=owner,
        statuses=statuses,
        workspace_id=workspace_id,
        project_id=project_id,
        session_id=session_id,
        exclude_categories=exclude_categories,
    )
    hits: list[ContextHit] = []
    seen: set[str] = set()

    for seed in seeds:
        seen.add(seed.node.id)
        hits.append(
            ContextHit(
                node=seed.node,
                score=seed.score,
                source=SOURCE_SEARCH,
                depth=0,
                matched_terms=seed.matched_terms,
            )
        )

    if seeds and depth > 0:
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
                if exclude_categories and expansion.node.category in exclude_categories:
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

    # A hot memory that is also a direct hit stays in ``hits`` (the caller asked
    # for it) but is rendered once, in the hot section.
    direct = [hit for hit in hits if hit.source == SOURCE_SEARCH and hit.node.id not in hot_ids]
    expanded = [hit for hit in hits if hit.source == SOURCE_EXPAND and hit.node.id not in hot_ids]
    if hot_budget_tokens is None:
        hot_budget_tokens = int(total_budget * SHARE_HOT) if hot_hits else 0
    remaining = max(0, total_budget - hot_budget_tokens)
    if expanded:
        search_budget = int(remaining * SHARE_SEARCH / (SHARE_SEARCH + SHARE_EXPAND))
        expand_budget = remaining - search_budget
    else:
        search_budget, expand_budget = remaining, 0
    text, budget = _render_sections(
        [
            (SOURCE_HOT, hot_hits, hot_budget_tokens),
            (SOURCE_SEARCH, direct, search_budget),
            (SOURCE_EXPAND, expanded, expand_budget),
        ],
        counter,
        total_budget,
    )
    if budget_tokens is None and len(text) > max_chars:
        # Compatibility: an explicit character cap is still honoured exactly.
        text = text[:max_chars]
    return ContextPack(goal=goal, hits=hits, text=text, budget=budget, hot=hot_hits)
