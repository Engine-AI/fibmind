"""Pluggable distillation: how a fold is summarized and how a goal is planned.

Both default implementations are deterministic and are exactly what the store
did before this seam existed. An LLM-backed implementation slots in behind
the same protocol and must record provenance — which model, which prompt
version, which inputs — so a summary can be regenerated from the log when a
better distiller arrives. The store keeps folded sources intact for that.

Nothing here changes the promotion bar: a distilled claim still needs
``promote_to_knowledge`` and its three independent supporters.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from fibmind.models import MemoryNode
from fibmind.planning import BrainGoal, GoalStatus, synthesize_plan

SUMMARIZER_KEY = "summarizer"


@dataclass(frozen=True, slots=True)
class Summary:
    text: str
    provenance: dict[str, Any] = field(default_factory=dict)


class Summarizer(Protocol):
    """Fold several nodes into one text at (ideally) a higher abstraction."""

    name: str

    def summarize(self, nodes: list[MemoryNode]) -> Summary: ...


class Planner(Protocol):
    """Turn an objective plus recalled evidence into an ordered goal."""

    name: str

    def plan(
        self,
        objective: str,
        memories: Iterable[dict[str, Any]],
        status: GoalStatus = GoalStatus.ACTIVE,
        goal_id: str = "",
    ) -> BrainGoal: ...


@dataclass(frozen=True, slots=True)
class ExcerptSummarizer:
    """Concatenate truncated excerpts. Shorter, not more abstract."""

    name: str = "excerpt-v1"
    excerpt_chars: int = 120

    def summarize(self, nodes: list[MemoryNode]) -> Summary:
        lines = []
        for node in nodes:
            excerpt = node.content.strip().replace("\n", " ")
            if len(excerpt) > self.excerpt_chars:
                excerpt = f"{excerpt[: self.excerpt_chars - 3]}..."
            lines.append(f"- {node.title}: {excerpt}")
        return Summary(
            text="\n".join(lines),
            provenance={"summarizer": self.name, "source_node_ids": [node.id for node in nodes]},
        )


@dataclass(frozen=True, slots=True)
class StructuredDistillation:
    title: str
    directive: str
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    verified_by: tuple[str, ...] = ()
    risk_level: str = "low"

    def to_text(self) -> str:
        parts = [f"- [RULE] {self.title}: {self.directive}"]
        if self.conditions:
            parts.append(f"  • Conditions: {'; '.join(self.conditions)}")
        if self.exceptions:
            parts.append(f"  • Exceptions: {'; '.join(self.exceptions)}")
        if self.verified_by:
            parts.append(f"  • VerifiedBy: {'; '.join(self.verified_by)}")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class StructuredSummarizer:
    """Extract directives, conditions, exceptions, and evidence without lossy character truncation."""

    name: str = "structured-v1"

    def distill_node(self, node: MemoryNode) -> StructuredDistillation:
        content = node.content.strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        # 1. Directive: first substantive sentence or content
        directive = lines[0] if lines else node.title
        for line in lines:
            lower = line.lower()
            if not any(
                lower.startswith(p) for p in ("when ", "if ", "note:", "warning:", "beware", "avoid ", "precondition:")
            ):
                directive = line
                break

        # 2. Conditions from metadata or lines
        conditions: list[str] = []
        meta_conds = node.metadata.get("conditions") or node.metadata.get("preconditions")
        if isinstance(meta_conds, (list, tuple)):
            conditions.extend(str(c) for c in meta_conds)
        for line in lines:
            lower = line.lower()
            if any(lower.startswith(prefix) for prefix in ("when ", "if ", "requires ", "under ", "precondition:")):
                conditions.append(line)

        # 3. Exceptions and Traps from metadata or lines
        exceptions: list[str] = []
        meta_ex = node.metadata.get("exceptions") or node.metadata.get("risks") or node.metadata.get("pitfalls")
        if isinstance(meta_ex, (list, tuple)):
            exceptions.extend(str(e) for e in meta_ex)
        for line in lines:
            lower = line.lower()
            if any(
                lower.startswith(prefix)
                for prefix in ("avoid ", "beware ", "do not ", "don't ", "except ", "warning:", "pitfall:", "unless ")
            ):
                exceptions.append(line)

        # 4. Verified By
        verified_by: list[str] = []
        meta_files = node.metadata.get("files") or node.metadata.get("changed_files")
        if isinstance(meta_files, (list, tuple)) and meta_files:
            verified_by.append(f"files: {', '.join(str(f) for f in meta_files)}")
        if getattr(node, "confidence", 0.0) > 0:
            verified_by.append(f"confidence: {node.confidence:.2f}")
        meta_source = node.metadata.get("source") or node.metadata.get("verification")
        if meta_source:
            verified_by.append(f"source: {meta_source}")

        risk = "high" if exceptions else ("medium" if conditions else "low")

        return StructuredDistillation(
            title=node.title,
            directive=directive,
            conditions=tuple(conditions),
            exceptions=tuple(exceptions),
            verified_by=tuple(verified_by),
            risk_level=risk,
        )

    def summarize(self, nodes: list[MemoryNode]) -> Summary:
        distillations = [self.distill_node(node) for node in nodes]
        rendered = "\n".join(d.to_text() for d in distillations)
        return Summary(
            text=rendered,
            provenance={
                "summarizer": self.name,
                "source_node_ids": [node.id for node in nodes],
                "count": len(nodes),
            },
        )


@dataclass(frozen=True, slots=True)
class DeterministicPlanner:
    """The template planner: recall → avoid/reuse → work → verify → reflect."""

    name: str = "template-v1"

    def plan(
        self,
        objective: str,
        memories: Iterable[dict[str, Any]],
        status: GoalStatus = GoalStatus.ACTIVE,
        goal_id: str = "",
    ) -> BrainGoal:
        return synthesize_plan(objective, memories=memories, status=status, goal_id=goal_id)


@dataclass
class CallableSummarizer:
    """Wrap any ``(nodes) -> str`` — an LLM call, for instance.

    ``model`` and ``prompt_version`` are recorded on every summary so the
    output can be reproduced or invalidated later. A failure falls back to
    the excerpt summarizer instead of blocking the fold.
    """

    fn: Any
    model: str
    prompt_version: str
    name: str = "callable"
    fallback: ExcerptSummarizer = field(default_factory=ExcerptSummarizer)

    def summarize(self, nodes: list[MemoryNode]) -> Summary:
        try:
            text = str(self.fn(nodes)).strip()
        except Exception as exc:  # noqa: BLE001 — a distiller failure must not lose the fold
            fallback = self.fallback.summarize(nodes)
            fallback.provenance["fallback_from"] = f"{self.name}: {type(exc).__name__}"
            return fallback
        if not text:
            return self.fallback.summarize(nodes)
        return Summary(
            text=text,
            provenance={
                "summarizer": self.name,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "source_node_ids": [node.id for node in nodes],
            },
        )


DEFAULT_SUMMARIZER: Summarizer = ExcerptSummarizer()
DEFAULT_PLANNER: Planner = DeterministicPlanner()
