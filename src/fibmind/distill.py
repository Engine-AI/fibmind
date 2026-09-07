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

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

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
