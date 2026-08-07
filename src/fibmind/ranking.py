"""Keyword-based ranking for query search over memory nodes.

This module is intentionally dependency-free and pure: it never mutates the
nodes it scores. Semantic/embedding ranking can later replace ``rank_nodes``
behind the same ``ScoredHit`` interface.

Note what is deliberately absent from the formula: recall frequency. Scoring
familiarity closes a loop — recalled nodes rank higher, ranking higher gets them
recalled again — which promotes whatever is familiar rather than whatever is
correct. Only ``confidence``, which moves solely on external evidence, is
trusted here.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable

from fibmind.models import MemoryNode, NodeType, utc_now

# Scoring weights (kept as named constants — no magic numbers in the formula).
WEIGHT_COVERAGE = 1.0  # fraction of query terms found anywhere in the node
WEIGHT_TITLE = 0.6  # fraction of query terms found in the title
WEIGHT_CONFIDENCE = 0.3  # externally evidenced correctness
WEIGHT_RECENCY = 0.2  # how recently the node was updated

# A node updated within this window counts as fully "recent"; older nodes decay
# linearly to zero across it.
RECENCY_HORIZON = timedelta(days=30)

# Function words create false positives in small keyword corpora: a query about
# Kubernetes used to recall unrelated memories solely because both contained
# "the" or "is". Keep the list intentionally small and English-only; CJK runs
# retain their full text and overlapping bigrams.
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text, including overlapping CJK bigrams."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    runs: list[tuple[str, str]] = []
    current: list[str] = []
    current_kind: str | None = None

    for character in normalized:
        kind = "cjk" if _is_cjk(character) else "word" if character.isalnum() else None
        if kind is None:
            if current:
                runs.append((current_kind or "word", "".join(current)))
                current = []
                current_kind = None
            continue
        if current and kind != current_kind:
            runs.append((current_kind or "word", "".join(current)))
            current = []
        current.append(character)
        current_kind = kind

    if current:
        runs.append((current_kind or "word", "".join(current)))

    tokens: list[str] = []
    seen: set[str] = set()
    for kind, run in runs:
        if kind == "word" and run in STOP_WORDS:
            continue
        candidates = [run]
        if kind == "cjk" and len(run) > 1:
            candidates.extend(run[index : index + 2] for index in range(len(run) - 1))
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                tokens.append(candidate)
    return tokens


@dataclass(frozen=True, slots=True)
class ScoredHit:
    """A node together with its relevance score for a query."""

    node: MemoryNode
    score: float
    matched_terms: tuple[str, ...] = field(default_factory=tuple)


def _recency_factor(node: MemoryNode) -> float:
    age = utc_now() - node.updated_at
    if age <= timedelta(0):
        return 1.0
    if age >= RECENCY_HORIZON:
        return 0.0
    return 1.0 - (age / RECENCY_HORIZON)


def score_node(node: MemoryNode, query_terms: set[str]) -> tuple[float, tuple[str, ...]]:
    """Score a single node against the query terms.

    Returns the score and the set of query terms that matched. A score of ``0``
    means no query term appeared in the node's searchable text.
    """
    if not query_terms:
        return 0.0, ()

    title_terms = set(tokenize(node.title))
    body_terms = title_terms.union(
        tokenize(node.content),
        tokenize(node.category),
        *(tokenize(tag) for tag in node.tags),
    )

    matched = tuple(term for term in query_terms if term in body_terms)
    if not matched:
        return 0.0, ()

    coverage = len(matched) / len(query_terms)
    title_matches = sum(1 for term in query_terms if term in title_terms)
    title_coverage = title_matches / len(query_terms)

    score = (
        coverage * WEIGHT_COVERAGE
        + title_coverage * WEIGHT_TITLE
        + max(0.0, min(1.0, node.confidence)) * WEIGHT_CONFIDENCE
        + _recency_factor(node) * WEIGHT_RECENCY
    )
    return score, matched


def rank_nodes(
    nodes: Iterable[MemoryNode],
    query: str,
    top_k: int = 5,
    categories: set[str] | None = None,
    min_score: float = 0.0,
    include_roots: bool = False,
) -> list[ScoredHit]:
    """Rank nodes by relevance to ``query``.

    Root nodes are excluded by default because they are structural anchors
    rather than recallable content. The result is sorted by descending score
    and truncated to ``top_k``.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    hits: list[ScoredHit] = []
    for node in nodes:
        if not include_roots and node.node_type == NodeType.ROOT:
            continue
        if categories is not None and node.category not in categories:
            continue
        score, matched = score_node(node, query_terms)
        if score <= min_score:
            continue
        hits.append(ScoredHit(node=node, score=score, matched_terms=matched))

    hits.sort(key=lambda hit: (hit.score, hit.node.updated_at), reverse=True)
    return hits[:top_k]
