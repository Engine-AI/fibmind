"""Hybrid retrieval: inverted index + BM25, optional vectors, rank fusion.

The old path scored every node against the query on every call. This one
builds candidates from an inverted index (only nodes sharing a query term are
scored), ranks them with BM25 plus the same title / confidence / recency
signals as before, optionally adds a nearest-vector list, and fuses the two
with reciprocal rank fusion. Graph expansion stays where it was: a
supplementary step in ``context.build_context``, not a first-class retriever.

What is deliberately unchanged from ``ranking.py``: familiarity never enters a
score, and ``confidence`` is the only trust signal.

``explain`` returns every component per candidate, and why excluded nodes were
excluded, so a recall can always be accounted for.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Iterable

from fibmind.models import MemoryNode, MemoryScope, MemoryStatus, NodeType, utc_now
from fibmind.ranking import (
    RECENCY_HORIZON,
    WEIGHT_CONFIDENCE,
    WEIGHT_RECENCY,
    WEIGHT_TITLE,
    ScoredHit,
    tokenize,
)

if TYPE_CHECKING:
    from fibmind.embedding import EmbeddingCache
    from fibmind.graph import FibMind

@dataclass(frozen=True, slots=True)
class RankingWeights:
    """The tunable part of the score. Everything else in retrieval is fixed.

    ``familiarity`` is deliberately not a field: no tuning may reintroduce it.
    Bounds keep a tuner from zeroing relevance or letting recency dominate.
    """

    title: float = WEIGHT_TITLE
    confidence: float = WEIGHT_CONFIDENCE
    recency: float = WEIGHT_RECENCY
    coverage_part: float = 0.6
    bm25_part: float = 0.4

    BOUNDS = {
        "title": (0.0, 1.0),
        "confidence": (0.0, 1.0),
        "recency": (0.0, 0.5),
        "coverage_part": (0.3, 0.9),
    }

    def validated(self) -> "RankingWeights":
        for name, (low, high) in self.BOUNDS.items():
            value = getattr(self, name)
            if not low <= value <= high:
                raise ValueError(f"ranking weight {name}={value} outside [{low}, {high}]")
        if abs(self.coverage_part + self.bm25_part - 1.0) > 1e-6:
            raise ValueError("coverage_part and bm25_part must sum to 1")
        return self

    def to_dict(self) -> dict[str, float]:
        return {
            "title": self.title,
            "confidence": self.confidence,
            "recency": self.recency,
            "coverage_part": self.coverage_part,
            "bm25_part": self.bm25_part,
            "familiarity": 0.0,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RankingWeights":
        if not data:
            return cls()
        coverage = float(data.get("coverage_part", 0.6))
        return cls(
            title=float(data.get("title", WEIGHT_TITLE)),
            confidence=float(data.get("confidence", WEIGHT_CONFIDENCE)),
            recency=float(data.get("recency", WEIGHT_RECENCY)),
            coverage_part=coverage,
            bm25_part=float(data.get("bm25_part", round(1.0 - coverage, 6))),
        ).validated()


DEFAULT_WEIGHTS = RankingWeights()

# BM25 constants. k1 saturates term frequency; b normalizes for document length.
BM25_K1 = 1.2
BM25_B = 0.75
# The lexical split (coverage vs BM25) and the title / confidence / recency
# weights live on RankingWeights so they can be tuned per store; see tuning.py.
# Reciprocal rank fusion constant (Cormack et al.); 60 is the usual choice.
RRF_K = 60
# Vector candidates must clear the provider's own absolute floor (what counts
# as "similar" differs by provider) and stay within a ratio of the best hit.
# Without the absolute floor a query about nothing in the store still gets
# nearest-neighbours, and "no relevant memory" becomes an impossible answer.
VECTOR_RELATIVE_FLOOR = 0.5
VECTOR_TOP_N = 30
LEXICAL_TOP_N = 60


class LexicalIndex:
    """Inverted index over node text, kept in step with the forest.

    Terms come from ``tokenize`` (words plus CJK bigrams). Title terms are also
    tracked so the title bonus does not need to retokenize.
    """

    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.terms: dict[str, dict[str, int]] = {}  # node_id -> term -> tf
        self.title_terms: dict[str, set[str]] = {}
        self.lengths: dict[str, int] = {}
        self.total_length = 0

    def __len__(self) -> int:
        return len(self.terms)

    def add(self, node: MemoryNode) -> None:
        if node.id in self.terms:
            self.remove(node.id)
        counts: dict[str, int] = defaultdict(int)
        for term in tokenize(node.title):
            counts[term] += 1
        for term in tokenize(node.content):
            counts[term] += 1
        for term in tokenize(node.category):
            counts[term] += 1
        for tag in node.tags:
            for term in tokenize(tag):
                counts[term] += 1
        self.terms[node.id] = dict(counts)
        self.title_terms[node.id] = set(tokenize(node.title))
        length = sum(counts.values())
        self.lengths[node.id] = length
        self.total_length += length
        for term in counts:
            self.postings[term].add(node.id)

    def remove(self, node_id: str) -> None:
        counts = self.terms.pop(node_id, None)
        if counts is None:
            return
        for term in counts:
            bucket = self.postings.get(term)
            if bucket is not None:
                bucket.discard(node_id)
                if not bucket:
                    self.postings.pop(term, None)
        self.title_terms.pop(node_id, None)
        self.total_length -= self.lengths.pop(node_id, 0)

    def rebuild(self, nodes: Iterable[MemoryNode]) -> None:
        self.postings = defaultdict(set)
        self.terms = {}
        self.title_terms = {}
        self.lengths = {}
        self.total_length = 0
        for node in nodes:
            self.add(node)

    def candidates(self, query_terms: set[str]) -> set[str]:
        found: set[str] = set()
        for term in query_terms:
            found |= self.postings.get(term, set())
        return found

    def average_length(self) -> float:
        return self.total_length / len(self.terms) if self.terms else 0.0

    def idf(self, term: str) -> float:
        n = len(self.terms)
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def bm25(self, node_id: str, query_terms: set[str]) -> tuple[float, tuple[str, ...]]:
        counts = self.terms.get(node_id)
        if not counts:
            return 0.0, ()
        avg = self.average_length() or 1.0
        length = self.lengths.get(node_id, 0)
        score = 0.0
        matched: list[str] = []
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            matched.append(term)
            denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / avg)
            score += self.idf(term) * tf * (BM25_K1 + 1.0) / denominator
        return score, tuple(sorted(matched))


@dataclass(slots=True)
class Candidate:
    node: MemoryNode
    matched_terms: tuple[str, ...] = ()
    coverage: float = 0.0
    title_coverage: float = 0.0
    bm25: float = 0.0
    bm25_norm: float = 0.0
    lexical: float = 0.0  # combined lexical relevance in [0, ~1]
    lexical_rank: int | None = None
    vector: float | None = None  # cosine similarity
    vector_rank: int | None = None
    rrf: float = 0.0
    relevance: float = 0.0  # fused relevance in [0, 1]
    confidence: float = 0.0
    recency: float = 0.0
    score: float = 0.0
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node.id,
            "title": self.node.title,
            "category": self.node.category,
            "sources": list(self.sources),
            "matched_terms": list(self.matched_terms),
            "coverage": round(self.coverage, 4),
            "title_coverage": round(self.title_coverage, 4),
            "bm25": round(self.bm25, 4),
            "bm25_norm": round(self.bm25_norm, 4),
            "lexical": round(self.lexical, 4),
            "lexical_rank": self.lexical_rank,
            "vector": None if self.vector is None else round(self.vector, 4),
            "vector_rank": self.vector_rank,
            "rrf": round(self.rrf, 6),
            "relevance": round(self.relevance, 4),
            "confidence": round(self.confidence, 4),
            "confidence_source": self.node.confidence_source,
            "recency": round(self.recency, 4),
            "score": round(self.score, 4),
        }


def _recency_factor(node: MemoryNode) -> float:
    age = utc_now() - node.updated_at
    if age <= timedelta(0):
        return 1.0
    if age >= RECENCY_HORIZON:
        return 0.0
    return 1.0 - (age / RECENCY_HORIZON)


def retrieve(
    memory: "FibMind",
    query: str,
    *,
    top_k: int = 5,
    categories: set[str] | None = None,
    exclude_categories: set[str] | None = None,
    min_score: float = 0.0,
    include_roots: bool = False,
    scopes: set[MemoryScope] | None = None,
    owner: str | None = None,
    include_folded: bool = False,
    statuses: set[MemoryStatus] | None = None,
    workspace_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    embeddings: "EmbeddingCache | None" = None,
    explain: bool = False,
    weights: RankingWeights | None = None,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Rank visible nodes for ``query``. Returns (ranked candidates, explanation)."""
    weights = weights or getattr(memory, "weights", None) or DEFAULT_WEIGHTS
    query_terms = set(tokenize(query))
    report: dict[str, Any] = {
        "query": query,
        "query_terms": sorted(query_terms),
        "lexical_candidates": 0,
        "vector_candidates": 0,
        "vector_enabled": embeddings is not None and embeddings.enabled,
        "vector_error": None,
        "weights": weights.to_dict(),
        "excluded": [],
    }
    if not query_terms:
        return [], report

    visible_filter = dict(
        scopes=scopes,
        owner=owner,
        include_folded=include_folded,
        statuses=statuses,
        workspace_id=workspace_id,
        project_id=project_id,
        session_id=session_id,
    )

    def admissible(node: MemoryNode, why: list[str]) -> bool:
        if not include_roots and node.node_type == NodeType.ROOT:
            why.append("root")
            return False
        if categories is not None and node.category not in categories:
            why.append("category")
            return False
        if exclude_categories and node.category in exclude_categories:
            why.append("excluded_category")
            return False
        if not memory.is_visible(node, **visible_filter):
            allowed = statuses if statuses is not None else {MemoryStatus.ACTIVE}
            if node.status not in allowed:
                why.append(f"status:{node.status.value}")
            elif not include_folded and node.folded_into is not None:
                why.append("folded")
            else:
                why.append("not_visible")
            return False
        return True

    # --- lexical candidates ------------------------------------------------
    candidates: dict[str, Candidate] = {}
    for node_id in memory.lexical.candidates(query_terms):
        node = memory.nodes.get(node_id)
        if node is None:
            continue
        why: list[str] = []
        if not admissible(node, why):
            if explain:
                report["excluded"].append({"node_id": node.id, "title": node.title, "reason": why[0]})
            continue
        bm25, matched = memory.lexical.bm25(node.id, query_terms)
        if not matched:
            continue
        title_terms = memory.lexical.title_terms.get(node.id, set())
        candidate = Candidate(
            node=node,
            matched_terms=matched,
            coverage=len(matched) / len(query_terms),
            title_coverage=sum(1 for term in query_terms if term in title_terms) / len(query_terms),
            bm25=bm25,
            sources=["lexical"],
        )
        candidates[node.id] = candidate
    report["lexical_candidates"] = len(candidates)

    if candidates:
        max_bm25 = max(candidate.bm25 for candidate in candidates.values()) or 1.0
        for candidate in candidates.values():
            candidate.bm25_norm = candidate.bm25 / max_bm25
            candidate.lexical = (
                candidate.coverage * weights.coverage_part + candidate.bm25_norm * weights.bm25_part
            )
        ordered = sorted(
            candidates.values(),
            key=lambda item: (item.lexical, item.title_coverage, item.node.updated_at),
            reverse=True,
        )[:LEXICAL_TOP_N]
        for rank, candidate in enumerate(ordered, start=1):
            candidate.lexical_rank = rank

    # --- vector candidates -------------------------------------------------
    vector_ranked: list[Candidate] = []
    if embeddings is not None and embeddings.enabled:
        pool: list[MemoryNode] = []
        for node in memory.nodes.values():
            why = []
            if admissible(node, why):
                pool.append(node)
        texts = [f"{node.title}\n{node.content}" for node in pool]
        vectors = embeddings.vectors_for([query] + texts)
        query_vector = vectors[0]
        if query_vector is None:
            report["vector_error"] = embeddings.last_error
        else:
            from fibmind.embedding import cosine

            scored: list[tuple[float, MemoryNode]] = []
            for node, vector in zip(pool, vectors[1:]):
                if vector is None:
                    continue
                similarity = cosine(query_vector, vector)
                if similarity > 0.0:
                    scored.append((similarity, node))
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored:
                absolute = float(getattr(embeddings.provider, "min_similarity", 0.0))
                floor = max(absolute, scored[0][0] * VECTOR_RELATIVE_FLOOR)
                scored = [item for item in scored if item[0] >= floor]
            for rank, (similarity, node) in enumerate(scored[:VECTOR_TOP_N], start=1):
                candidate = candidates.get(node.id)
                if candidate is None:
                    candidate = Candidate(node=node)
                    candidates[node.id] = candidate
                candidate.vector = similarity
                candidate.vector_rank = rank
                candidate.sources.append("vector")
                vector_ranked.append(candidate)
            report["vector_candidates"] = len(vector_ranked)
            if embeddings.last_error and embeddings.failures:
                report["vector_error"] = embeddings.last_error

    # --- fusion --------------------------------------------------------------
    if vector_ranked:
        for candidate in candidates.values():
            rrf = 0.0
            if candidate.lexical_rank is not None:
                rrf += 1.0 / (RRF_K + candidate.lexical_rank)
            if candidate.vector_rank is not None:
                rrf += 1.0 / (RRF_K + candidate.vector_rank)
            candidate.rrf = rrf
        max_rrf = max(candidate.rrf for candidate in candidates.values()) or 1.0
        for candidate in candidates.values():
            candidate.relevance = candidate.rrf / max_rrf
    else:
        for candidate in candidates.values():
            candidate.relevance = candidate.lexical

    ranked: list[Candidate] = []
    for candidate in candidates.values():
        if candidate.lexical_rank is None and candidate.vector_rank is None:
            continue
        candidate.confidence = max(0.0, min(1.0, candidate.node.confidence))
        candidate.recency = _recency_factor(candidate.node)
        candidate.score = (
            candidate.relevance
            + candidate.title_coverage * weights.title
            + candidate.confidence * weights.confidence
            + candidate.recency * weights.recency
        )
        if candidate.score <= min_score:
            continue
        ranked.append(candidate)
    ranked.sort(key=lambda item: (item.score, item.node.updated_at), reverse=True)
    return ranked[:top_k], report


def to_scored_hits(candidates: Iterable[Candidate]) -> list[ScoredHit]:
    return [
        ScoredHit(node=candidate.node, score=candidate.score, matched_terms=candidate.matched_terms)
        for candidate in candidates
    ]
