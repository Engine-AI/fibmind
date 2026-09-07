"""Metrics shared by all FibMind retrieval evaluations."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Iterable

from evals.baselines import EvaluationCase, RetrievalResult
from fibmind.tokens import estimate_tokens as _estimate_tokens


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    baseline: str
    case_id: str
    retrieved_count: int
    recall_at_k: float | None
    reciprocal_rank: float | None
    answerability: float | None
    no_relevant_accuracy: float | None
    irrelevant_injection_rate: float
    stale_pollution_rate: float
    forbidden_hit: float
    estimated_tokens: int
    latency_ms: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["latency_ms"] = round(self.latency_ms, 4)
        return data


def estimate_tokens(text: str) -> int:
    """Deterministic model-agnostic size estimate; see ``fibmind.tokens``."""
    return _estimate_tokens(text)


def evaluate_case(case: EvaluationCase, result: RetrievalResult, top_k: int) -> CaseMetrics:
    """Score one normalized retrieval result against one case."""
    retrieved = result.retrieved_ids[: case.top_k or top_k]
    relevant = set(case.relevant_ids)
    stale = set(case.stale_ids)
    forbidden = set(case.forbidden_ids)

    if relevant:
        recall = len(relevant.intersection(retrieved)) / len(relevant)
        first_rank = next(
            (index for index, memory_id in enumerate(retrieved, start=1) if memory_id in relevant),
            None,
        )
        reciprocal_rank = 1.0 / first_rank if first_rank is not None else 0.0
        no_relevant_accuracy = None
    else:
        recall = None
        reciprocal_rank = None
        no_relevant_accuracy = 1.0 if not retrieved else 0.0

    answerability = None
    if case.expected_answer_terms:
        normalized_text = _normalize(result.text)
        answerability = float(
            all(_normalize(term) in normalized_text for term in case.expected_answer_terms)
        )

    retrieved_set = set(retrieved)
    denominator = max(1, len(retrieved))
    irrelevant = sum(1 for memory_id in retrieved if memory_id not in relevant)
    return CaseMetrics(
        baseline=result.baseline,
        case_id=case.id,
        retrieved_count=len(retrieved),
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        answerability=answerability,
        no_relevant_accuracy=no_relevant_accuracy,
        irrelevant_injection_rate=irrelevant / denominator,
        stale_pollution_rate=len(retrieved_set.intersection(stale)) / denominator,
        forbidden_hit=float(bool(retrieved_set.intersection(forbidden))),
        estimated_tokens=estimate_tokens(result.text),
        latency_ms=result.latency_ms,
    )


def aggregate_metrics(metrics: Iterable[CaseMetrics]) -> dict[str, int | float | None]:
    """Aggregate case metrics into one baseline report."""
    items = list(metrics)
    if not items:
        raise ValueError("cannot aggregate an empty metric list")
    latencies = sorted(item.latency_ms for item in items)
    return {
        "case_count": len(items),
        "recall_at_k": _mean_optional(item.recall_at_k for item in items),
        "mrr": _mean_optional(item.reciprocal_rank for item in items),
        "answerability_rate": _mean_optional(item.answerability for item in items),
        "no_relevant_accuracy": _mean_optional(
            item.no_relevant_accuracy for item in items
        ),
        "irrelevant_injection_rate": _round(mean(item.irrelevant_injection_rate for item in items)),
        "stale_pollution_rate": _round(mean(item.stale_pollution_rate for item in items)),
        "forbidden_case_rate": _round(mean(item.forbidden_hit for item in items)),
        "mean_estimated_tokens": _round(mean(item.estimated_tokens for item in items), 2),
        "total_estimated_tokens": sum(item.estimated_tokens for item in items),
        "latency_p50_ms": _round(median(latencies), 4),
        "latency_p95_ms": _round(_nearest_rank(latencies, 0.95), 4),
    }


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()




def _mean_optional(values: Iterable[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return _round(mean(available)) if available else None


def _nearest_rank(values: list[float], quantile: float) -> float:
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)
