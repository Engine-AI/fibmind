"""Outcome-driven tuning with an evaluation gate.

The brain is allowed to change its own behaviour in exactly one way: propose
a bounded change to the ranking weights, prove on the fixed evaluation suite
that nothing regressed, and only then adopt it. Every proposal — accepted or
rejected — is written to the log as an ``observe`` event, so the history of
what the brain tried is as durable as the memories themselves.

Proposals are deterministic functions of the evidence in the store:

- Which signals did *confirmed* memories carry, compared with *refuted* ones?
  If confirmed memories were markedly more often confidence-backed, raise the
  confidence weight a step; if refuted ones were newer, lower recency; and so
  on. Steps are small and bounded by ``RankingWeights.BOUNDS``.
- Which admission categories kept getting rejected or retired? Those are
  reported so the caller can tighten review, but admission thresholds are not
  auto-changed here: a false skip loses information silently, a false write
  is visible and reversible.

The gate is the P0 suite (``evals.runner``): Recall@K, stale pollution, and
forbidden hits must not get worse, and MRR must not drop by more than the
tolerance. A candidate that fails is recorded and discarded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from fibmind.graph import FibMind
from fibmind.models import EventOp, MemoryEvent, MemoryStatus, Verdict, new_id, utc_now
from fibmind.retrieval import RankingWeights

STEP = 0.05
MIN_EVIDENCE = 6  # fewer judged memories than this and we do not touch anything
MRR_TOLERANCE = 0.02
TUNING_SOURCE = "tuning"
TUNING_CATEGORY = "tuning"
TUNING_TAG = "fibbrain-tuning"


@dataclass(frozen=True, slots=True)
class EvidenceProfile:
    """Signal averages over confirmed vs refuted memories."""

    confirmed: int
    refuted: int
    confirmed_conf: float
    refuted_conf: float
    confirmed_recent: float
    refuted_recent: float
    confirmed_title_match: float
    refuted_title_match: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": self.confirmed,
            "refuted": self.refuted,
            "mean_confidence": {"confirmed": round(self.confirmed_conf, 4), "refuted": round(self.refuted_conf, 4)},
            "share_recent": {"confirmed": round(self.confirmed_recent, 4), "refuted": round(self.refuted_recent, 4)},
            "share_title_match": {"confirmed": round(self.confirmed_title_match, 4), "refuted": round(self.refuted_title_match, 4)},
        }


@dataclass(frozen=True, slots=True)
class Proposal:
    weights: RankingWeights
    changes: dict[str, tuple[float, float]]
    rationale: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "changes": {k: {"from": v[0], "to": v[1]} for k, v in self.changes.items()},
            "rationale": list(self.rationale),
        }


@dataclass(slots=True)
class GateResult:
    passed: bool
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons), "baseline": self.baseline, "candidate": self.candidate}


# ---------------------------------------------------------------- evidence


def profile_evidence(memory: FibMind, recent_days: int = 30) -> EvidenceProfile:
    """Compare the signals carried by confirmed and refuted memories.

    A memory counts as *confirmed* if its log has at least one confirmed
    outcome and no refutation; *refuted* if it has a refutation. Propagated
    knowledge invalidations are not counted — they are consequences, not
    judgements about that node.
    """
    verdicts: dict[str, set[str]] = {}
    for event in memory.events:
        if event.op != EventOp.OBSERVE:
            continue
        verdict = event.payload.get("verdict")
        if verdict not in (Verdict.CONFIRMED.value, Verdict.REFUTED.value):
            continue
        verdicts.setdefault(str(event.payload.get("node_id")), set()).add(str(verdict))

    horizon = utc_now() - timedelta(days=recent_days)
    confirmed: list[Any] = []
    refuted: list[Any] = []
    for node_id, seen in verdicts.items():
        node = memory.nodes.get(node_id)
        if node is None:
            continue
        (refuted if Verdict.REFUTED.value in seen else confirmed).append(node)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def title_match(node: Any) -> float:
        # Does the title carry the content's vocabulary? A proxy for "the
        # title is informative", which is what the title weight rewards.
        from fibmind.ranking import tokenize

        title = set(tokenize(node.title))
        body = set(tokenize(node.content))
        return 1.0 if title and body and len(title & body) / len(title) >= 0.5 else 0.0

    return EvidenceProfile(
        confirmed=len(confirmed),
        refuted=len(refuted),
        confirmed_conf=mean([node.confidence for node in confirmed]),
        refuted_conf=mean([node.confidence for node in refuted]),
        confirmed_recent=mean([1.0 if node.created_at >= horizon else 0.0 for node in confirmed]),
        refuted_recent=mean([1.0 if node.created_at >= horizon else 0.0 for node in refuted]),
        confirmed_title_match=mean([title_match(node) for node in confirmed]),
        refuted_title_match=mean([title_match(node) for node in refuted]),
    )


def propose(current: RankingWeights, profile: EvidenceProfile) -> Proposal | None:
    """One bounded step per signal, in the direction the evidence points.

    Returns ``None`` when there is not enough evidence or nothing would move.
    """
    if profile.confirmed + profile.refuted < MIN_EVIDENCE or not profile.refuted or not profile.confirmed:
        return None
    changes: dict[str, tuple[float, float]] = {}
    rationale: list[str] = []
    values = {"title": current.title, "confidence": current.confidence, "recency": current.recency}

    def nudge(name: str, direction: int, why: str) -> None:
        low, high = RankingWeights.BOUNDS[name]
        before = values[name]
        after = round(min(high, max(low, before + direction * STEP)), 4)
        if after != before:
            values[name] = after
            changes[name] = (before, after)
            rationale.append(why)

    # Confirmed memories carry more confidence than refuted ones → trust it more.
    # (This is the expected shape; the check guards against a store where
    # confidence was being earned by things that later failed.)
    gap = profile.confirmed_conf - profile.refuted_conf
    if gap > 0.1:
        nudge("confidence", +1, f"confirmed memories carry {gap:.2f} more confidence than refuted ones")
    elif gap < -0.1:
        nudge("confidence", -1, f"refuted memories carried {-gap:.2f} more confidence than confirmed ones")

    gap = profile.confirmed_recent - profile.refuted_recent
    if gap < -0.2:
        nudge("recency", -1, f"refuted memories were newer ({profile.refuted_recent:.0%} recent vs {profile.confirmed_recent:.0%})")
    elif gap > 0.2:
        nudge("recency", +1, f"confirmed memories were newer ({profile.confirmed_recent:.0%} recent vs {profile.refuted_recent:.0%})")

    gap = profile.confirmed_title_match - profile.refuted_title_match
    if gap > 0.2:
        nudge("title", +1, "confirmed memories had informative titles more often")
    elif gap < -0.2:
        nudge("title", -1, "refuted memories had informative titles more often")

    if not changes:
        return None
    return Proposal(
        weights=RankingWeights(
            title=values["title"],
            confidence=values["confidence"],
            recency=values["recency"],
            coverage_part=current.coverage_part,
            bm25_part=current.bm25_part,
        ).validated(),
        changes=changes,
        rationale=rationale,
    )


# ---------------------------------------------------------------- gate


def gate(
    baseline_weights: RankingWeights,
    candidate_weights: RankingWeights,
    *,
    evaluate: Callable[[RankingWeights], dict[str, Any]],
    mrr_tolerance: float = MRR_TOLERANCE,
) -> GateResult:
    """Run the suite twice and compare. ``evaluate`` returns one baseline's summary."""
    before = evaluate(baseline_weights)
    after = evaluate(candidate_weights)
    reasons: list[str] = []

    def worse(key: str, higher_is_better: bool, tolerance: float = 0.0) -> None:
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            return
        if higher_is_better and a < b - tolerance:
            reasons.append(f"{key} fell {b:.4f} -> {a:.4f}")
        if not higher_is_better and a > b + tolerance:
            reasons.append(f"{key} rose {b:.4f} -> {a:.4f}")

    worse("recall_at_k", True)
    worse("mrr", True, mrr_tolerance)
    worse("stale_pollution_rate", False)
    worse("forbidden_case_rate", False)
    worse("no_relevant_accuracy", True)
    return GateResult(passed=not reasons, baseline=before, candidate=after, reasons=reasons)


def evaluate_with_weights(weights: RankingWeights, dataset_dir: str | Path | None = None) -> dict[str, Any]:
    """The P0 suite's ``fibmind_current`` summary under ``weights``."""
    from evals import baselines as eval_baselines
    from evals.runner import DEFAULT_DATASET_DIR, run_evaluation

    original = eval_baselines.FibMindCurrentBaseline.__init__

    def patched(self, dataset, max_context_chars=5200):  # type: ignore[no-untyped-def]
        original(self, dataset, max_context_chars=max_context_chars)
        self.memory.weights = weights

    eval_baselines.FibMindCurrentBaseline.__init__ = patched  # type: ignore[method-assign]
    try:
        report = run_evaluation(dataset_dir or DEFAULT_DATASET_DIR, baseline_names=["fibmind_current"])
    finally:
        eval_baselines.FibMindCurrentBaseline.__init__ = original  # type: ignore[method-assign]
    return report["baselines"]["fibmind_current"]


# ---------------------------------------------------------------- record


def record_tuning(memory: FibMind, outcome: dict[str, Any]) -> MemoryEvent:
    """Append the tuning attempt to the log so the history survives replay."""
    return memory._record(  # noqa: SLF001 — tuning is part of the store's own history
        EventOp.OBSERVE,
        {
            "node_id": None,
            "verdict": "tuning",
            "source": TUNING_SOURCE,
            "confidence": 0.0,
            "tuning": outcome,
            "at": utc_now().isoformat(),
        },
    )


def tuning_history(memory: FibMind) -> list[dict[str, Any]]:
    return [
        {"seq": event.seq, "at": event.payload.get("at"), **(event.payload.get("tuning") or {})}
        for event in memory.events
        if event.op == EventOp.OBSERVE and event.payload.get("verdict") == "tuning"
    ]


def admission_pressure(memory: FibMind) -> dict[str, dict[str, int]]:
    """Per category: how many review-written memories were rejected or retired.

    Reported, not acted on. A category where most automatic writes get
    rejected is a signal to tighten its review or extraction — by a human.
    """
    out: dict[str, dict[str, int]] = {}
    for node in memory.nodes.values():
        if "fibbrain-review" not in node.tags:
            continue
        row = out.setdefault(node.category, {"written": 0, "rejected": 0, "pending": 0, "active": 0})
        row["written"] += 1
        if node.status == MemoryStatus.STALE and (node.status_reason or "").startswith("rejected"):
            row["rejected"] += 1
        elif node.status == MemoryStatus.PENDING:
            row["pending"] += 1
        elif node.status == MemoryStatus.ACTIVE:
            row["active"] += 1
    return out
