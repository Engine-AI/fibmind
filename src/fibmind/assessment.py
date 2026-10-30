"""Metacognition for FibBrain: knowing what it does not know.

Separates relevance from trust and classifies goal evidence into four states:
- ``known``: relevant, active, trustworthy, precondition-checked, fresh.
- ``uncertain``: relevant, active, but lacking proof or environment integrity.
- ``contradicted``: relevant but refuted.
- ``unknown``: aspects of the goal uncovered by usable evidence.

Provides minimum-cost routing for gaps (avoid -> verify -> code -> resolve -> ask -> search -> model).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fibmind.distill import Aspect, GapAssessor
from fibmind.models import MemoryKind, MemoryNode, MemoryStatus, utc_now
from fibmind.ranking import STOP_WORDS, _is_cjk, normalize_cjk, tokenize
from fibmind.reflex import check_node_preconditions
from fibmind.retrieval import is_lexical_shortcut_query

if TYPE_CHECKING:
    from fibmind.embedding import EmbeddingProvider
    from fibmind.retrieval import LexicalIndex

# Calibration threshold constants
RELEVANCE_FLOOR: float = 0.12
TRUST_FLOOR: float = 0.5
KNOWN_COVERAGE: float = 0.6
FRESHNESS_DAYS: int = 90


@dataclass(frozen=True, slots=True)
class DeterministicAssessor:
    """Extract unknown goal aspects by grouping uncovered terms into surface units.

    English/identifiers report as themselves; CJK runs report as whole runs when
    more than half of their bigrams are uncovered. Sorted by descending IDF, max 5.
    """

    name: str = "deterministic-v1"
    idf_fn: Callable[[str], float] | None = None

    def aspects(self, goal: str, query_terms: set[str]) -> list[Aspect]:
        if not goal or not query_terms:
            return []

        aspects_map: dict[str, set[str]] = {}

        # 1. Whitespace tokens check (paths, code symbols, hyphenated/underscored identifiers)
        for token in goal.split():
            clean = token.strip(".,;:?!'\"()[]{}")
            if not clean:
                continue
            has_cjk_char = any(_is_cjk(ch) for ch in clean)
            if not has_cjk_char:
                token_terms = [t for t in tokenize(clean) if t not in STOP_WORDS]
                uncovered = [t for t in token_terms if t in query_terms]
                if uncovered:
                    aspects_map.setdefault(clean, set()).update(uncovered)

        # 2. CJK character runs check
        normalized = unicodedata.normalize("NFKC", goal).casefold()
        normalized = normalize_cjk(normalized)
        current_cjk: list[str] = []
        cjk_runs: list[str] = []
        for ch in normalized:
            if _is_cjk(ch):
                current_cjk.append(ch)
            else:
                if current_cjk:
                    cjk_runs.append("".join(current_cjk))
                    current_cjk = []
        if current_cjk:
            cjk_runs.append("".join(current_cjk))

        for run in cjk_runs:
            if len(run) == 1:
                if run in query_terms:
                    aspects_map.setdefault(run, set()).add(run)
            else:
                bigrams = [run[i : i + 2] for i in range(len(run) - 1)]
                uncovered = [b for b in bigrams if b in query_terms]
                if len(uncovered) / len(bigrams) > 0.5:
                    aspects_map.setdefault(run, set()).update([run, *uncovered])

        # 3. Fallback for any standalone query terms not captured in surface units
        for term in query_terms:
            if not any(term in terms for terms in aspects_map.values()) and term not in aspects_map:
                aspects_map[term] = {term}

        # Convert to Aspect instances and rank by maximum IDF
        results: list[tuple[float, Aspect]] = []
        for aspect_str, terms_set in aspects_map.items():
            aspect_terms = tuple(sorted(terms_set))
            score = max((self.idf_fn(t) if self.idf_fn else 1.0) for t in aspect_terms)
            results.append((score, Aspect(aspect=aspect_str, terms=aspect_terms)))

        results.sort(key=lambda item: (item[0], item[1].aspect), reverse=True)
        return [item[1] for item in results[:5]]

    def classify(
        self,
        goal: str,
        candidates: Iterable[MemoryNode],
        refuted_candidates: Iterable[MemoryNode] = (),
        *,
        trust_floor: float | None = None,
        p_refuted_given_known: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return classify(
            goal,
            candidates,
            refuted_candidates,
            assessor=self,
            trust_floor=trust_floor,
            p_refuted_given_known=p_refuted_given_known,
            **kwargs,
        )


def route_uncertain(node: MemoryNode, reasons: list[str]) -> tuple[str, str]:
    """Route an uncertain memory to its minimum-cost verification target."""
    meta = node.metadata or {}

    # 1. Has recorded command -> verify
    cmd = meta.get("command") or meta.get("verify")
    if not cmd and isinstance(meta.get("commands"), list) and meta.get("commands"):
        cmd = meta["commands"][0]
    if cmd:
        return "verify", str(cmd)

    # 2. File hash drift, missing file, or referenced files -> code
    for r in reasons:
        for prefix in (
            "file_hash_mismatch: ",
            "missing_hashed_file: ",
            "missing_target_file: ",
            "symbol_hash_mismatch: ",
            "missing_symbol: ",
        ):
            if r.startswith(prefix):
                return "code", r[len(prefix) :].strip()

    files = meta.get("files") or meta.get("changed_files")
    if isinstance(files, (list, tuple)) and files:
        return "code", str(files[0])

    if isinstance(meta.get("symbol_hashes"), dict) and meta["symbol_hashes"]:
        return "code", str(next(iter(meta["symbol_hashes"].keys())))

    # 3. Dialectical conflict -> resolve
    if "conflicted" in reasons or meta.get("conflicted") or "upstream_refuted" in reasons:
        return "resolve", node.id

    # 4. Preference or requirement -> ask user
    kind_val = node.memory_kind.value if node.memory_kind else ""
    if kind_val in (MemoryKind.PREFERENCE.value, MemoryKind.REQUIREMENT.value) or node.category in (
        "preference",
        "profile",
        "requirement",
    ):
        return "ask", "user"

    # 5. Default -> caution
    return "caution", node.id


def route_unknown(
    goal: str,
    aspect: Aspect,
    open_questions: Iterable[MemoryNode] = (),
) -> tuple[str, str, str]:
    """Route an unknown aspect to the most suitable source of truth.

    Returns ``(route, target, why)``.
    """
    text = aspect.aspect

    # 1. Path or code symbol
    if (
        is_lexical_shortcut_query(text)
        or any(sep in text for sep in ("/", "\\", "::"))
        or ("." in text and not text.endswith("."))
        or ("_" in text and text.isascii())
    ):
        return "code", text, f"no relevant memory; goal names {text}"

    # 2. Matches unresolved open question
    for q in open_questions:
        q_terms = set(tokenize(q.title) + tokenize(q.content))
        if set(aspect.terms) & q_terms:
            return "ask", "user", f"matches open question: {q.title}"

    # 3. Preference or convention intent
    goal_lower = goal.casefold()
    if any(w in goal_lower for w in ("should", "prefer", "convention", "应该", "偏好", "规范")):
        return "ask", "user", "goal asks about convention or preference"

    # 4. Time-sensitive intent
    time_words = ("latest", "version", "release", "deprecated", "最新", "版本")
    if any(w in goal_lower for w in time_words) or bool(re.search(r"\b20\d\d\b", goal)):
        return "search", "web search", "goal is time-sensitive"

    # 5. Fallback to model's own reasoning
    return "model", "model", "no relevant memory"


def calibrate_trust_floor(p_refuted_given_known: float = 0.0, base_floor: float = TRUST_FLOOR) -> float:
    """Calibrate the metacognitive trust floor based on the empirical false-known rate."""
    if p_refuted_given_known <= 0.0:
        return float(base_floor)
    return min(0.75, round(base_floor + min(0.25, p_refuted_given_known * 0.5), 4))


def classify(
    goal: str,
    candidates: Iterable[MemoryNode],
    refuted_candidates: Iterable[MemoryNode] = (),
    *,
    lexical: LexicalIndex | None = None,
    outcome_lookup: Callable[[str], dict[str, Any]] | None = None,
    last_confirmation_lookup: Callable[[str], datetime | None] | None = None,
    cwd: str | Path | None = None,
    model_id: str | None = None,
    open_questions: Iterable[MemoryNode] = (),
    vector_provider: EmbeddingProvider | None = None,
    vector_scores: dict[str, float] | None = None,
    assessor: GapAssessor | None = None,
    now: datetime | None = None,
    state: dict[str, Any] | None = None,
    trust_floor: float | None = None,
    p_refuted_given_known: float = 0.0,
) -> dict[str, Any]:
    """Classify memories against ``goal`` into known, uncertain, contradicted, unknown, and weak."""
    effective_trust_floor = (
        float(trust_floor) if trust_floor is not None else calibrate_trust_floor(p_refuted_given_known)
    )
    current_time = now or utc_now()

    def idf_val(term: str) -> float:
        if lexical is not None:
            return lexical.idf(term)
        return 1.0

    goal_terms = tokenize(goal)
    goal_terms_set = set(goal_terms)
    total_goal_idf = sum(idf_val(t) for t in goal_terms_set)

    known: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    contradicted: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []

    seen_node_ids: set[str] = set()

    # Combine candidates with refuted hits
    all_nodes: list[MemoryNode] = list(candidates)
    for rnode in refuted_candidates:
        if rnode.id not in {n.id for n in all_nodes}:
            all_nodes.append(rnode)

    for node in all_nodes:
        if node.id in seen_node_ids:
            continue
        seen_node_ids.add(node.id)

        node_terms = set(
            tokenize(node.title)
            + tokenize(node.content)
            + tokenize(node.category)
            + [t for tag in node.tags for t in tokenize(tag)]
        )
        matched = tuple(sorted(node_terms & goal_terms_set))
        matched_idf = sum(idf_val(t) for t in matched)
        wcov = matched_idf / total_goal_idf if total_goal_idf > 0 else 0.0

        vec_score = vector_scores.get(node.id) if vector_scores else None
        has_semantic_vector = (
            vector_provider is not None
            and vector_provider.name != "hashing"
            and vec_score is not None
            and vec_score >= vector_provider.min_similarity
        )
        is_relevant = (wcov >= RELEVANCE_FLOOR) or has_semantic_vector

        relevance = round(max(wcov, vec_score or 0.0), 4)

        if not is_relevant:
            weak.append(
                {
                    "node_id": node.id,
                    "title": node.title,
                    "relevance": relevance,
                    "matched_terms": list(matched),
                }
            )
            continue

        if node.status == MemoryStatus.REFUTED:
            contradicted.append(
                {
                    "node_id": node.id,
                    "title": node.title,
                    "relevance": relevance,
                    "matched_terms": list(matched),
                    "route": "avoid",
                    "reason": node.status_reason or "refuted",
                }
            )
            continue

        if node.status != MemoryStatus.ACTIVE:
            continue

        # Environmental preconditions check
        precond_ok, precond_reason = check_node_preconditions(node, cwd=cwd, model_id=model_id)

        outcomes = outcome_lookup(node.id) if outcome_lookup else {"confirmed": 0, "refuted": 0, "sessions": 0}
        last_conf = last_confirmation_lookup(node.id) if last_confirmation_lookup else None
        last_time = last_conf or (node.created_at if outcomes.get("confirmed", 0) > 0 else None)

        reasons: list[str] = []
        if node.confidence == 0.0:
            reasons.append("unchecked")
        elif node.confidence < effective_trust_floor:
            reasons.append("low_confidence")

        # Confirmation freshness
        if outcomes.get("confirmed", 0) > 0 and last_time is not None:
            if (current_time - last_time).days > FRESHNESS_DAYS:
                reasons.append("confirmation_stale")
        elif node.confidence >= effective_trust_floor:
            ref_date = last_time or node.created_at
            if (current_time - ref_date).days > FRESHNESS_DAYS:
                reasons.append("confirmation_stale")

        if node.metadata.get("conflicted"):
            reasons.append("conflicted")

        if node.metadata.get("upstream_refuted") and "upstream_refuted" not in reasons:
            reasons.append("upstream_refuted")

        if not precond_ok:
            if precond_reason.startswith("conflicted_memory"):
                if "conflicted" not in reasons:
                    reasons.append("conflicted")
            elif precond_reason.startswith("model_heuristic_mismatch"):
                reasons.append("model_heuristic_mismatch")
            else:
                reasons.append(precond_reason)

        if not reasons:
            confirmed_count = outcomes.get("confirmed", 0)
            sessions_count = outcomes.get("sessions", 0)
            last_date_str = (last_time or node.created_at).strftime("%Y-%m-%d")
            src = (
                node.confidence_source
                or node.metadata.get("source")
                or node.metadata.get("verification")
                or node.metadata.get("command")
            )
            ev = f"confirmed {confirmed_count}x in {sessions_count} sessions"
            if src:
                ev += f" via {src}"
            ev += f"; last {last_date_str}"
            known.append(
                {
                    "node_id": node.id,
                    "title": node.title,
                    "confidence": round(node.confidence, 4),
                    "relevance": relevance,
                    "matched_terms": list(matched),
                    "evidence": ev,
                }
            )
        else:
            route, target = route_uncertain(node, reasons)
            uncertain.append(
                {
                    "node_id": node.id,
                    "title": node.title,
                    "confidence": round(node.confidence, 4),
                    "relevance": relevance,
                    "matched_terms": list(matched),
                    "reasons": reasons,
                    "route": route,
                    "target": target,
                }
            )

    # Covered terms from relevant hits
    covered_terms: set[str] = set()
    for item in known:
        covered_terms.update(item["matched_terms"])
    for item in uncertain:
        covered_terms.update(item["matched_terms"])
    for item in contradicted:
        covered_terms.update(item["matched_terms"])

    uncovered_terms = goal_terms_set - covered_terms

    # Unknown aspects decomposition
    assessor_impl = assessor or DeterministicAssessor(idf_fn=idf_val)
    raw_aspects = assessor_impl.aspects(goal, uncovered_terms)
    unknown: list[dict[str, Any]] = []
    for aspect in raw_aspects:
        route, target, why = route_unknown(goal, aspect, open_questions)
        unknown.append(
            {
                "aspect": aspect.aspect,
                "terms": list(aspect.terms),
                "route": route,
                "target": target,
                "why": why,
            }
        )

    # Coverage calculations
    known_terms = set()
    for k in known:
        known_terms.update(k["matched_terms"])
    known_idf = sum(idf_val(t) for t in (known_terms & goal_terms_set))
    known_cov = known_idf / total_goal_idf if total_goal_idf > 0 else 0.0

    any_terms = covered_terms & goal_terms_set
    any_idf = sum(idf_val(t) for t in any_terms)
    any_cov = any_idf / total_goal_idf if total_goal_idf > 0 else 0.0

    coverage = {
        "known": round(known_cov, 2),
        "any": round(any_cov, 2),
    }

    # Verdict determination
    if any_cov == 0.0:
        verdict = "unknown"
    elif known_cov >= KNOWN_COVERAGE:
        has_conflict = False
        for c in contradicted:
            if set(c["matched_terms"]) & known_terms:
                has_conflict = True
                break
        if not has_conflict:
            for u in uncertain:
                if "conflicted" in u["reasons"] and (set(u["matched_terms"]) & known_terms):
                    has_conflict = True
                    break
        verdict = "known" if not has_conflict else "partial"
    else:
        verdict = "partial"

    # Assemble deduplicated next actions
    next_steps: list[dict[str, str]] = []
    seen_next: set[tuple[str, str]] = set()

    for c in contradicted:
        pair = ("avoid", c["node_id"])
        if pair not in seen_next:
            seen_next.add(pair)
            next_steps.append({"route": "avoid", "target": c["node_id"]})

    for u in uncertain:
        if u["route"] != "caution":
            pair = (u["route"], u["target"])
            if pair not in seen_next:
                seen_next.add(pair)
                next_steps.append({"route": u["route"], "target": u["target"]})

    for uk in unknown:
        target = uk.get("target") or uk["aspect"]
        pair = (uk["route"], target)
        if pair not in seen_next:
            seen_next.add(pair)
            next_steps.append({"route": uk["route"], "target": target})

    return {
        "goal": goal,
        "verdict": verdict,
        "coverage": coverage,
        "trust_floor": effective_trust_floor,
        "known": known,
        "uncertain": uncertain,
        "contradicted": contradicted,
        "unknown": unknown,
        "weak": weak,
        "next": next_steps,
        "assessor": assessor_impl.name,
        "thresholds": {
            "relevance_floor": RELEVANCE_FLOOR,
            "trust_floor": effective_trust_floor,
            "known_coverage": KNOWN_COVERAGE,
            "freshness_days": FRESHNESS_DAYS,
        },
        "state": dict(state or {}),
    }
