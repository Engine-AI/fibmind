"""Gap evaluation runner: measures metacognitive accuracy on knowledge gaps.

Outputs:
- verdict_accuracy: fraction of cases where goal verdict matches expected
- false_known_rate: fraction of cases claiming "known" without sufficient evidence (red line: 0.0)
- false_unknown_rate: fraction of cases saying "unknown" when usable evidence exists
- route_accuracy: fraction of cases matching expected verification routes
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fibmind.brain import FibBrain
from fibmind.graph import FibMind
from fibmind.models import EventOp, MemoryEvent, MemoryNode, MemoryScope, MemoryStatus, NodeType
from fibmind.service import MemoryService

DEFAULT_GAPS_PATH = Path(__file__).resolve().parent / "gaps" / "knowledge_gaps.json"


def build_gaps_store(data: dict[str, Any], tmp_dir: str | Path) -> FibBrain:
    """Construct an in-memory FibMind store populated from gaps benchmark memories."""
    db_path = Path(tmp_dir) / "gaps.json"
    service = MemoryService(db_path)

    def populate(memory: FibMind) -> None:
        memories_data = data.get("memories", [])
        for item in memories_data:
            node_id = item["id"]
            status_val = item.get("status", "active")
            status = MemoryStatus(status_val)
            status_reason = item.get("status_reason", "")
            scope_val = item.get("scope", "personal")
            scope = MemoryScope(scope_val)
            confidence = float(item.get("confidence", 0.0))
            meta = dict(item.get("metadata", {}))

            node = MemoryNode(
                id=node_id,
                category=item["category"],
                title=item["title"],
                content=item["content"],
                tags=tuple(item.get("tags", [])),
                scope=scope,
                status=status,
                status_reason=status_reason,
                confidence=confidence,
                metadata=meta,
                node_type=NodeType.RAW,
            )
            memory.nodes[node_id] = node
            memory.lexical.add(node)

            confirmations = int(item.get("confirmations", 0))
            refutations = int(item.get("refutations", 0))
            sessions_count = int(item.get("sessions", 1))
            last_conf_str = item.get("last_confirmation")
            last_conf_dt = (
                datetime.fromisoformat(last_conf_str.replace("Z", "+00:00"))
                if last_conf_str
                else datetime(2026, 10, 15, 0, 0, 0)
            )

            if confirmations > 0 or refutations > 0:
                sessions_set = {f"sess-{i}" for i in range(sessions_count)}
                memory.outcome_index[node_id] = {
                    "confirmed": confirmations,
                    "refuted": refutations,
                    "inconclusive": 0,
                    "sessions": sessions_set,
                    "last_confirmed_at": last_conf_dt if confirmations > 0 else None,
                }
                if confirmations > 0:
                    memory.events.append(
                        MemoryEvent(
                            op=EventOp.OBSERVE,
                            created_at=last_conf_dt,
                            payload={
                                "node_id": node_id,
                                "verdict": "confirmed",
                                "session_id": "sess-0",
                            },
                        )
                    )
                if refutations > 0:
                    memory.events.append(
                        MemoryEvent(
                            op=EventOp.OBSERVE,
                            created_at=last_conf_dt,
                            payload={
                                "node_id": node_id,
                                "verdict": "refuted",
                                "session_id": "sess-0",
                            },
                        )
                    )

    service.run(populate)
    return FibBrain(service)


def run_gaps_evaluation(path: Path | None = None) -> dict[str, Any]:
    """Execute gaps benchmark against knowledge_gaps.json and return summary metrics."""
    gaps_file = path or DEFAULT_GAPS_PATH
    data = json.loads(gaps_file.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        brain = build_gaps_store(data, tmp_dir)

        cases = data.get("cases", [])
        total_cases = len(cases)
        if total_cases == 0:
            return {
                "total_cases": 0,
                "verdict_accuracy": 1.0,
                "false_known_rate": 0.0,
                "false_unknown_rate": 0.0,
                "route_accuracy": 1.0,
                "details": [],
            }

        correct_verdicts = 0
        false_known_count = 0
        false_unknown_count = 0
        correct_routes = 0
        details: list[dict[str, Any]] = []

        for case in cases:
            case_id = case["id"]
            goal = case["goal"]
            expect_verdict = case["expect_verdict"]
            expect_known_ids = set(case.get("expect_known_ids", []))
            expect_uncertain_ids = set(case.get("expect_uncertain_ids", []))
            expect_contradicted_ids = set(case.get("expect_contradicted_ids", []))
            expect_routes = list(case.get("expect_routes", []))

            result = brain.assess(goal)
            actual_verdict = result["verdict"]
            actual_known_ids = {k["node_id"] for k in result["known"]}
            actual_uncertain_ids = {u["node_id"] for u in result["uncertain"]}
            actual_contradicted_ids = {c["node_id"] for c in result["contradicted"]}
            actual_routes = [n["route"] for n in result["next"]]

            verdict_ok = (
                actual_verdict == expect_verdict
                and (not expect_uncertain_ids or bool(actual_uncertain_ids & expect_uncertain_ids))
                and (not expect_contradicted_ids or bool(actual_contradicted_ids & expect_contradicted_ids))
            )
            if verdict_ok:
                correct_verdicts += 1

            # False-known: said known when expected was not known, or put non-expected node in known
            is_false_known = (actual_verdict == "known" and expect_verdict != "known") or bool(
                actual_known_ids - expect_known_ids
            )
            if is_false_known:
                false_known_count += 1

            # False-unknown: said unknown when expected was known or partial
            is_false_unknown = actual_verdict == "unknown" and expect_verdict in ("known", "partial")
            if is_false_unknown:
                false_unknown_count += 1

            # Route matching
            route_ok = True
            for exp_route in expect_routes:
                if exp_route not in actual_routes:
                    route_ok = False
                    break
            if route_ok:
                correct_routes += 1

            details.append(
                {
                    "case_id": case_id,
                    "verdict_ok": verdict_ok,
                    "expect_verdict": expect_verdict,
                    "actual_verdict": actual_verdict,
                    "known_ids": list(actual_known_ids),
                    "uncertain_ids": list(actual_uncertain_ids),
                    "contradicted_ids": list(actual_contradicted_ids),
                    "routes": actual_routes,
                }
            )

    summary = {
        "total_cases": total_cases,
        "verdict_accuracy": round(correct_verdicts / total_cases, 4),
        "false_known_rate": round(false_known_count / total_cases, 4),
        "false_unknown_rate": round(false_unknown_count / total_cases, 4),
        "route_accuracy": round(correct_routes / total_cases, 4),
        "details": details,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps", type=Path, default=DEFAULT_GAPS_PATH)
    args = parser.parse_args(argv)

    summary = run_gaps_evaluation(args.gaps)
    print("=== Gaps Evaluation Report ===")
    print(f"Total Cases:        {summary['total_cases']}")
    print(f"Verdict Accuracy:   {summary['verdict_accuracy']:.2%}")
    print(f"False-Known Rate:   {summary['false_known_rate']:.2%}")
    print(f"False-Unknown Rate: {summary['false_unknown_rate']:.2%}")
    print(f"Route Accuracy:     {summary['route_accuracy']:.2%}")

    if summary["false_known_rate"] > 0.0 or summary["verdict_accuracy"] < 0.90:
        print("\nGaps evaluation failed acceptance thresholds!", file=sys.stderr)
        return 1

    print("\nGaps evaluation passed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
