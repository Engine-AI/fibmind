"""Unit tests for StructuredSummarizer and semantic distillation."""

import unittest

from fibmind import MemoryNode, NodeType
from fibmind.distill import (
    CallableSummarizer,
    ExcerptSummarizer,
    StructuredDistillation,
    StructuredSummarizer,
    verify_distillation_fidelity,
)


class DistillTests(unittest.TestCase):
    def test_excerpt_summarizer_concatenates(self) -> None:
        node1 = MemoryNode(
            id="n1",
            title="Postgres Connection Pool",
            content="Use connection pool size 20 to avoid exhausting database connections during traffic spikes.",
            category="architecture",
            node_type=NodeType.CONCEPT,
        )
        node2 = MemoryNode(
            id="n2",
            title="Redis Cache TTL",
            content="Set Redis cache TTL to 3600 seconds for hot user sessions.",
            category="architecture",
            node_type=NodeType.CONCEPT,
        )

        summarizer = ExcerptSummarizer(excerpt_chars=50)
        summary = summarizer.summarize([node1, node2])

        self.assertIn("Postgres Connection Pool", summary.text)
        self.assertIn("Redis Cache TTL", summary.text)
        self.assertEqual(summary.provenance["summarizer"], "excerpt-v1")
        self.assertEqual(summary.provenance["source_node_ids"], ["n1", "n2"])

    def test_structured_summarizer_preserves_conditions_and_exceptions(self) -> None:
        node = MemoryNode(
            id="n_sql",
            title="SQLAlchemy Bulk Insert",
            content=(
                "Always use explicit session.commit() or conn.begin() block when executing bulk inserts.\n"
                "When running under SQLAlchemy >= 2.0 with engine connections.\n"
                "Avoid raw execute() without commit as changes will be silently discarded.\n"
                "Beware of deadlock when concurrent transactions insert into the same table."
            ),
            category="decision",
            node_type=NodeType.CONCEPT,
            metadata={
                "files": ["src/db/bulk.py"],
                "conditions": ["Python >= 3.11"],
                "exceptions": ["Do not use with read-only replicas"],
            },
            confidence=0.85,
        )

        summarizer = StructuredSummarizer()
        distilled: StructuredDistillation = summarizer.distill_node(node)

        self.assertEqual(distilled.title, "SQLAlchemy Bulk Insert")
        self.assertIn("session.commit()", distilled.directive)
        # Conditions preserved from both metadata and content lines
        self.assertTrue(any("Python >= 3.11" in c for c in distilled.conditions))
        self.assertTrue(any("SQLAlchemy >= 2.0" in c for c in distilled.conditions))
        # Exceptions preserved from both metadata and content lines
        self.assertTrue(any("read-only replicas" in e for e in distilled.exceptions))
        self.assertTrue(any("deadlock" in e for e in distilled.exceptions))
        # Verified by captured files and confidence
        self.assertTrue(any("src/db/bulk.py" in v for v in distilled.verified_by))
        self.assertTrue(any("0.85" in v for v in distilled.verified_by))
        self.assertEqual(distilled.risk_level, "high")

        summary = summarizer.summarize([node])
        self.assertIn("[RULE] SQLAlchemy Bulk Insert:", summary.text)
        self.assertIn("• Conditions:", summary.text)
        self.assertIn("• Exceptions:", summary.text)
        self.assertIn("• VerifiedBy:", summary.text)
        self.assertEqual(summary.provenance["summarizer"], "structured-v1")
        self.assertEqual(summary.provenance["count"], 1)

    def test_callable_summarizer_fidelity_gate_falls_back_on_dropped_constraints(self) -> None:
        node = MemoryNode(
            id="n_fidelity",
            title="Database Transaction Guard",
            content="Always commit inside transaction block.\nDo not call raw close() inside nested handlers.",
            category="decision",
            node_type=NodeType.CONCEPT,
            metadata={"files": ["src/db/tx.py"], "exceptions": ["Never bypass lock on SQLite"]},
        )

        # Lossy LLM summarizer that drops both 'tx.py' and the exceptions
        lossy = CallableSummarizer(
            fn=lambda nodes: "Use transactions for database operations.",
            model="lossy-llm-v1",
            prompt_version="v1",
        )
        summary = lossy.summarize([node])
        self.assertIn("fidelity_check_failed", summary.provenance.get("fallback_from", ""))
        self.assertIn("Never bypass lock on SQLite", summary.text)
        self.assertIn("src/db/tx.py", summary.text)

        # Faithful LLM summarizer that preserves anchors and exceptions passes
        faithful = CallableSummarizer(
            fn=lambda nodes: (
                "In src/db/tx.py, always commit inside transaction block. "
                "Never bypass lock on SQLite and do not call raw close() inside nested handlers."
            ),
            model="faithful-llm-v1",
            prompt_version="v1",
        )
        ok_summary = faithful.summarize([node])
        self.assertEqual(ok_summary.provenance["summarizer"], "callable")
        passed, missing = verify_distillation_fidelity([node], ok_summary.text)
        self.assertTrue(passed)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
