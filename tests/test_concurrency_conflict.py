"""Tests for optimistic concurrency control (OCC), dialectical conflict tracking,
cross-agent provenance verification, and symbol-aware retrieval.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fibmind import (
    ConflictError,
    FibMind,
    MemoryScope,
    MemoryStatus,
    RelationType,
)
from fibmind.service import MemoryService


class ConcurrencyAndConflictTests(unittest.TestCase):
    def test_occ_revision_increments_and_rejects_version_mismatch(self) -> None:
        memory = FibMind()
        node_id = memory.append("decision", "Cache policy", "use 60s TTL")
        node = memory.nodes[node_id]
        self.assertEqual(node.revision, 1)

        # Agent 1 updates with expected_version=1 -> success, revision=2
        updated_1 = memory.revise(node_id, content="use 120s TTL", expected_version=1)
        self.assertEqual(updated_1.revision, 2)
        self.assertEqual(updated_1.content, "use 120s TTL")

        # Agent 2 attempts update with stale expected_version=1 -> raises ConflictError
        with self.assertRaises(ConflictError) as ctx:
            memory.revise(node_id, content="use 300s TTL", expected_version=1)
        self.assertIn("revision 2, expected 1", str(ctx.exception))

        # Replay should restore exact revision state
        replayed = FibMind.rebuild_from_log(memory.events)
        self.assertEqual(replayed.nodes[node_id].revision, 2)
        self.assertEqual(replayed.nodes[node_id].content, "use 120s TTL")

    def test_dialectical_conflict_marking_and_resolution(self) -> None:
        memory = FibMind()
        id_a = memory.append("decision", "SQLite WAL mode", "always enable WAL mode for concurrency")
        id_b = memory.append("decision", "SQLite DELETE mode", "never use WAL mode, use DELETE mode")

        # Mark conflict
        node_a, node_b = memory.mark_conflict(id_a, id_b, reason="contradictory journal modes")
        self.assertTrue(node_a.metadata.get("conflicted"))
        self.assertTrue(node_b.metadata.get("conflicted"))
        self.assertEqual(node_a.metadata.get("conflict_reason"), "contradictory journal modes")

        # Check CONTRADICTS edges exist in both directions
        contradict_edges = [e for e in memory.edges.values() if e.relation_type == RelationType.CONTRADICTS]
        self.assertTrue(any(e.from_node_id == id_a and e.to_node_id == id_b for e in contradict_edges))
        self.assertTrue(any(e.from_node_id == id_b and e.to_node_id == id_a for e in contradict_edges))

        # Resolve conflict: A wins over B based on pytest benchmark
        resolution = memory.resolve_conflict(
            winning_node_id=id_a,
            losing_node_id=id_b,
            reason="WAL mode achieved 10x throughput in benchmark",
            source="pytest",
        )
        self.assertEqual(resolution["winner_id"], id_a)
        self.assertEqual(resolution["loser_id"], id_b)

        # Winning node has conflict markers cleared
        self.assertNotIn("conflicted", memory.nodes[id_a].metadata)
        self.assertNotIn("conflict_reason", memory.nodes[id_a].metadata)

        # Losing node has been refuted
        self.assertEqual(memory.nodes[id_b].status, MemoryStatus.REFUTED)
        self.assertEqual(memory.nodes[id_b].confidence_source, "pytest")

    def test_cross_agent_provenance_gate_on_knowledge_promotion(self) -> None:
        memory = FibMind()
        # Create 3 observations from the same session without physical verification
        id1 = memory.append("decision", "Fact 1", "echo 1", session_id="session_1")
        id2 = memory.append("decision", "Fact 2", "echo 2", session_id="session_1")
        id3 = memory.append("decision", "Fact 3", "echo 3", session_id="session_1")

        # Promotion with require_cross_provenance=True fails due to single-session echo chamber
        with self.assertRaises(ConflictError) as ctx:
            memory.promote_to_knowledge(
                "Unified Fact",
                "general pattern",
                [id1, id2, id3],
                require_cross_provenance=True,
            )
        self.assertIn("requires independent provenance", str(ctx.exception))

        # Add an observation from another session
        id4 = memory.append("decision", "Fact 4", "echo 4 from independent agent", session_id="session_2")
        promoted_id = memory.promote_to_knowledge(
            "Unified Fact",
            "general pattern confirmed across sessions",
            [id1, id2, id4],
            require_cross_provenance=True,
        )
        self.assertIn(promoted_id, memory.nodes)
        self.assertEqual(memory.nodes[promoted_id].scope, MemoryScope.KNOWLEDGE)

    def test_service_and_mcp_concurrency_and_conflict_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(Path(tmp) / "store.db")
            node = service.append("decision", "Auth header", "use Bearer token", owner="alice")
            node_id = node["node_id"]

            # Service revise with OCC
            revised = service.revise(node_id, content="use JWT Bearer", expected_version=1, identity={"owner": "alice"})
            self.assertEqual(revised["revision"], 2)

            with self.assertRaises(ConflictError):
                service.revise(node_id, content="stale", expected_version=1, identity={"owner": "alice"})

            # Service mark conflict and resolve conflict
            node2 = service.append("decision", "Auth header alt", "use Basic auth", owner="alice")
            conflict_res = service.mark_conflict(
                node_id, node2["node_id"], reason="incompatible auth schemes", identity={"owner": "alice"}
            )
            self.assertEqual(conflict_res["reason"], "incompatible auth schemes")

            resolved = service.resolve_conflict(
                winning_node_id=node_id,
                losing_node_id=node2["node_id"],
                reason="RFC compliant",
                source="integration_test",
                identity={"owner": "alice"},
            )
            self.assertEqual(resolved["winner_id"], node_id)

    def test_symbol_and_code_context_anchored_recall(self) -> None:
        memory = FibMind()
        # Node A: anchored to storage.py and SqliteStore
        id_a = memory.append(
            "code",
            "Sqlite connection pooling",
            "use WAL mode and connection pooling",
            metadata={"files": ["src/fibmind/storage.py"], "symbols": ["SqliteStore"]},
        )
        # Node B: generic sqlite note without code anchoring
        memory.append(
            "code",
            "Generic sqlite settings",
            "use connection pooling for database access",
        )

        # Query targeting the specific file and symbol
        hits = memory.search("storage.py SqliteStore connection pooling")
        self.assertTrue(len(hits) >= 2)
        # Node A should rank highest due to symbol boost
        self.assertEqual(hits[0].node.id, id_a)

        # Explain recall reports symbol boost and symbol source
        explanation = memory.explain_recall("storage.py SqliteStore connection pooling")
        top_result = explanation["results"][0]
        self.assertEqual(top_result["node_id"], id_a)
        self.assertGreater(top_result["symbol_boost"], 0.0)
        self.assertIn("symbol", top_result["sources"])


if __name__ == "__main__":
    unittest.main()
