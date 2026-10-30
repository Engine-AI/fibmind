"""Tests for Track 1: Dao Epistemology & Dialectical Graph Evolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from fibmind.admission import (
    MemoryCandidate,
    detect_version_predecessors,
    discover_knowledge_clusters,
    propose_conflict_synthesis,
)
from fibmind.assessment import calibrate_trust_floor, classify, route_uncertain
from fibmind.background import build_background
from fibmind.brain import FibBrain
from fibmind.context import build_context
from fibmind.distill import ExcerptSummarizer, StructuredSummarizer, verify_distillation_fidelity
from fibmind.fibonacci import FibonacciLayerPolicy
from fibmind.graph import RING3_CONFIDENCE_CAP, FibMind
from fibmind.models import (
    EdgeDirection,
    MemoryNode,
    MemoryScope,
    MemoryStatus,
    NodeType,
    RelationType,
    Verdict,
)
from fibmind.replay import apply_event
from fibmind.retrieval import RankingWeights, epistemic_topology_boost, retrieve
from fibmind.service import MemoryService
from fibmind.tuning import gate, profile_evidence
from fibmind.verify import run_command_and_reflect


class DaoEpistemologyTests(unittest.TestCase):
    def test_trust_ring_validation(self) -> None:
        mind = FibMind()
        node_id = mind.append("decision", "Test Node", "Content for test", scope=MemoryScope.PERSONAL, owner="abbila")

        # Invalid trust ring values
        with self.assertRaises(ValueError):
            mind.record_outcome(node_id, Verdict.CONFIRMED, source="test", trust_ring="ring1")
        with self.assertRaises(ValueError):
            mind.record_outcome(node_id, Verdict.CONFIRMED, source="test", trust_ring="invalid")

        # Valid trust ring values (case insensitive)
        mind.record_outcome(node_id, Verdict.CONFIRMED, source="test", trust_ring="RING0")
        self.assertEqual(mind.nodes[node_id].metadata["last_trust_ring"], "ring0")

        mind.record_outcome(node_id, Verdict.CONFIRMED, source="test", trust_ring="Ring3")
        self.assertEqual(mind.nodes[node_id].metadata["last_trust_ring"], "ring3")

        mind.record_outcome(node_id, Verdict.CONFIRMED, source="test", trust_ring=None)
        # When trust_ring is None, last_trust_ring in metadata is not overwritten to None
        self.assertEqual(mind.nodes[node_id].metadata["last_trust_ring"], "ring3")

    def test_trust_ring_confidence_cap_and_observe_event(self) -> None:
        mind = FibMind()
        node_id = mind.append(
            "decision", "Architecture Rule", "Clean boundaries", scope=MemoryScope.PERSONAL, owner="abbila"
        )
        self.assertEqual(mind.nodes[node_id].confidence, 0.0)

        # Confirm step 1 under ring3: 0.0 -> 0.25
        n1 = mind.record_outcome(node_id, Verdict.CONFIRMED, source="self_eval", trust_ring="ring3")
        self.assertEqual(n1.confidence, 0.25)
        self.assertEqual(mind.events[-1].payload.get("trust_ring"), "ring3")

        # Confirm step 2 under ring3: 0.25 -> 0.50
        n2 = mind.record_outcome(node_id, Verdict.CONFIRMED, source="self_eval", trust_ring="ring3")
        self.assertEqual(n2.confidence, 0.50)

        # Confirm step 3 under ring3: 0.50 + 0.25 -> capped at 0.60 (RING3_CONFIDENCE_CAP)
        n3 = mind.record_outcome(node_id, Verdict.CONFIRMED, source="self_eval", trust_ring="ring3")
        self.assertEqual(n3.confidence, RING3_CONFIDENCE_CAP)
        self.assertEqual(n3.confidence, 0.60)

        # Confirm step 4 under ring3: remains 0.60
        n4 = mind.record_outcome(node_id, Verdict.CONFIRMED, source="self_eval", trust_ring="ring3")
        self.assertEqual(n4.confidence, 0.60)

        # Physical confirmation under ring0 breaks through the cap: 0.60 + 0.25 = 0.85
        n5 = mind.record_outcome(node_id, Verdict.CONFIRMED, source="pytest", trust_ring="ring0")
        self.assertEqual(n5.confidence, 0.85)
        self.assertEqual(mind.events[-1].payload.get("trust_ring"), "ring0")

        # Second ring0 confirmation: 0.85 + 0.25 = 1.0 (capped at max 1.0)
        n6 = mind.record_outcome(node_id, Verdict.CONFIRMED, source="pytest", trust_ring="ring0")
        self.assertEqual(n6.confidence, 1.0)

        # Subsequent ring3 assertion does not reduce confidence when >= 0.60
        n7 = mind.record_outcome(node_id, Verdict.CONFIRMED, source="self_eval", trust_ring="ring3")
        self.assertEqual(n7.confidence, 1.0)

    def test_protection_of_pending_revisions_against_ring3(self) -> None:
        mind = FibMind()
        # Original verified memory node
        node1_id = mind.append(
            "decision",
            "Cache Policy",
            "Use LRU cache with maxsize 100",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            status=MemoryStatus.ACTIVE,
        )
        mind.record_outcome(node1_id, Verdict.CONFIRMED, source="physical_test", trust_ring="ring0")
        self.assertEqual(mind.nodes[node1_id].status, MemoryStatus.ACTIVE)

        # Newer revision submitted as PENDING
        node2_id = mind.append(
            "decision",
            "Cache Policy",
            "Use LRU cache with maxsize 500 and TTL",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            status=MemoryStatus.PENDING,
            metadata={"base_revision": mind.nodes[node1_id].revision},
        )
        mind.link_nodes(node2_id, node1_id, RelationType.VERSION_OF, direction=EdgeDirection.DIRECTED)

        # Ring 3 self-assertion attempts to confirm node2
        mind.record_outcome(node2_id, Verdict.CONFIRMED, source="llm_self_check", trust_ring="ring3")

        # Crucial epistemic invariants:
        # 1. node2 must remain PENDING
        self.assertEqual(mind.nodes[node2_id].status, MemoryStatus.PENDING)
        # 2. node1 must remain ACTIVE (not marked stale!)
        self.assertEqual(mind.nodes[node1_id].status, MemoryStatus.ACTIVE)
        # 3. node2 confidence increased up to cap
        self.assertEqual(mind.nodes[node2_id].confidence, 0.25)

        # Now, Ring 0 verification confirms node2
        mind.record_outcome(node2_id, Verdict.CONFIRMED, source="integration_test", trust_ring="ring0")

        # Now node2 is promoted to ACTIVE
        self.assertEqual(mind.nodes[node2_id].status, MemoryStatus.ACTIVE)
        self.assertEqual(mind.nodes[node2_id].status_reason, "verified by integration_test")
        # And node1 is superseded and marked STALE
        self.assertEqual(mind.nodes[node1_id].status, MemoryStatus.STALE)

    def test_detect_version_predecessors_matching(self) -> None:
        mind = FibMind()
        base_id = mind.append(
            "pattern",
            "Database Connection Retry",
            "Always retry database connections with exponential backoff.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            status=MemoryStatus.ACTIVE,
            metadata={"files": ["src/db/connection.py"]},
        )

        # Condition A: Matching normalized title, different content
        cand_a = MemoryCandidate(
            category="pattern",
            title="database   connection retry ",
            content="Always retry database connections with jittered exponential backoff.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        preds_a = detect_version_predecessors(mind, cand_a)
        self.assertEqual(preds_a, [base_id])

        # Condition B: Different title, but shared files and high core token Jaccard similarity (>= 0.45)
        cand_b = MemoryCandidate(
            category="pattern",
            title="Database Connection Backoff Strategy",
            content="Always retry database connections with exponential backoff strategy.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        preds_b = detect_version_predecessors(mind, cand_b, metadata={"files": ["src/db/connection.py"]})
        self.assertEqual(preds_b, [base_id])

        # Exact duplicate must be excluded
        cand_dup = MemoryCandidate(
            category="pattern",
            title="Database Connection Retry",
            content="Always retry database connections with exponential backoff.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        preds_dup = detect_version_predecessors(mind, cand_dup)
        self.assertEqual(preds_dup, [])

        # Opposite polarity (negation) must NOT be a version predecessor (it's a conflict)
        cand_neg = MemoryCandidate(
            category="pattern",
            title="Database Connection Retry",
            content="Never retry database connections with exponential backoff.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        preds_neg = detect_version_predecessors(mind, cand_neg)
        self.assertEqual(preds_neg, [])

        # Different category must not match
        cand_cat = MemoryCandidate(
            category="architecture",
            title="Database Connection Retry",
            content="Always retry database connections with exponential backoff.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        preds_cat = detect_version_predecessors(mind, cand_cat)
        self.assertEqual(preds_cat, [])

    def test_service_remember_dialectical_graph_auto_linking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = MemoryService(Path(tmpdir) / "epistemic.db")

            # 1. Base memory
            r1 = service.remember(
                MemoryCandidate(
                    category="decision",
                    title="WAL Journal Mode",
                    content="Always configure SQLite with PRAGMA journal_mode=WAL for concurrency.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                ),
                metadata={"files": ["src/storage.py"]},
            )
            node1_id = r1["node_id"]

            # 2. Version successor (same title, improved content, same files)
            r2 = service.remember(
                MemoryCandidate(
                    category="decision",
                    title="WAL Journal Mode",
                    content="Always configure SQLite with PRAGMA journal_mode=WAL and synchronous=NORMAL.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                ),
                metadata={"files": ["src/storage.py"]},
            )
            node2_id = r2["node_id"]
            self.assertIn("version_predecessors", r2)
            self.assertEqual(r2["version_predecessors"], [node1_id])

            # Check that directed VERSION_OF edge was created from node2 to node1
            with service._lock:
                mem = service._load()
                v_edges = [
                    edge
                    for edge in mem.edges.values()
                    if edge.relation_type == RelationType.VERSION_OF
                    and edge.from_node_id == node2_id
                    and edge.to_node_id == node1_id
                ]
                self.assertEqual(len(v_edges), 1)
                self.assertEqual(v_edges[0].direction, EdgeDirection.DIRECTED)
                self.assertTrue(v_edges[0].metadata.get("auto_detected"))

            # 3. Contradicting memory (opposing negation)
            r3 = service.remember(
                MemoryCandidate(
                    category="decision",
                    title="WAL Journal Mode",
                    content="Never configure SQLite with PRAGMA journal_mode=WAL for concurrency.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                ),
                metadata={"files": ["src/storage.py"]},
            )
            node3_id = r3["node_id"]
            self.assertIn("potential_conflicts", r3)
            self.assertIn(node1_id, r3["potential_conflicts"])

            # Check that bidirectional CONTRADICTS edges were created
            with service._lock:
                mem = service._load()
                c_edges = [
                    edge
                    for edge in mem.edges.values()
                    if edge.relation_type == RelationType.CONTRADICTS
                    and {edge.from_node_id, edge.to_node_id} == {node3_id, node1_id}
                ]
                self.assertEqual(len(c_edges), 1)
                self.assertEqual(c_edges[0].direction, EdgeDirection.BIDIRECTIONAL)
                self.assertTrue(c_edges[0].metadata.get("auto_detected"))

    def test_epistemic_compression_modes(self) -> None:
        mind = FibMind()

        # Case 1: all_verified
        v1 = MemoryNode(
            title="Fact 1",
            content="Verified observation 1",
            category="pattern",
            layer="raw",
            confidence=0.8,
            confidence_source="test_runner",
            memory_weight=1,
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        v2 = MemoryNode(
            title="Fact 2",
            content="Verified observation 2",
            category="pattern",
            layer="raw",
            confidence=0.9,
            confidence_source="test_runner",
            memory_weight=1,
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind._add_node(v1)
        mind._add_node(v2)
        comp1_id = mind._compress_nodes("pattern", "raw", "compressed", [v1, v2])
        comp1 = mind.nodes[comp1_id]
        self.assertEqual(comp1.metadata.get("epistemic_mode"), "all_verified")
        self.assertAlmostEqual(comp1.confidence, 0.85, places=2)

        # Case 2: mixed_support_bounded (unverified 0.0 mixed with verified 0.9)
        u1 = MemoryNode(
            title="Observation Unverified",
            content="Speculative observation",
            category="pattern",
            layer="raw",
            confidence=0.0,
            memory_weight=1,
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        v3 = MemoryNode(
            title="Observation Verified High",
            content="Strong observation",
            category="pattern",
            layer="raw",
            confidence=1.0,
            confidence_source="formal_proof",
            memory_weight=9,  # High weight: (0*1 + 1.0*9)/10 = 0.9, would normally be 0.9
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind._add_node(u1)
        mind._add_node(v3)
        comp2_id = mind._compress_nodes("pattern", "raw", "compressed", [u1, v3])
        comp2 = mind.nodes[comp2_id]
        self.assertEqual(comp2.metadata.get("epistemic_mode"), "mixed_support_bounded")
        # Must be capped at RING3_CONFIDENCE_CAP (0.60) instead of 0.90!
        self.assertEqual(comp2.confidence, RING3_CONFIDENCE_CAP)

        # Case 3: unverified (all confidence 0.0)
        u2 = MemoryNode(
            title="Draft 1",
            content="Draft text 1",
            category="pattern",
            layer="raw",
            confidence=0.0,
            memory_weight=1,
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        u3 = MemoryNode(
            title="Draft 2",
            content="Draft text 2",
            category="pattern",
            layer="raw",
            confidence=0.0,
            memory_weight=1,
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind._add_node(u2)
        mind._add_node(u3)
        comp3_id = mind._compress_nodes("pattern", "raw", "compressed", [u2, u3])
        comp3 = mind.nodes[comp3_id]
        self.assertEqual(comp3.metadata.get("epistemic_mode"), "unverified")
        self.assertEqual(comp3.confidence, 0.0)

    def test_service_record_outcome_plumbing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = MemoryService(Path(tmpdir) / "service_ring.db")
            res = service.remember(
                MemoryCandidate(
                    category="note",
                    title="Service Outcome Test",
                    content="Plumbing trust_ring through service.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                )
            )
            node_id = res["node_id"]

            # Pass trust_ring="ring3" via service
            outcome3 = service.record_outcome(node_id, "confirmed", "self_test", trust_ring="ring3")
            self.assertEqual(outcome3["confidence"], 0.25)

            # Pass trust_ring="ring0" via service
            outcome0 = service.record_outcome(node_id, "confirmed", "pytest", trust_ring="ring0")
            self.assertEqual(outcome0["confidence"], 0.50)

    def test_verify_run_command_and_reflect_ring0(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = MemoryService(Path(tmpdir) / "verify_ring.db")
            brain = FibBrain(service)
            res = service.remember(
                MemoryCandidate(
                    category="code",
                    title="Verify Ring0 Test",
                    content="Verification command execution test.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                ),
                metadata={"files": ["test_target.py"]},
            )
            node_id = res["node_id"]

            # Run a simple echo command touching target
            result = run_command_and_reflect(
                ["echo", "PASSED test_target.py"],
                node_id=node_id,
                target_files=["test_target.py"],
                service=service,
                brain=brain,
                owner="abbila",
            )
            self.assertEqual(result["attribution"]["trust_ring"], "ring0")
            self.assertEqual(result["verdict"], "confirmed")

    def test_synthesize_conflict_dialectics_and_replay(self) -> None:
        mind = FibMind()

        # Thesis: Node A
        node_a_id = mind.append(
            "decision",
            "Use WAL Journal Mode",
            "Always use PRAGMA journal_mode=WAL for SQLite concurrency.",
            tags=["sqlite", "perf"],
            metadata={"files": ["src/db.py"], "conflicted": True, "conflict_reason": "opposes DELETE"},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind.record_outcome(node_a_id, Verdict.CONFIRMED, source="test_suite", trust_ring="ring0")

        # Antithesis: Node B
        node_b_id = mind.append(
            "decision",
            "Use DELETE Journal Mode",
            "Always use PRAGMA journal_mode=DELETE for SQLite low memory.",
            tags=["sqlite", "embedded"],
            metadata={"files": ["src/db.py"], "conflicted": True, "conflict_reason": "opposes WAL"},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind.record_outcome(node_b_id, Verdict.CONFIRMED, source="test_suite", trust_ring="ring0")

        # Synthesize conflict
        synth_node = mind.synthesize_conflict(
            node_a_id,
            node_b_id,
            title="SQLite Journal Mode Selection Strategy",
            content="Use WAL mode for multi-threaded concurrency; use DELETE mode for memory-constrained embedded environments.",
            conditions_a=["multi-process concurrent readers", "NVMe storage"],
            conditions_b=["single process memory constrained", "network filesystem"],
            source="dialectical_analysis",
        )

        # 1. Properties of synthesized node
        self.assertEqual(synth_node.title, "SQLite Journal Mode Selection Strategy")
        self.assertEqual(synth_node.tags, {"embedded", "perf", "sqlite", "synthesized"})
        self.assertEqual(synth_node.metadata["synthesized_from"], [node_a_id, node_b_id])
        self.assertEqual(synth_node.metadata["files"], ["src/db.py"])
        self.assertEqual(
            synth_node.metadata["conditions"],
            [
                "multi-process concurrent readers",
                "NVMe storage",
                "single process memory constrained",
                "network filesystem",
            ],
        )
        self.assertEqual(len(synth_node.metadata["branches"]), 2)
        self.assertEqual(synth_node.metadata["branches"][0]["node_id"], node_a_id)
        self.assertEqual(synth_node.metadata["branches"][1]["node_id"], node_b_id)
        self.assertEqual(synth_node.confidence, 0.25)

        # 2. Status of folded nodes A and B
        node_a = mind.nodes[node_a_id]
        node_b = mind.nodes[node_b_id]
        self.assertEqual(node_a.folded_into, synth_node.id)
        self.assertEqual(node_b.folded_into, synth_node.id)
        self.assertNotIn("conflicted", node_a.metadata)
        self.assertNotIn("conflicted", node_b.metadata)
        self.assertNotIn("conflict_reason", node_a.metadata)
        self.assertNotIn("conflict_reason", node_b.metadata)
        self.assertEqual(node_a.metadata["conditions"], ["multi-process concurrent readers", "NVMe storage"])
        self.assertEqual(node_b.metadata["conditions"], ["single process memory constrained", "network filesystem"])

        # 3. Graph relations: DERIVED_FROM and VERSION_OF edges
        derived_edges = [
            e
            for e in mind.edges.values()
            if e.relation_type == RelationType.DERIVED_FROM and e.from_node_id == synth_node.id
        ]
        version_edges = [
            e
            for e in mind.edges.values()
            if e.relation_type == RelationType.VERSION_OF and e.from_node_id == synth_node.id
        ]
        self.assertEqual({e.to_node_id for e in derived_edges}, {node_a_id, node_b_id})
        self.assertEqual({e.to_node_id for e in version_edges}, {node_a_id, node_b_id})

        # 4. Replay equivalence
        rebuilt = FibMind()
        for ev in mind.events:
            apply_event(rebuilt, ev)
        self.assertEqual(rebuilt.nodes[node_a_id].folded_into, synth_node.id)
        self.assertEqual(rebuilt.nodes[node_b_id].folded_into, synth_node.id)
        self.assertEqual(rebuilt.nodes[synth_node.id].confidence, synth_node.confidence)
        self.assertEqual(rebuilt.nodes[synth_node.id].title, synth_node.title)

    def test_service_synthesize_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = MemoryService(Path(tmpdir) / "synth_svc.db")

            r1 = service.remember(
                MemoryCandidate(
                    category="decision",
                    title="Pragma Synchronous NORMAL",
                    content="Always use synchronous=NORMAL for SQLite performance.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                )
            )
            r2 = service.remember(
                MemoryCandidate(
                    category="decision",
                    title="Pragma Synchronous FULL",
                    content="Always use synchronous=FULL for SQLite zero data loss.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                )
            )

            synth_res = service.synthesize_conflict(
                r1["node_id"],
                r2["node_id"],
                title="SQLite Synchronous Pragma Strategy",
                content="Use NORMAL in WAL mode on UPS; use FULL on untrusted power supplies.",
                conditions_a=["WAL mode with UPS"],
                conditions_b=["power loss vulnerability without battery backup"],
                source="architectural_synthesis",
                identity={"owner": "abbila"},
            )

            self.assertEqual(synth_res["title"], "SQLite Synchronous Pragma Strategy")
            self.assertEqual(synth_res["synthesized_from"], [r1["node_id"], r2["node_id"]])

            # Inspect to verify both original nodes are now folded
            f1 = service.read(lambda m: m.nodes[r1["node_id"]].folded_into)
            f2 = service.read(lambda m: m.nodes[r2["node_id"]].folded_into)
            self.assertEqual(f1, synth_res["node_id"])
            self.assertEqual(f2, synth_res["node_id"])

    def test_discover_knowledge_clusters_and_auto_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = MemoryService(Path(tmpdir) / "clusters.db")

            # 3 active verified personal observations across 2 sessions (distinct observations sharing src/auth.py)
            c1 = service.remember(
                MemoryCandidate(
                    category="pattern",
                    title="JWT Bearer Token Preemptive Refresh",
                    content="Always refresh JWT bearer token 60s before expiration timestamp.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                    session_id="session-1",
                ),
                metadata={"files": ["src/auth.py"]},
            )
            service.record_outcome(c1["node_id"], "confirmed", "pytest", trust_ring="ring0")

            c2 = service.remember(
                MemoryCandidate(
                    category="pattern",
                    title="JWT Bearer Token Signature Validation",
                    content="Enforce strict HMAC SHA256 signature verification on incoming bearer tokens.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                    session_id="session-1",
                ),
                metadata={"files": ["src/auth.py"]},
            )
            service.record_outcome(c2["node_id"], "confirmed", "pytest", trust_ring="ring0")

            c3 = service.remember(
                MemoryCandidate(
                    category="pattern",
                    title="JWT Bearer Token Header Sanitization",
                    content="Strip extraneous authentication headers before forwarding claims downstream.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                    session_id="session-2",
                ),
                metadata={"files": ["src/auth.py"]},
            )
            service.record_outcome(c3["node_id"], "confirmed", "pytest", trust_ring="ring0")

            # Unverified observation (confidence = 0.0) -> must be excluded
            _c4 = service.remember(
                MemoryCandidate(
                    category="pattern",
                    title="Token Refresh Speculation",
                    content="Refresh JWT token with magical delay speculation.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                    session_id="session-2",
                ),
                metadata={"files": ["src/auth.py"]},
            )

            # Direct admission call preview
            direct_clusters = discover_knowledge_clusters(
                service._load(),
                owner="abbila",
                min_supporters=3,
                min_sessions=2,
            )
            self.assertEqual(len(direct_clusters), 1)

            # 1. Preview mode (auto_promote=False)
            preview = service.discover_knowledge_clusters(
                min_supporters=3,
                min_sessions=2,
                auto_promote=False,
                identity={"owner": "abbila"},
            )
            self.assertEqual(len(preview["clusters"]), 1)
            self.assertEqual(preview["promoted"], [])
            cluster = preview["clusters"][0]
            self.assertEqual(cluster["category"], "pattern")
            self.assertEqual(len(cluster["supporting_node_ids"]), 3)
            self.assertEqual(set(cluster["supporting_node_ids"]), {c1["node_id"], c2["node_id"], c3["node_id"]})
            self.assertIn("src/auth.py", cluster["shared_anchors"])
            self.assertEqual(cluster["sessions"], ["session-1", "session-2"])

            # 2. Auto-promote mode (auto_promote=True)
            promoted_run = service.discover_knowledge_clusters(
                min_supporters=3,
                min_sessions=2,
                auto_promote=True,
                identity={"owner": "abbila"},
            )
            self.assertEqual(len(promoted_run["promoted"]), 1)
            k_node_summary = promoted_run["promoted"][0]
            self.assertEqual(k_node_summary["scope"], MemoryScope.KNOWLEDGE.value)
            self.assertEqual(k_node_summary["category"], "pattern")

            # Verify promoted node exists in storage
            k_inspected = service.inspect(k_node_summary["node_id"])
            self.assertEqual(k_inspected["scope"], MemoryScope.KNOWLEDGE.value)
            self.assertEqual(
                set(k_inspected["metadata"]["supporting_node_ids"]), {c1["node_id"], c2["node_id"], c3["node_id"]}
            )

            # 3. Subsequent call should discover 0 clusters (since these nodes are now already promoted)
            subsequent = service.discover_knowledge_clusters(
                min_supporters=3,
                min_sessions=2,
                auto_promote=False,
                identity={"owner": "abbila"},
            )
            self.assertEqual(len(subsequent["clusters"]), 0)

    def test_shadow_revision_promotion_and_predecessor_retirement(self) -> None:
        mind = FibMind()
        # Original verified memory node
        old_id = mind.append(
            "pattern",
            "Initial Verified Pattern",
            "Original verified algorithm implementation.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            status=MemoryStatus.ACTIVE,
        )
        mind.record_outcome(old_id, Verdict.CONFIRMED, source="test_suite", trust_ring="ring0")
        self.assertEqual(mind.nodes[old_id].confidence, 0.25)
        self.assertEqual(mind.nodes[old_id].status, MemoryStatus.ACTIVE)

        # Create shadow candidate with protect_verified=True
        shadow_node = mind.revise(
            old_id,
            title="Upgraded Verified Pattern",
            content="Upgraded and optimized algorithm implementation.",
            protect_verified=True,
        )
        self.assertEqual(shadow_node.status, MemoryStatus.PENDING)
        self.assertEqual(shadow_node.metadata.get("supersedes_node_id"), old_id)
        self.assertEqual(mind.nodes[old_id].status, MemoryStatus.ACTIVE)

        # Ring 3 confirmation must NOT promote shadow revision or retire predecessor
        mind.record_outcome(shadow_node.id, Verdict.CONFIRMED, source="self_reflect", trust_ring="ring3")
        self.assertEqual(mind.nodes[shadow_node.id].status, MemoryStatus.PENDING)
        self.assertEqual(mind.nodes[old_id].status, MemoryStatus.ACTIVE)

        # Ring 0 confirmation promotes shadow revision to ACTIVE and marks predecessor STALE
        mind.record_outcome(shadow_node.id, Verdict.CONFIRMED, source="physical_verification", trust_ring="ring0")
        self.assertEqual(mind.nodes[shadow_node.id].status, MemoryStatus.ACTIVE)
        self.assertTrue(mind.nodes[shadow_node.id].metadata.get("promoted_from_shadow"))
        self.assertEqual(mind.nodes[old_id].status, MemoryStatus.STALE)

    def test_multi_hop_cascading_belief_revision(self) -> None:
        mind = FibMind()

        # Hop 0: Base observation
        obs_id = mind.append(
            "pattern",
            "Observed Cache Metric",
            "Cache hit ratio measured at 99.2% on standard load.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind.record_outcome(obs_id, Verdict.CONFIRMED, source="benchmark", trust_ring="ring0")
        mind.record_outcome(obs_id, Verdict.CONFIRMED, source="benchmark", trust_ring="ring0")

        # Hop 1: Personal derived decision depending on Hop 0
        dec_id = mind.append(
            "decision",
            "Keep Cache Sizing",
            "Maintain current cache allocation based on observed 99.2% hit ratio.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind.link_nodes(dec_id, obs_id, RelationType.DERIVED_FROM, weight=1.0)
        # Give dec_id a high confidence (e.g. 0.75)
        mind.record_outcome(dec_id, Verdict.CONFIRMED, source="perf_eval", trust_ring="ring0")
        mind.record_outcome(dec_id, Verdict.CONFIRMED, source="perf_eval", trust_ring="ring0")
        mind.record_outcome(dec_id, Verdict.CONFIRMED, source="perf_eval", trust_ring="ring0")
        self.assertGreater(mind.nodes[dec_id].confidence, 0.25)

        # Hop 2: Knowledge claim depending on Hop 1
        know_id = mind.append(
            "knowledge",
            "Standard Cache Sizing Law",
            "Global rule for cache sizing across all projects.",
            scope=MemoryScope.KNOWLEDGE,
            auto_link=False,
        )
        mind.link_nodes(know_id, dec_id, RelationType.DERIVED_FROM, weight=1.0)
        mind.record_outcome(know_id, Verdict.CONFIRMED, source="team_consensus", trust_ring="ring0")
        self.assertEqual(mind.nodes[know_id].status, MemoryStatus.ACTIVE)

        # Now refute base observation (Hop 0)
        mind.record_outcome(obs_id, Verdict.REFUTED, source="production_incident_investigation")

        # Verify cascade across hops:
        # 1. Hop 0 is REFUTED
        self.assertEqual(mind.nodes[obs_id].status, MemoryStatus.REFUTED)

        # 2. Hop 1 (personal) confidence demoted to 0.25 and recorded upstream_refuted
        dec_node = mind.nodes[dec_id]
        self.assertEqual(dec_node.confidence, 0.25)
        self.assertIn(obs_id, dec_node.metadata.get("upstream_refuted", []))

        # 3. Hop 2 (knowledge) retired to STALE
        know_node = mind.nodes[know_id]
        self.assertEqual(know_node.status, MemoryStatus.STALE)

    def test_context_aware_branch_activation(self) -> None:
        mind = FibMind()

        # Node A
        node_a_id = mind.append(
            "decision",
            "WAL Mode",
            "Configure journal_mode=WAL for high concurrency.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        # Node B
        node_b_id = mind.append(
            "decision",
            "DELETE Mode",
            "Configure journal_mode=DELETE for low memory footprint.",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )

        # Synthesize conflict
        synth_node = mind.synthesize_conflict(
            node_a_id,
            node_b_id,
            title="SQLite Journaling Strategy",
            content="Context-aware SQLite journal configuration.",
            conditions_a=["multi-process concurrent readers", "NVMe storage"],
            conditions_b=["single process memory constrained", "embedded IoT system"],
            source="architecture_team",
        )

        summarizer = StructuredSummarizer()

        # 1. Goal matching Branch B
        goal_b = "We are deploying to an embedded IoT system with single process memory constrained."
        dist_b = summarizer.distill_node(synth_node, goal=goal_b)
        self.assertIsNotNone(dist_b.active_branch)
        self.assertEqual(dist_b.active_branch["node_id"], node_b_id)
        rendered_b = dist_b.to_text()
        self.assertIn("ActiveBranch (single process memory constrained, embedded IoT system): DELETE Mode", rendered_b)

        # 2. Goal matching Branch A
        goal_a = "Building a high-throughput server with multi-process concurrent readers on NVMe storage."
        dist_a = summarizer.distill_node(synth_node, goal=goal_a)
        self.assertIsNotNone(dist_a.active_branch)
        self.assertEqual(dist_a.active_branch["node_id"], node_a_id)
        rendered_a = dist_a.to_text()
        self.assertIn("ActiveBranch (multi-process concurrent readers, NVMe storage): WAL Mode", rendered_a)

        # 3. Goal without matching conditions
        goal_unrelated = "Completely unrelated task about frontend CSS animations."
        dist_unrelated = summarizer.distill_node(synth_node, goal=goal_unrelated)
        self.assertIsNone(dist_unrelated.active_branch)
        self.assertNotIn("ActiveBranch", dist_unrelated.to_text())

        # 4. No goal provided
        dist_none = summarizer.distill_node(synth_node)
        self.assertIsNone(dist_none.active_branch)
        self.assertNotIn("ActiveBranch", dist_none.to_text())

    def test_full_replay_equivalence_epistemic_mutations(self) -> None:
        mind = FibMind()

        # 1. Shadow revision promotion
        # Create verified node
        v_node_id = mind.append(
            "decision",
            "Database Schema",
            "Initial schema with v1 tables",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            status=MemoryStatus.ACTIVE,
        )
        mind.record_outcome(v_node_id, Verdict.CONFIRMED, source="test", trust_ring="ring0")

        # Staged pending revision
        shadow_id = mind.append(
            "decision",
            "Database Schema v2",
            "Updated schema with v2 tables",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            status=MemoryStatus.PENDING,
            metadata={"supersedes_node_id": v_node_id},
            auto_link=False,
        )
        mind.link_nodes(shadow_id, v_node_id, RelationType.VERSION_OF, weight=1.0)

        # Confirm shadow node with ring0 -> promotes shadow, marks v_node_id STALE
        mind.record_outcome(shadow_id, Verdict.CONFIRMED, source="migration_test", trust_ring="ring0")
        self.assertEqual(mind.nodes[shadow_id].status, MemoryStatus.ACTIVE)
        self.assertTrue(mind.nodes[shadow_id].metadata.get("promoted_from_shadow"))
        self.assertEqual(mind.nodes[v_node_id].status, MemoryStatus.STALE)
        self.assertEqual(mind.nodes[v_node_id].status_reason, f"superseded by verified revision {shadow_id}")

        # 2. Multi-hop invalidation cascade
        obs_id = mind.append(
            "pattern",
            "Base Metric Observation",
            "Observed cache latency under 5ms",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind.record_outcome(obs_id, Verdict.CONFIRMED, source="benchmark", trust_ring="ring0")

        dec_id = mind.append(
            "decision",
            "Latency Optimization Decision",
            "Use in-memory cache to guarantee under 5ms latency",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind.link_nodes(dec_id, obs_id, RelationType.DERIVED_FROM, weight=1.0)
        mind.record_outcome(dec_id, Verdict.CONFIRMED, source="benchmark", trust_ring="ring0")

        know_id = mind.append(
            "knowledge",
            "Latency Invariant Rule",
            "Global system rule regarding cache latency SLA",
            scope=MemoryScope.KNOWLEDGE,
            auto_link=False,
        )
        mind.link_nodes(know_id, dec_id, RelationType.DERIVED_FROM, weight=1.0)
        mind.record_outcome(know_id, Verdict.CONFIRMED, source="sla_audit", trust_ring="ring0")

        # Refute base observation
        mind.record_outcome(obs_id, Verdict.REFUTED, source="sla_breach_investigation")
        self.assertEqual(mind.nodes[obs_id].status, MemoryStatus.REFUTED)
        self.assertEqual(mind.nodes[dec_id].confidence, 0.25)
        self.assertIn(obs_id, mind.nodes[dec_id].metadata.get("upstream_refuted", []))
        self.assertEqual(mind.nodes[know_id].status, MemoryStatus.STALE)

        # 3. Dialectical conflict synthesis
        node_a = mind.append(
            "decision",
            "Sync Mode",
            "Use synchronous disk flush",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        node_b = mind.append(
            "decision",
            "Async Mode",
            "Use asynchronous disk flush",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        mind.mark_conflict(node_a, node_b, reason="Flush mode contradiction")
        self.assertTrue(mind.nodes[node_a].metadata.get("conflicted"))
        self.assertTrue(mind.nodes[node_b].metadata.get("conflicted"))

        synth_node = mind.synthesize_conflict(
            node_a,
            node_b,
            title="Disk Flush Strategy",
            content="Synthesized disk flush policy",
            conditions_a=["financial transaction data"],
            conditions_b=["temporary metric data"],
            source="architecture_review",
        )
        self.assertIn("synthesized", synth_node.tags)
        self.assertEqual(synth_node.metadata.get("synthesized_from"), [node_a, node_b])
        self.assertNotIn("conflicted", mind.nodes[node_a].metadata)
        self.assertEqual(mind.nodes[node_a].metadata.get("conditions"), ["financial transaction data"])
        self.assertNotIn("conflicted", mind.nodes[node_b].metadata)
        self.assertEqual(mind.nodes[node_b].metadata.get("conditions"), ["temporary metric data"])

        # Replay entire event stream from scratch
        replayed = FibMind.rebuild_from_log(mind.events)

        # Assert full replay equivalence across all mutated nodes
        # 1. Shadow promotion equivalence
        self.assertEqual(replayed.nodes[shadow_id].status, MemoryStatus.ACTIVE)
        self.assertTrue(replayed.nodes[shadow_id].metadata.get("promoted_from_shadow"))
        self.assertEqual(replayed.nodes[v_node_id].status, MemoryStatus.STALE)
        self.assertEqual(replayed.nodes[v_node_id].status_reason, f"superseded by verified revision {shadow_id}")

        # 2. Invalidation cascade equivalence
        self.assertEqual(replayed.nodes[obs_id].status, MemoryStatus.REFUTED)
        self.assertEqual(replayed.nodes[dec_id].confidence, 0.25)
        self.assertEqual(replayed.nodes[dec_id].metadata.get("upstream_refuted"), [obs_id])
        self.assertEqual(replayed.nodes[know_id].status, MemoryStatus.STALE)

        # 3. Conflict synthesis equivalence
        self.assertIn(synth_node.id, replayed.nodes)
        self.assertIn("synthesized", replayed.nodes[synth_node.id].tags)
        self.assertEqual(replayed.nodes[synth_node.id].metadata.get("synthesized_from"), [node_a, node_b])
        self.assertNotIn("conflicted", replayed.nodes[node_a].metadata)
        self.assertEqual(replayed.nodes[node_a].metadata.get("conditions"), ["financial transaction data"])
        self.assertNotIn("conflicted", replayed.nodes[node_b].metadata)
        self.assertEqual(replayed.nodes[node_b].metadata.get("conditions"), ["temporary metric data"])

    def test_metacognition_symbol_drift_and_upstream_refuted_routing(self) -> None:
        # 1. Node with upstream_refuted
        node_refuted = MemoryNode(
            id="mem_dependent_001",
            title="Redis Connection Pooling",
            content="Pool max connections set to 50 based on old metric.",
            category="decision",
            confidence=0.8,
            metadata={"upstream_refuted": ["mem_base_metric"]},
            status=MemoryStatus.ACTIVE,
        )

        res = classify("Redis Connection Pooling for cache", [node_refuted])
        uncertain_ids = [u["node_id"] for u in res["uncertain"]]
        self.assertIn("mem_dependent_001", uncertain_ids)

        unc_entry = next(u for u in res["uncertain"] if u["node_id"] == "mem_dependent_001")
        self.assertIn("upstream_refuted", unc_entry["reasons"])
        self.assertEqual(unc_entry["route"], "resolve")
        self.assertEqual(unc_entry["target"], "mem_dependent_001")

        # 2. route_uncertain direct routing tests
        # A. upstream_refuted in reasons routes to resolve
        r_route, r_target = route_uncertain(node_refuted, ["upstream_refuted"])
        self.assertEqual(r_route, "resolve")
        self.assertEqual(r_target, "mem_dependent_001")

        # B. symbol_hash_mismatch routes to code with symbol name
        sym_node = MemoryNode(
            id="mem_sym_002",
            title="Auth Verify Method",
            content="Use verify_token method from auth service.",
            category="decision",
            confidence=0.9,
            metadata={"symbols": ["src/auth.py::verify_token"]},
            status=MemoryStatus.ACTIVE,
        )
        route_sym, target_sym = route_uncertain(sym_node, ["symbol_hash_mismatch: src/auth.py::verify_token"])
        self.assertEqual(route_sym, "code")
        self.assertEqual(target_sym, "src/auth.py::verify_token")

        # C. missing_symbol routes to code with symbol name
        route_miss, target_miss = route_uncertain(sym_node, ["missing_symbol: src/auth.py::verify_token"])
        self.assertEqual(route_miss, "code")
        self.assertEqual(target_miss, "src/auth.py::verify_token")

        # D. symbol_hashes dictionary fallback in route_uncertain
        sym_hash_node = MemoryNode(
            id="mem_sym_hash_003",
            title="Hash Token Spec",
            content="Specification for token hashing function",
            category="pattern",
            confidence=0.7,
            metadata={"symbol_hashes": {"src/auth.py::generate_hash": "a1b2c3d4"}},
            status=MemoryStatus.ACTIVE,
        )
        route_sh, target_sh = route_uncertain(sym_hash_node, ["low_confidence"])
        self.assertEqual(route_sh, "code")
        self.assertEqual(target_sh, "src/auth.py::generate_hash")

    def test_context_rendering_relation_and_branch_aware(self) -> None:
        mind = FibMind()

        # 1. Contradiction expansion hit
        node_seed_id = mind.append(
            "decision",
            "Primary Authentication Strategy",
            "Use session cookies for web dashboard authentication.",
        )
        node_contradict_id = mind.append(
            "decision",
            "Stateless Token Verification",
            "Use JWT bearer tokens exclusively without server sessions.",
            auto_link=False,
        )
        mind.link_nodes(node_seed_id, node_contradict_id, RelationType.CONTRADICTS, weight=1.0)

        ctx = build_context(mind, "Primary Authentication Strategy", depth=1)
        # Check that expansion hit has via_relation
        expand_hits = [h for h in ctx.hits if h.node.id == node_contradict_id]
        self.assertEqual(len(expand_hits), 1)
        self.assertEqual(expand_hits[0].via_relation, "contradicts")
        # Text rendering contains [CONTRADICTS]
        self.assertIn("[CONTRADICTS] Stateless Token Verification", ctx.text)

        # 2. Synthesized conflict with active branch selection
        node_a_id = mind.append(
            "decision",
            "Postgres Connection Pooling",
            "Use PgBouncer for external connection pooling.",
        )
        node_b_id = mind.append(
            "decision",
            "Direct Database Connections",
            "Connect directly to Postgres with internal thread pool.",
        )
        synth_node = mind.synthesize_conflict(
            node_a_id,
            node_b_id,
            title="Postgres Connection Architecture",
            content="Strategy for handling database client connections.",
            conditions_a=["high serverless scale", "thousands of lambdas"],
            conditions_b=["monolithic service", "dedicated low latency instances"],
            source="db_infra_team",
        )
        self.assertIn("synthesized", synth_node.tags)

        # Context build with goal matching conditions_a
        goal_a = "Postgres Connection Architecture for thousands of lambdas under high serverless scale."
        ctx_a = build_context(mind, goal=goal_a, depth=0)
        self.assertIn(
            "ActiveBranch (high serverless scale, thousands of lambdas): Postgres Connection Pooling",
            ctx_a.text,
        )

        # Context build with goal matching conditions_b
        goal_b = "Postgres Connection Architecture for monolithic service on dedicated low latency instances."
        ctx_b = build_context(mind, goal=goal_b, depth=0)
        self.assertIn(
            "ActiveBranch (monolithic service, dedicated low latency instances): Direct Database Connections",
            ctx_b.text,
        )

    def test_fibonacci_layer_epistemic_shielding_and_replay(self) -> None:
        mind = FibMind()

        # Create 22 verified nodes with confidence > 0.6
        v_ids: list[str] = []
        for i in range(22):
            nid = mind.append("fact", f"Verified Invariant {i}", f"Proven invariant fact {i}", compress=False)
            mind.record_outcome(nid, Verdict.CONFIRMED, source="test", trust_ring="ring0")
            mind.record_outcome(nid, Verdict.CONFIRMED, source="test", trust_ring="ring0")
            mind.record_outcome(nid, Verdict.CONFIRMED, source="test", trust_ring="ring0")
            self.assertGreater(mind.nodes[nid].confidence, RING3_CONFIDENCE_CAP)
            v_ids.append(nid)

        # Append 1 unverified node (confidence == 0.0)
        u_id = mind.append("fact", "Unverified Fact", "Unverified ephemeral content", compress=False)
        self.assertEqual(mind.nodes[u_id].confidence, 0.0)

        # Total nodes in raw layer is 23 (> 21 raw capacity)
        self.assertEqual(len(mind._nodes_in_category_layer("fact", "raw")), 23)

        # Trigger overflow compression
        mind.compress_overflow("fact")

        # Invariant check:
        # 1. Unverified node was folded into a COMPRESSED summary node
        self.assertIsNotNone(mind.nodes[u_id].folded_into)
        summary_id = mind.nodes[u_id].folded_into
        self.assertIsNotNone(summary_id)
        self.assertEqual(mind.nodes[summary_id].node_type, NodeType.COMPRESSED)

        # 2. Verified nodes in overflow were shielded and promoted intact to 'compressed' layer
        promoted_v = [nid for nid in v_ids if mind.nodes[nid].layer == "compressed"]
        self.assertGreater(len(promoted_v), 0)
        for p_id in promoted_v:
            p_node = mind.nodes[p_id]
            self.assertEqual(p_node.node_type, NodeType.CONCEPT)
            self.assertIsNone(p_node.folded_into)

        # 3. Full replay isomorphism
        rebuilt = FibMind.rebuild_from_log(mind.events)
        self.assertEqual(rebuilt.nodes[u_id].folded_into, summary_id)
        self.assertEqual(rebuilt.nodes[summary_id].node_type, NodeType.COMPRESSED)
        for p_id in promoted_v:
            rp_node = rebuilt.nodes[p_id]
            self.assertEqual(rp_node.layer, "compressed")
            self.assertEqual(rp_node.node_type, NodeType.CONCEPT)
            self.assertIsNone(rp_node.folded_into)

    def test_causal_unfolding_on_summary_refutation_and_unfold_node(self) -> None:
        # Part A: Refutation of 1 folded source node triggers unfolding
        mind = FibMind()
        s1 = mind.append("test", "Alpha Service Spec", "Alpha unique service specification details")
        s2 = mind.append("test", "Beta Service Spec", "Beta unique service specification details")
        s3 = mind.append("test", "Gamma Service Spec", "Gamma unique service specification details")

        summary_id = mind._compress_nodes("test", "raw", "compressed", [mind.nodes[s1], mind.nodes[s2], mind.nodes[s3]])
        summary = mind.nodes[summary_id]
        self.assertEqual(summary.status, MemoryStatus.ACTIVE)
        self.assertEqual(mind.nodes[s1].folded_into, summary_id)
        self.assertEqual(mind.nodes[s2].folded_into, summary_id)
        self.assertEqual(mind.nodes[s3].folded_into, summary_id)

        # While folded, searching for unique Beta terms returns the summary (folded nodes hidden)
        search_hits_before = [h.node.id for h in mind.search("Beta unique service specification")]
        self.assertIn(summary_id, search_hits_before)
        self.assertNotIn(s2, search_hits_before)

        # Refute s1
        mind.record_outcome(s1, Verdict.REFUTED, source="integration_test")

        # Summary becomes STALE
        self.assertEqual(mind.nodes[summary_id].status, MemoryStatus.STALE)
        self.assertEqual(mind.nodes[s1].status, MemoryStatus.REFUTED)

        # Healthy sources s2 and s3 are unfolded (folded_into is None)
        self.assertIsNone(mind.nodes[s2].folded_into)
        self.assertIsNone(mind.nodes[s3].folded_into)

        # Now searching for Beta terms returns s2 directly!
        search_hits_after = [h.node.id for h in mind.search("Beta unique service specification")]
        self.assertIn(s2, search_hits_after)

        # Replay isomorphism
        rebuilt = FibMind.rebuild_from_log(mind.events)
        self.assertEqual(rebuilt.nodes[summary_id].status, MemoryStatus.STALE)
        self.assertIsNone(rebuilt.nodes[s2].folded_into)
        self.assertIsNone(rebuilt.nodes[s3].folded_into)
        rebuilt_hits = [h.node.id for h in rebuilt.search("Beta unique service specification")]
        self.assertIn(s2, rebuilt_hits)

        # Part B: Manual unfold_node API
        mind2 = FibMind()
        m1 = mind2.append("manual", "Manual Spec 1", "Manual content 1")
        m2 = mind2.append("manual", "Manual Spec 2", "Manual content 2")
        manual_summary_id = mind2._compress_nodes("manual", "raw", "compressed", [mind2.nodes[m1], mind2.nodes[m2]])

        unfolded_ids = mind2.unfold_node(manual_summary_id, "manual decommissioning")
        self.assertEqual(sorted(unfolded_ids), sorted([m1, m2]))
        self.assertEqual(mind2.nodes[manual_summary_id].status, MemoryStatus.STALE)
        self.assertIsNone(mind2.nodes[m1].folded_into)
        self.assertIsNone(mind2.nodes[m2].folded_into)

        rebuilt2 = FibMind.rebuild_from_log(mind2.events)
        self.assertEqual(rebuilt2.nodes[manual_summary_id].status, MemoryStatus.STALE)
        self.assertIsNone(rebuilt2.nodes[m1].folded_into)
        self.assertIsNone(rebuilt2.nodes[m2].folded_into)

    def test_tuning_ring_evidence_profile_and_tri_gate(self) -> None:
        mind = FibMind()

        # Node 1: Ring 0 confirmed
        n1 = mind.append("decision", "Database Index", "Index created on user_id")
        mind.record_outcome(n1, Verdict.CONFIRMED, source="pytest", trust_ring="ring0")

        # Node 2: Ring 3 confirmed
        n2 = mind.append("decision", "Log Format", "Use JSON formatting")
        mind.record_outcome(n2, Verdict.CONFIRMED, source="llm_eval", trust_ring="ring3")

        # Profile evidence
        profile = profile_evidence(mind)
        p_dict = profile.to_dict()
        self.assertEqual(profile.confirmed, 2)
        self.assertEqual(profile.ring0_confirmed, 1)
        self.assertEqual(profile.ring3_confirmed, 1)
        self.assertEqual(p_dict["ring0_confirmed"], 1)
        self.assertEqual(p_dict["ring3_confirmed"], 1)

        # Test Tri-Gate with check_gaps=True, check_kernel=True
        w1 = RankingWeights()
        w2 = RankingWeights()

        def mock_evaluate(weights: RankingWeights) -> dict[str, Any]:
            return {
                "recall_at_k": 0.95,
                "mrr": 0.90,
                "stale_pollution_rate": 0.0,
                "forbidden_case_rate": 0.0,
                "no_relevant_accuracy": 1.0,
            }

        gate_res = gate(w1, w2, evaluate=mock_evaluate, check_gaps=True, check_kernel=True)
        self.assertTrue(gate_res.passed)
        self.assertEqual(gate_res.reasons, [])
        self.assertIn("gaps", gate_res.candidate)
        self.assertIn("kernel", gate_res.candidate)
        self.assertGreaterEqual(gate_res.candidate["gaps"]["verdict_accuracy"], 0.90)
        self.assertEqual(gate_res.candidate["gaps"]["false_known_rate"], 0.0)
        for score in gate_res.candidate["kernel"].values():
            self.assertEqual(score, 1.0)

    def test_background_epistemic_tiers_and_active_branch(self) -> None:
        mind = FibMind()

        # 1. Ring 0 verified node (> 0.6)
        n_ring0 = mind.append("profile", "Database Connection Pool", "Use HikariCP pool", owner="abbila")
        mind.record_outcome(n_ring0, Verdict.CONFIRMED, source="test", trust_ring="ring0")
        mind.record_outcome(n_ring0, Verdict.CONFIRMED, source="test", trust_ring="ring0")
        mind.record_outcome(n_ring0, Verdict.CONFIRMED, source="test", trust_ring="ring0")
        self.assertGreater(mind.nodes[n_ring0].confidence, RING3_CONFIDENCE_CAP)

        # 2. Confirmed node (<= 0.6)
        n_conf = mind.append("profile", "Log Retention Policy", "Retain logs for 30 days", owner="abbila")
        mind.record_outcome(n_conf, Verdict.CONFIRMED, source="self", trust_ring="ring3")
        self.assertEqual(mind.nodes[n_conf].confidence, 0.25)

        # 3. Tainted node (upstream_refuted)
        n_tainted = mind.append(
            "profile",
            "Cache Size Allocation",
            "Allocate 4GB memory for cache",
            owner="abbila",
            metadata={"upstream_refuted": ["old_metric_node"]},
        )

        # 4. Unchecked node (confidence == 0.0)
        n_uncheck = mind.append("profile", "Metrics Exporter", "Export Prometheus metrics", owner="abbila")
        self.assertEqual(mind.nodes[n_uncheck].confidence, 0.0)

        # Build background
        bg = build_background(mind, owner="abbila", top_k=10, profile_top_k=10, budget_tokens=2000)
        tier_map = {m["node_id"]: m["epistemic_tier"] for m in bg["memories"]}
        self.assertEqual(tier_map.get(n_ring0), "ring0_verified")
        self.assertEqual(tier_map.get(n_conf), "confirmed")
        self.assertEqual(tier_map.get(n_tainted), "tainted")
        self.assertEqual(tier_map.get(n_uncheck), "unchecked")

        # 5. Synthesized node with dialectical branches
        node_a = mind.append("decision", "Fast API Framework", "Use FastAPI for REST API", owner="abbila")
        node_b = mind.append("decision", "Flask Framework", "Use Flask for REST API", owner="abbila")
        synth_node = mind.synthesize_conflict(
            node_a,
            node_b,
            title="REST API Framework Selection",
            content="Decision on web framework.",
            conditions_a=["async high throughput"],
            conditions_b=["synchronous simple scripts"],
            source="architecture_team",
        )
        self.assertIn("synthesized", synth_node.tags)

        # Build background with goal matching branch A
        goal_a = "REST API Framework Selection for async high throughput"
        bg_a = build_background(mind, goal=goal_a, owner="abbila", top_k=10)
        self.assertIn("ActiveBranch (async high throughput): Fast API Framework.", bg_a["text"])

        # Build background with goal matching branch B
        goal_b = "REST API Framework Selection for synchronous simple scripts"
        bg_b = build_background(mind, goal=goal_b, owner="abbila", top_k=10)
        self.assertIn("ActiveBranch (synchronous simple scripts): Flask Framework.", bg_b["text"])

    def test_retrieval_epistemic_topology_boost_and_taint_decay(self) -> None:
        mind = FibMind()

        # 1. Ordinary node
        n_ord = mind.append("decision", "Postgres Connection Pool", "Manage database connections", compress=False)

        # 2. Synthesized node (same query terms)
        n_synth = mind.append(
            "decision",
            "Synthesized Postgres Connection Pool",
            "Manage database connections",
            tags=["synthesized"],
            compress=False,
        )

        # 3. Concept node with high confidence
        n_concept = mind.append(
            "decision",
            "Concept Postgres Connection Pool",
            "Manage database connections",
            compress=False,
        )
        mind.nodes[n_concept].node_type = NodeType.CONCEPT
        mind.record_outcome(n_concept, Verdict.CONFIRMED, source="test", trust_ring="ring0")
        mind.record_outcome(n_concept, Verdict.CONFIRMED, source="test", trust_ring="ring0")
        mind.record_outcome(n_concept, Verdict.CONFIRMED, source="test", trust_ring="ring0")
        self.assertGreater(mind.nodes[n_concept].confidence, 0.6)

        # 4. Tainted node with drift_detected
        n_drift = mind.append(
            "decision",
            "Drifted Postgres Connection Pool",
            "Manage database connections",
            compress=False,
            metadata={"drift_detected": True},
        )

        # 5. Tainted node with upstream_refuted
        n_taint = mind.append(
            "decision",
            "Refuted Postgres Connection Pool",
            "Manage database connections",
            compress=False,
            metadata={"upstream_refuted": ["old_node"]},
        )

        # Verify epistemic_topology_boost directly
        self.assertEqual(epistemic_topology_boost(mind.nodes[n_synth]), 0.05)
        self.assertEqual(epistemic_topology_boost(mind.nodes[n_concept]), 0.05)
        self.assertEqual(epistemic_topology_boost(mind.nodes[n_ord]), 0.0)
        self.assertEqual(epistemic_topology_boost(mind.nodes[n_drift]), -0.10)
        self.assertEqual(epistemic_topology_boost(mind.nodes[n_taint]), -0.10)

        # Verify in retrieve()
        hits, _ = retrieve(mind, "Postgres Connection Pool", top_k=5)
        hit_map = {h.node.id: h for h in hits}

        # Synthesized and concept nodes received +0.05 boost and "topology" in sources
        self.assertEqual(hit_map[n_synth].topology_boost, 0.05)
        self.assertIn("topology", hit_map[n_synth].sources)

        self.assertEqual(hit_map[n_concept].topology_boost, 0.05)
        self.assertIn("topology", hit_map[n_concept].sources)

        # Ordinary node has 0.0 boost
        self.assertEqual(hit_map[n_ord].topology_boost, 0.0)
        self.assertNotIn("topology", hit_map[n_ord].sources)

        # Drift and tainted nodes received -0.10 penalty
        self.assertEqual(hit_map[n_drift].topology_boost, -0.10)
        self.assertEqual(hit_map[n_taint].topology_boost, -0.10)
        self.assertNotIn("topology", hit_map[n_drift].sources)
        self.assertNotIn("topology", hit_map[n_taint].sources)

        # Ranking order: synthesized/concept above ordinary above tainted
        self.assertGreater(hit_map[n_synth].score, hit_map[n_ord].score)
        self.assertGreater(hit_map[n_ord].score, hit_map[n_drift].score)
        self.assertGreater(hit_map[n_ord].score, hit_map[n_taint].score)

    def test_propose_conflict_synthesis_extracts_conditions(self) -> None:
        node_a = MemoryNode(
            title="Enable Connection Pooling",
            content="Use connection pooling in prod.\nWhen high load is expected\nrequires db pool size >= 20",
            category="decision",
            tags={"prod", "shared-tag", "synthesized"},
            metadata={
                "conditions": ["env == production"],
                "files": ["src/prod_config.py", "src/common.py"],
            },
        )
        node_b = MemoryNode(
            title="Disable Connection Pooling",
            content="Disable connection pooling in dev.\nIf running locally in dev mode\nunder low memory footprint",
            category="decision",
            tags={"dev", "shared-tag"},
            metadata={
                "conditions": ["env == dev"],
                "files": ["src/dev_config.py", "src/common.py"],
            },
        )

        synthesis = propose_conflict_synthesis(node_a, node_b)

        self.assertEqual(synthesis["node_id_a"], node_a.id)
        self.assertEqual(synthesis["node_id_b"], node_b.id)
        self.assertEqual(synthesis["title_a"], node_a.title)
        self.assertEqual(synthesis["title_b"], node_b.title)

        # Conditions A should contain exclusive tags (prod, but not synthesized/shared-tag), metadata conditions, exclusive files, conditional lines
        conds_a = synthesis["conditions_a"]
        self.assertIn("prod", conds_a)
        self.assertNotIn("synthesized", conds_a)
        self.assertNotIn("shared-tag", conds_a)
        self.assertIn("env == production", conds_a)
        self.assertIn("src/prod_config.py", conds_a)
        self.assertNotIn("src/common.py", conds_a)
        self.assertIn("When high load is expected", conds_a)
        self.assertIn("requires db pool size >= 20", conds_a)

        # Conditions B should contain exclusive tags (dev), metadata conditions, exclusive files, conditional lines
        conds_b = synthesis["conditions_b"]
        self.assertIn("dev", conds_b)
        self.assertNotIn("shared-tag", conds_b)
        self.assertIn("env == dev", conds_b)
        self.assertIn("src/dev_config.py", conds_b)
        self.assertNotIn("src/common.py", conds_b)
        self.assertIn("If running locally in dev mode", conds_b)
        self.assertIn("under low memory footprint", conds_b)

        # Suggested title and content
        self.assertEqual(
            synthesis["suggested_title"],
            f"Synthesized: {node_a.title} vs {node_b.title}",
        )
        self.assertTrue(synthesis["suggested_content"].startswith("When "))

        # Fallback test: when nodes have no distinguishing tags/files/conditions
        plain_a = MemoryNode(title="Alpha", content="Plain content alpha", category="decision")
        plain_b = MemoryNode(title="Beta", content="Plain content beta", category="decision")
        fallback_synth = propose_conflict_synthesis(plain_a, plain_b)
        self.assertEqual(fallback_synth["conditions_a"], ["context:Alpha"])
        self.assertEqual(fallback_synth["conditions_b"], ["context:Beta"])

    def test_calibrate_trust_floor_and_metacognitive_classify(self) -> None:
        # 1. Test calibrate_trust_floor calculations
        self.assertEqual(calibrate_trust_floor(0.0), 0.5)
        self.assertEqual(calibrate_trust_floor(0.2), 0.6)
        self.assertEqual(calibrate_trust_floor(0.4), 0.7)
        self.assertEqual(calibrate_trust_floor(0.8), 0.75)  # capped at 0.75
        self.assertEqual(calibrate_trust_floor(1.0), 0.75)

        # 2. Test classify with dynamic calibration
        mind = FibMind()
        node_id = mind.append(
            "decision",
            "Postgres Connection Pool",
            "Configure postgres pool size to 20 for database connections",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
        )
        # Give node confidence 0.6
        mind.record_outcome(node_id, Verdict.CONFIRMED, source="test", trust_ring="ring3")
        mind.record_outcome(node_id, Verdict.CONFIRMED, source="test", trust_ring="ring3")
        mind.record_outcome(node_id, Verdict.CONFIRMED, source="test", trust_ring="ring3")
        self.assertEqual(mind.nodes[node_id].confidence, 0.6)

        node = mind.nodes[node_id]
        goal = "Postgres Connection Pool configuration"

        # Baseline: p_refuted_given_known=0.0 -> trust_floor = 0.5
        # confidence (0.6) >= trust_floor (0.5) -> classified as known
        res_baseline = classify(goal, [node], p_refuted_given_known=0.0)
        self.assertEqual(res_baseline["trust_floor"], 0.5)
        self.assertEqual(len(res_baseline["known"]), 1)
        self.assertEqual(len(res_baseline["uncertain"]), 0)

        # Calibrated: p_refuted_given_known=0.4 -> trust_floor = 0.7
        # confidence (0.6) < trust_floor (0.7) -> classified as uncertain (low_confidence)
        res_calibrated = classify(goal, [node], p_refuted_given_known=0.4)
        self.assertEqual(res_calibrated["trust_floor"], 0.7)
        self.assertEqual(len(res_calibrated["known"]), 0)
        self.assertEqual(len(res_calibrated["uncertain"]), 1)
        self.assertIn("low_confidence", res_calibrated["uncertain"][0]["reasons"])

        # Explicit trust_floor parameter overrides calibration
        res_override = classify(goal, [node], trust_floor=0.55)
        self.assertEqual(res_override["trust_floor"], 0.55)
        self.assertEqual(len(res_override["known"]), 1)

    def test_compress_overflow_fidelity_guard_repairs_lossy_summarizer(self) -> None:
        policy = FibonacciLayerPolicy(
            capacities={"raw": 3, "compressed": 5},
            order=("raw", "compressed"),
            low_watermarks={"raw": 1},
        )
        # ExcerptSummarizer with excerpt_chars=25 will aggressively truncate content,
        # dropping the critical file anchors and negative constraints.
        mind = FibMind(
            layer_policy=policy,
            summarizer=ExcerptSummarizer(excerpt_chars=25),
        )

        for i in range(4):
            mind.append(
                "security",
                f"Rule {i}",
                f"Do not expose root password in src/auth/crypto_store_{i}.py\nAvoid using MD5_{i} algorithm",
                scope=MemoryScope.PERSONAL,
                owner="abbila",
                metadata={
                    "files": [f"src/auth/crypto_store_{i}.py"],
                    "exceptions": [f"Avoid using MD5_{i} algorithm"],
                },
            )

        compressed_nodes = [n for n in mind.nodes.values() if n.layer == "compressed"]
        self.assertEqual(len(compressed_nodes), 1)
        summary_node = compressed_nodes[0]

        # Fidelity guard must have repaired the distillation
        self.assertTrue(summary_node.metadata.get("fidelity_repaired"))
        distillation_meta = summary_node.metadata.get("distillation", {})
        self.assertTrue(distillation_meta.get("fidelity_repaired"))
        self.assertTrue(len(distillation_meta.get("repaired_items", [])) > 0)

        # Both preserved anchors and exception lines must be present in content
        self.assertIn("PreservedAnchors:", summary_node.content)
        self.assertIn("Exceptions:", summary_node.content)
        for i in range(3):  # First 3 nodes were compressed
            self.assertIn(f"src/auth/crypto_store_{i}.py", summary_node.content)
            self.assertIn(f"Avoid using MD5_{i} algorithm", summary_node.content)

        # Post-repair audit: verify_distillation_fidelity must now pass with 0 missing items
        source_nodes = [mind.nodes[sid] for sid in summary_node.metadata["source_node_ids"]]
        passed, missing = verify_distillation_fidelity(source_nodes, summary_node.content)
        self.assertTrue(passed)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
