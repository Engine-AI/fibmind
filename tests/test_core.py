import tempfile
import unittest
from pathlib import Path

from fibmind import (
    DEFAULT_LAYER_POLICY,
    EdgeDirection,
    FibMind,
    JsonStore,
    MemoryCandidate,
    MemoryScope,
    MemoryService,
    MemoryStatus,
    NodeType,
    RelationType,
    TraversalDirection,
    Verdict,
    fib_capacities,
    fibonacci_numbers,
)


class FibMindTests(unittest.TestCase):
    def test_fibonacci_numbers_respect_value_limit(self) -> None:
        self.assertEqual(fibonacci_numbers(0), [])
        self.assertEqual(fibonacci_numbers(1), [1])
        self.assertEqual(fibonacci_numbers(50), [1, 2, 3, 5, 8, 13, 21, 34])
        self.assertEqual(fibonacci_numbers(2, include_duplicate_one=True), [1, 1, 2])

    def test_default_policy_is_generated_from_fibonacci_capacities(self) -> None:
        layers = ("raw", "compressed", "summary", "long_term")

        self.assertEqual(fib_capacities(layers, start=21), DEFAULT_LAYER_POLICY.capacities)
        self.assertEqual(DEFAULT_LAYER_POLICY.capacities["raw"], 21)
        self.assertEqual(DEFAULT_LAYER_POLICY.capacities["long_term"], 89)

    def test_append_creates_tree_and_node(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Bug", "A login bug")

        self.assertIn(node_id, memory.nodes)
        self.assertEqual(len(memory.trees), 1)
        self.assertEqual(memory.nodes[node_id].category, "code")

    def test_link_and_search_from_node(self) -> None:
        memory = FibMind()
        first = memory.append("code", "Bug", "A login bug")
        second = memory.append("requirement", "Requirement", "Refresh token")
        memory.link_nodes(first, second, RelationType.RELATED_TO)

        hits = memory.search_from(first, depth=1)
        hit_ids = {hit.node.id for hit in hits}

        self.assertIn(first, hit_ids)
        self.assertIn(second, hit_ids)

    def test_directed_edges_only_traverse_backward_when_requested(self) -> None:
        memory = FibMind()
        first = memory.append("code", "Bug", "A login bug")
        second = memory.append("requirement", "Requirement", "Refresh token")
        memory.link_nodes(first, second, RelationType.RELATED_TO)

        outgoing_hits = memory.search_from(second, depth=1, direction=TraversalDirection.OUT)
        outgoing_hit_ids = {hit.node.id for hit in outgoing_hits}

        self.assertIn(second, outgoing_hit_ids)
        self.assertNotIn(first, outgoing_hit_ids)

        both_hits = memory.search_from(second, depth=1)
        related = next(hit for hit in both_hits if hit.node.id == first)
        self.assertEqual(related.via_direction, TraversalDirection.IN)

    def test_bidirectional_edges_traverse_both_directions(self) -> None:
        memory = FibMind()
        first = memory.append("code", "Bug", "A login bug")
        second = memory.append("requirement", "Requirement", "Refresh token")
        memory.link_nodes(first, second, RelationType.RELATED_TO, direction=EdgeDirection.BIDIRECTIONAL)

        reverse_hits = memory.search_from(second, depth=1, direction=TraversalDirection.OUT)
        reverse_hit_ids = {hit.node.id for hit in reverse_hits}

        self.assertIn(first, reverse_hit_ids)

    def test_search_reinforcement_is_explicit(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Bug", "A login bug")

        memory.search_from(node_id, depth=0)
        self.assertEqual(memory.nodes[node_id].access_count, 0)
        self.assertEqual(memory.nodes[node_id].familiarity, 0.0)

        memory.search_from(node_id, depth=0, reinforce=True)
        self.assertEqual(memory.nodes[node_id].access_count, 1)
        self.assertGreater(memory.nodes[node_id].familiarity, 0.0)

    def test_recall_never_becomes_evidence(self) -> None:
        """Reading a memory back must not make it look more correct.

        This is the loop the familiarity/confidence split exists to break: if
        recall fed the trust signal, a wrong memory that gets looked up often
        would climb the rankings on its own.
        """
        memory = FibMind()
        node_id = memory.append("code", "Wrong claim", "sqrt(-1) == -1")

        for _ in range(60):
            memory.search_from(node_id, depth=0, reinforce=True)

        node = memory.nodes[node_id]
        self.assertEqual(node.familiarity, 1.0)
        self.assertEqual(node.confidence, 0.0)
        self.assertIsNone(node.confidence_source)

    def test_confidence_moves_only_on_recorded_evidence(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Fix", "clear the cache before retrying")

        memory.record_outcome(node_id, Verdict.CONFIRMED, source="pytest tests/test_retry.py")
        self.assertGreater(memory.nodes[node_id].confidence, 0.0)
        self.assertEqual(memory.nodes[node_id].confidence_source, "pytest tests/test_retry.py")

        memory.record_outcome(node_id, Verdict.REFUTED, source="reverted in a1b2c3d")
        self.assertEqual(memory.nodes[node_id].confidence, 0.0)

        with self.assertRaises(ValueError):
            memory.record_outcome(node_id, Verdict.CONFIRMED, source="  ")

    def test_promote_node_keeps_the_memory_retrievable(self) -> None:
        """Marking a memory important must not hide it.

        Ranking skips ROOT nodes because they are structural anchors, so
        promoting a real memory to ROOT used to remove it from search entirely —
        the exact opposite of what promotion means.
        """
        memory = FibMind()
        node_id = memory.append("code", "Login problem", "Auth issue")
        self.assertTrue(memory.search("login auth"))

        memory.promote_node(node_id)

        self.assertEqual(memory.nodes[node_id].node_type, NodeType.CONCEPT)
        self.assertIn(node_id, {hit.node.id for hit in memory.search("login auth")})
        with self.assertRaises(ValueError):
            memory.promote_node(node_id, target_type=NodeType.ROOT)

    def test_expand_node_to_tree_gives_a_root_without_hiding_it(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Login problem", "Auth issue")

        tree_id = memory.expand_node_to_tree(node_id)

        self.assertIn(tree_id, memory.trees)
        self.assertEqual(memory.trees[tree_id].root_node_id, node_id)
        self.assertEqual(memory.nodes[node_id].node_type, NodeType.CONCEPT)
        self.assertIn(node_id, {hit.node.id for hit in memory.search("login auth")})

    def test_append_links_related_memories(self) -> None:
        """A one-hop expansion has to reach something other than the tree root.

        Only tree_child edges used to be created, leaving a star: every walk from
        a node hit the root, which context building filters out, so expansion
        returned nothing.
        """
        memory = FibMind()
        first = memory.append("code", "Login 401", "token expiry causes 401 on login")
        second = memory.append("code", "Token refresh", "refresh rotates the token on 401")

        neighbours = {
            hit.node.id
            for hit in memory.search_from(second, depth=1)
            if hit.node.node_type != NodeType.ROOT and hit.node.id != second
        }
        self.assertIn(first, neighbours)

    def test_compress_overflow_creates_summary_node(self) -> None:
        memory = FibMind()
        for index in range(23):
            memory.append("dialogue", f"Message {index}", f"content {index}")

        compressed = [
            node for node in memory.nodes.values() if node.category == "dialogue" and node.layer == "compressed"
        ]
        raw = [
            node
            for node in memory.nodes.values()
            if node.category == "dialogue" and node.layer == "raw" and node.folded_into is None
        ]

        self.assertGreaterEqual(len(compressed), 1)
        self.assertLessEqual(len(raw), 21)
        self.assertGreaterEqual(sum(node.memory_weight for node in compressed), 2)

    def test_folded_nodes_are_kept_but_hidden(self) -> None:
        """Folding must not destroy the original text.

        The summariser is a placeholder; keeping the sources means a better one
        can redo the work later. Folded nodes stay in the store, out of search.
        """
        memory = FibMind()
        for index in range(22):
            memory.append("dialogue", f"Message {index}", f"content {index}")

        folded = [node for node in memory.nodes.values() if node.folded_into is not None]
        self.assertTrue(folded)
        for node in folded:
            self.assertNotEqual(node.content, "")
            self.assertIn(node.folded_into, memory.nodes)

        visible = {hit.node.id for hit in memory.search("content", top_k=100)}
        self.assertTrue(visible.isdisjoint({node.id for node in folded}))

    def test_compress_overflow_compacts_to_fibonacci_low_watermark(self) -> None:
        memory = FibMind()
        for index in range(22):
            memory.append("dialogue", f"Message {index}", f"content {index}")

        raw = [
            node
            for node in memory.nodes.values()
            if node.category == "dialogue" and node.layer == "raw" and node.folded_into is None
        ]
        compressed = [
            node for node in memory.nodes.values() if node.category == "dialogue" and node.layer == "compressed"
        ]

        self.assertEqual(sum(node.memory_weight for node in raw), 13)
        self.assertEqual(len(compressed), 1)
        self.assertEqual(compressed[0].memory_weight, 9)
        self.assertEqual(len(compressed[0].metadata["source_node_ids"]), 9)

    def test_batch_compression_bounds_graph_growth(self) -> None:
        memory = FibMind()
        for index in range(200):
            memory.append("bulk", f"Node {index}", "payload")

        for layer in memory.layer_policy.order:
            active_weight = sum(
                node.memory_weight
                for node in memory.nodes.values()
                if node.category == "bulk" and node.layer == layer and node.folded_into is None
            )
            self.assertLessEqual(active_weight, memory.layer_policy.capacity_for(layer))

    def test_json_store_round_trip(self) -> None:
        memory = FibMind()
        first = memory.append("code", "Bug", "A login bug")
        second = memory.append("requirement", "Requirement", "Refresh token")
        memory.link_nodes(first, second, RelationType.RELATED_TO)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = JsonStore(path)
            store.save(memory)
            restored = store.load()

        self.assertEqual(set(memory.nodes), set(restored.nodes))
        self.assertEqual(set(memory.edges), set(restored.edges))
        self.assertEqual(set(memory.trees), set(restored.trees))

        restored_hits = restored.search_from(first, depth=1)
        self.assertIn(second, {hit.node.id for hit in restored_hits})

    def test_summary_of_cascade_refutation_and_replay(self) -> None:
        memory = FibMind()
        s1 = memory.append("test", "Fact 1", "Fact 1 content")
        s2 = memory.append("test", "Fact 2", "Fact 2 content")
        s3 = memory.append("test", "Fact 3", "Fact 3 content")

        # Create a compressed summary node summarizing s1, s2, s3
        summary_id = memory._compress_nodes(
            "test", "raw", "compressed", [memory.nodes[s1], memory.nodes[s2], memory.nodes[s3]]
        )
        summary_node = memory.nodes[summary_id]
        summary_node.confidence = 1.0

        # Refute s1
        memory.record_outcome(s1, Verdict.REFUTED, source="test_runner")
        self.assertEqual(memory.nodes[s1].status, MemoryStatus.REFUTED)

        # Summary node confidence should be scaled by active sources (2/3)
        self.assertEqual(summary_node.confidence, round(1.0 * (2 / 3), 4))
        self.assertIn("source refuted", summary_node.confidence_source)
        self.assertEqual(summary_node.status, MemoryStatus.ACTIVE)

        # Refute s2
        memory.record_outcome(s2, Verdict.REFUTED, source="test_runner")
        self.assertEqual(summary_node.confidence, round(0.6667 * (1 / 3), 4))
        self.assertEqual(summary_node.status, MemoryStatus.ACTIVE)

        # Refute s3 (all sources refuted now)
        memory.record_outcome(s3, Verdict.REFUTED, source="test_runner")
        self.assertEqual(summary_node.confidence, 0.0)
        self.assertEqual(summary_node.status, MemoryStatus.STALE)
        self.assertIn("all 3 summarized source memories were refuted", summary_node.status_reason)

        # Replay consistency: rebuild_from_log should reconstruct the exact same state!
        rebuilt = FibMind.rebuild_from_log(memory.events)
        rebuilt_summary = rebuilt.nodes[summary_id]
        self.assertEqual(rebuilt_summary.status, MemoryStatus.STALE)
        self.assertEqual(rebuilt_summary.confidence, 0.0)
        self.assertEqual(rebuilt_summary.status_reason, summary_node.status_reason)

    def test_detect_potential_conflicts_and_service_remember(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = MemoryService(Path(tmpdir) / "test.db")

            # 1. Store base active memory
            r1 = service.remember(
                MemoryCandidate(
                    category="decision",
                    title="SQLite concurrency pragma",
                    content="Always enable PRAGMA journal_mode=WAL on SQLite connections.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                ),
                metadata={"files": ["src/storage.py"]},
            )
            node1_id = r1["node_id"]
            self.assertNotIn("potential_conflicts", r1)

            # 2. Store conflicting memory with opposing negation
            r2 = service.remember(
                MemoryCandidate(
                    category="decision",
                    title="SQLite concurrency pragma",
                    content="Never enable PRAGMA journal_mode=WAL on SQLite connections.",
                    scope=MemoryScope.PERSONAL,
                    owner="abbila",
                ),
                metadata={"files": ["src/storage.py"]},
            )
            node2_id = r2["node_id"]
            self.assertIn("potential_conflicts", r2)
            self.assertIn(node1_id, r2["potential_conflicts"])

            # Verify persisted node metadata also contains potential_conflicts
            inspected = service.inspect(node2_id, identity={"owner": "abbila"})
            self.assertEqual(inspected["metadata"]["potential_conflicts"], [node1_id])


if __name__ == "__main__":
    unittest.main()
