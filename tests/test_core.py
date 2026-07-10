import tempfile
import unittest
from pathlib import Path

from fibmind import (
    DEFAULT_LAYER_POLICY,
    EdgeDirection,
    FibMind,
    JsonStore,
    NodeType,
    RelationType,
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

    def test_directed_edges_do_not_traverse_backward(self) -> None:
        memory = FibMind()
        first = memory.append("code", "Bug", "A login bug")
        second = memory.append("requirement", "Requirement", "Refresh token")
        memory.link_nodes(first, second, RelationType.RELATED_TO)

        reverse_hits = memory.search_from(second, depth=1)
        reverse_hit_ids = {hit.node.id for hit in reverse_hits}

        self.assertIn(second, reverse_hit_ids)
        self.assertNotIn(first, reverse_hit_ids)

    def test_bidirectional_edges_traverse_both_directions(self) -> None:
        memory = FibMind()
        first = memory.append("code", "Bug", "A login bug")
        second = memory.append("requirement", "Requirement", "Refresh token")
        memory.link_nodes(first, second, RelationType.RELATED_TO, direction=EdgeDirection.BIDIRECTIONAL)

        reverse_hits = memory.search_from(second, depth=1)
        reverse_hit_ids = {hit.node.id for hit in reverse_hits}

        self.assertIn(first, reverse_hit_ids)

    def test_search_reinforcement_is_explicit(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Bug", "A login bug")

        memory.search_from(node_id, depth=0)
        self.assertEqual(memory.nodes[node_id].access_count, 0)
        self.assertEqual(memory.nodes[node_id].importance, 0.0)

        memory.search_from(node_id, depth=0, reinforce=True)
        self.assertEqual(memory.nodes[node_id].access_count, 1)
        self.assertGreater(memory.nodes[node_id].importance, 0.0)

    def test_promote_node_expands_to_tree(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Login problem", "Auth issue")

        tree_id = memory.promote_node(node_id)

        self.assertIn(tree_id, memory.trees)
        self.assertEqual(memory.trees[tree_id].root_node_id, node_id)
        self.assertEqual(memory.nodes[node_id].node_type, NodeType.ROOT)

    def test_compress_overflow_creates_summary_node(self) -> None:
        memory = FibMind()
        for index in range(23):
            memory.append("dialogue", f"Message {index}", f"content {index}")

        compressed = [
            node
            for node in memory.nodes.values()
            if node.category == "dialogue" and node.layer == "compressed"
        ]
        raw = [
            node
            for node in memory.nodes.values()
            if node.category == "dialogue" and node.layer == "raw"
        ]

        self.assertGreaterEqual(len(compressed), 1)
        self.assertLessEqual(len(raw), 21)
        self.assertGreaterEqual(sum(node.memory_weight for node in compressed), 2)

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


if __name__ == "__main__":
    unittest.main()
