"""Tests for the event log: the store's source of truth.

The materialized node/edge tables are a cache. These tests pin the property that
makes that true — a forest can be thrown away and rebuilt from its log — plus the
one place the log is deliberately mutated: redaction on forget.
"""

import tempfile
import unittest
from pathlib import Path

from fibmind import (
    EventOp,
    FibMind,
    JsonStore,
    MemoryStatus,
    NodeType,
    SqliteStore,
    Verdict,
)
from fibmind.graph import TOMBSTONE


def _sample_forest() -> FibMind:
    memory = FibMind()
    first = memory.append("code", "Login 401", "token expiry causes 401 on login")
    second = memory.append("code", "Token refresh", "refresh rotates the token on 401")
    memory.record_outcome(first, Verdict.CONFIRMED, source="pytest tests/test_auth.py")
    memory.revise(second, content="refresh rotates the token before it expires")
    return memory


class EventLogTests(unittest.TestCase):
    def test_every_mutation_is_logged(self) -> None:
        memory = _sample_forest()
        ops = [event.op for event in memory.events]

        self.assertIn(EventOp.CREATE_TREE, ops)
        self.assertIn(EventOp.CREATE_NODE, ops)
        self.assertIn(EventOp.LINK, ops)
        self.assertIn(EventOp.OBSERVE, ops)
        self.assertIn(EventOp.REVISE, ops)
        self.assertEqual([event.seq for event in memory.events], list(range(1, len(ops) + 1)))

    def test_replay_reconstructs_the_forest(self) -> None:
        memory = _sample_forest()

        rebuilt = FibMind.rebuild_from_log(memory.events)

        self.assertEqual(set(memory.nodes), set(rebuilt.nodes))
        self.assertEqual(set(memory.edges), set(rebuilt.edges))
        self.assertEqual(memory.trees, rebuilt.trees)
        self.assertEqual(memory.category_roots, rebuilt.category_roots)
        for node_id, node in memory.nodes.items():
            replayed = rebuilt.nodes[node_id]
            self.assertEqual(node.title, replayed.title)
            self.assertEqual(node.content, replayed.content)
            self.assertEqual(node.confidence, replayed.confidence)
            self.assertEqual(node.scope, replayed.scope)

    def test_replay_drops_recall_counters(self) -> None:
        """Familiarity describes usage, not knowledge, and is not replayed."""
        memory = FibMind()
        node_id = memory.append("code", "Bug", "a login bug")
        for _ in range(5):
            memory.search_from(node_id, depth=0, reinforce=True)
        self.assertGreater(memory.nodes[node_id].familiarity, 0.0)

        rebuilt = FibMind.rebuild_from_log(memory.events)

        self.assertEqual(rebuilt.nodes[node_id].familiarity, 0.0)
        self.assertEqual(rebuilt.nodes[node_id].access_count, 0)

    def test_folded_nodes_survive_replay(self) -> None:
        memory = FibMind()
        for index in range(22):
            memory.append("dialogue", f"Message {index}", f"content {index}")
        folded = {
            node.id: node.folded_into
            for node in memory.nodes.values()
            if node.folded_into is not None
        }
        self.assertTrue(folded)

        rebuilt = FibMind.rebuild_from_log(memory.events)

        for node_id, into in folded.items():
            self.assertEqual(rebuilt.nodes[node_id].folded_into, into)

    def test_promotion_survives_replay(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Login problem", "auth issue")
        memory.promote_node(node_id)

        rebuilt = FibMind.rebuild_from_log(memory.events)

        self.assertEqual(rebuilt.nodes[node_id].node_type, NodeType.CONCEPT)
        self.assertIn(node_id, {hit.node.id for hit in rebuilt.search("login auth")})

    def test_replay_drops_trees_whose_root_was_forgotten(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Concept", "worth its own tree")
        tree_id = memory.expand_node_to_tree(node_id)
        category = memory.trees[tree_id].category
        memory.forget(node_id, reason="no longer relevant")

        rebuilt = FibMind.rebuild_from_log(memory.events)

        self.assertNotIn(tree_id, rebuilt.trees)
        self.assertNotIn(category, rebuilt.category_roots)
        self.assertEqual(rebuilt.trees, memory.trees)
        self.assertEqual(rebuilt.category_roots, memory.category_roots)

    def test_json_store_round_trips_the_log(self) -> None:
        memory = _sample_forest()
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonStore(Path(tmp) / "memory.json")
            store.save(memory)
            restored = store.load()

        self.assertEqual(
            [event.to_dict() for event in memory.events],
            [event.to_dict() for event in restored.events],
        )

    def test_sqlite_store_round_trips_the_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteStore(Path(tmp) / "memory.db")
            with store.transaction() as memory:
                node_id = memory.append("code", "Login 401", "token expiry causes 401")
                memory.record_outcome(node_id, Verdict.CONFIRMED, source="pytest")

            restored = store.load()
            rebuilt = FibMind.rebuild_from_log(restored.events)

        self.assertEqual(set(restored.nodes), set(rebuilt.nodes))
        self.assertEqual(restored.nodes[node_id].confidence, rebuilt.nodes[node_id].confidence)


class ForgetTests(unittest.TestCase):
    def test_forget_removes_the_node_and_its_edges(self) -> None:
        memory = FibMind()
        first = memory.append("code", "Login 401", "token expiry causes 401 on login")
        second = memory.append("code", "Token refresh", "refresh rotates the token on 401")

        result = memory.forget(first, reason="stored by mistake")

        self.assertNotIn(first, memory.nodes)
        self.assertIn(second, memory.nodes)
        self.assertGreater(result["removed_edges"], 0)
        for edge in memory.edges.values():
            self.assertNotIn(first, {edge.from_node_id, edge.to_node_id})

    def test_forget_redacts_content_so_replay_cannot_resurrect_it(self) -> None:
        """Deletion has to reach the log, or the next rebuild undoes it."""
        memory = FibMind()
        node_id = memory.append("code", "Secret", "hunter2 is the production password")
        memory.revise(node_id, content="hunter2 is still the production password")

        memory.forget(node_id, reason="user asked for removal")

        serialized = repr([event.to_dict() for event in memory.events])
        self.assertNotIn("hunter2", serialized)
        self.assertIn(TOMBSTONE, serialized)

        rebuilt = FibMind.rebuild_from_log(memory.events)
        self.assertNotIn(node_id, rebuilt.nodes)

    def test_forget_survives_a_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteStore(Path(tmp) / "memory.db")
            with store.transaction() as memory:
                node_id = memory.append("code", "Secret", "hunter2 is the password")
            with store.transaction() as memory:
                memory.forget(node_id, reason="user asked for removal")

            restored = store.load()

        serialized = repr([event.to_dict() for event in restored.events])
        self.assertNotIn("hunter2", serialized)
        self.assertNotIn(node_id, restored.nodes)
        self.assertNotIn(node_id, FibMind.rebuild_from_log(restored.events).nodes)

    def test_forget_requires_a_reason(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Bug", "a login bug")
        with self.assertRaises(ValueError):
            memory.forget(node_id, reason="   ")


class ReviseTests(unittest.TestCase):
    def test_revise_resets_confidence(self) -> None:
        """Evidence backs specific content, so it does not carry over."""
        memory = FibMind()
        node_id = memory.append("code", "Retry policy", "retry three times")
        memory.record_outcome(node_id, Verdict.CONFIRMED, source="pytest tests/test_retry.py")
        self.assertGreater(memory.nodes[node_id].confidence, 0.0)

        memory.revise(node_id, content="retry with exponential backoff")

        self.assertEqual(memory.nodes[node_id].content, "retry with exponential backoff")
        self.assertEqual(memory.nodes[node_id].confidence, 0.0)
        self.assertIsNone(memory.nodes[node_id].confidence_source)

    def test_revise_reactivates_a_refuted_memory(self) -> None:
        memory = FibMind()
        node_id = memory.append("code", "Retry policy", "retry forever")
        memory.record_outcome(node_id, Verdict.REFUTED, source="failing load test")

        self.assertEqual(memory.nodes[node_id].status, MemoryStatus.REFUTED)
        self.assertEqual(memory.search("retry forever"), [])

        memory.revise(node_id, content="retry three times")

        self.assertEqual(memory.nodes[node_id].status, MemoryStatus.ACTIVE)
        self.assertIn(node_id, {hit.node.id for hit in memory.search("retry three times")})


class StatusTests(unittest.TestCase):
    def test_stale_and_refuted_memories_are_hidden_but_maintainable(self) -> None:
        memory = FibMind()
        stale = memory.append("code", "Legacy import", "use FastMCP")
        refuted = memory.append("code", "Retry claim", "retry forever")
        memory.mark_stale(stale, "replaced by MCPServer")
        memory.record_outcome(refuted, Verdict.REFUTED, source="load test")

        self.assertEqual(memory.search("FastMCP retry forever", top_k=10), [])
        inactive = {
            hit.node.id
            for hit in memory.search(
                "FastMCP retry forever",
                top_k=10,
                statuses={MemoryStatus.STALE, MemoryStatus.REFUTED},
            )
        }
        self.assertEqual(inactive, {stale, refuted})

    def test_status_changes_survive_event_replay(self) -> None:
        memory = FibMind()
        stale = memory.append("code", "Legacy import", "use FastMCP")
        refuted = memory.append("code", "Retry claim", "retry forever")
        memory.mark_stale(stale, "replaced")
        memory.record_outcome(refuted, Verdict.REFUTED, source="pytest")

        rebuilt = FibMind.rebuild_from_log(memory.events)

        self.assertEqual(rebuilt.nodes[stale].status, MemoryStatus.STALE)
        self.assertEqual(rebuilt.nodes[refuted].status, MemoryStatus.REFUTED)
        self.assertIn(EventOp.SET_STATUS, {event.op for event in memory.events})


if __name__ == "__main__":
    unittest.main()
