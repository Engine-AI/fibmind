"""Tests for the MemoryService persistence/orchestration layer."""

import tempfile
import unittest
from pathlib import Path

from fibmind.service import MemoryService


class MemoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store_path = Path(self._tmp.name) / "memory.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_append_persists_to_disk(self) -> None:
        service = MemoryService(self.store_path)

        result = service.append("code", "Login bug", "401 after token expiry")

        self.assertIn("node_id", result)
        self.assertTrue(self.store_path.exists())

    def test_append_survives_reload(self) -> None:
        service = MemoryService(self.store_path)
        result = service.append("code", "Login bug", "401 after token expiry")

        reloaded = MemoryService(self.store_path)
        found = reloaded.search("login token")

        node_ids = {hit["node_id"] for hit in found["results"]}
        self.assertIn(result["node_id"], node_ids)

    def test_append_validates_required_fields(self) -> None:
        service = MemoryService(self.store_path)

        with self.assertRaises(ValueError):
            service.append("code", "  ", "content")
        with self.assertRaises(ValueError):
            service.append("", "title", "content")

    def test_search_returns_ranked_results(self) -> None:
        service = MemoryService(self.store_path)
        service.append("code", "Login bug", "401 after token expiry")
        service.append("code", "Cache", "kubernetes manifest")

        result = service.search("login token expiry")

        self.assertEqual(result["results"][0]["title"], "Login bug")

    def test_search_rejects_empty_query(self) -> None:
        service = MemoryService(self.store_path)

        with self.assertRaises(ValueError):
            service.search("   ")

    def test_search_does_not_write(self) -> None:
        service = MemoryService(self.store_path)
        service.append("code", "Login bug", "401 after token expiry")
        mtime_before = self.store_path.stat().st_mtime_ns

        service.search("login")

        self.assertEqual(self.store_path.stat().st_mtime_ns, mtime_before)

    def test_search_from_expands_related_nodes(self) -> None:
        service = MemoryService(self.store_path)
        a = service.append("code", "Login bug", "401 after token expiry")
        b = service.append("requirement", "Refresh token", "rotate on expiry")
        service.link(a["node_id"], b["node_id"], "related_to")

        result = service.search_from(a["node_id"], depth=1)

        titles = {hit["title"] for hit in result["hits"]}
        self.assertIn("Refresh token", titles)

    def test_search_from_defaults_to_both_directions(self) -> None:
        service = MemoryService(self.store_path)
        a = service.append("code", "Login bug", "401 after token expiry")
        b = service.append("requirement", "Refresh token", "rotate on expiry")
        service.link(a["node_id"], b["node_id"], "related_to")

        result = service.search_from(b["node_id"], depth=1)

        related = next(hit for hit in result["hits"] if hit["node_id"] == a["node_id"])
        self.assertEqual(result["direction"], "both")
        self.assertEqual(related["via_direction"], "in")

    def test_search_from_can_use_outgoing_only(self) -> None:
        service = MemoryService(self.store_path)
        # Deliberately unrelated wording: similar memories are auto-linked with a
        # bidirectional edge, which would legitimately make `a` reachable and
        # obscure the direction filtering this test is about.
        a = service.append("code", "Login bug", "401 after token expiry")
        b = service.append("requirement", "Sidebar width", "widen the settings panel")
        service.link(a["node_id"], b["node_id"], "related_to")

        result = service.search_from(b["node_id"], depth=1, direction="out")

        self.assertNotIn(a["node_id"], {hit["node_id"] for hit in result["hits"]})

    def test_link_returns_edge_summary_and_validates_weight(self) -> None:
        service = MemoryService(self.store_path)
        a = service.append("code", "Login bug", "401 after token expiry")
        b = service.append("requirement", "Refresh token", "rotate on expiry")

        result = service.link(
            a["node_id"],
            b["node_id"],
            "related_to",
            weight=0.75,
            bidirectional=True,
            metadata={"source": "test"},
        )

        self.assertEqual(result["weight"], 0.75)
        self.assertEqual(result["direction"], "bidirectional")
        with self.assertRaises(ValueError):
            service.link(a["node_id"], b["node_id"], "related_to", weight=1.1)

    def test_search_from_unknown_node_raises_value_error(self) -> None:
        service = MemoryService(self.store_path)

        with self.assertRaises(ValueError):
            service.search_from("node_does_not_exist")

    def test_context_returns_text_and_hits(self) -> None:
        service = MemoryService(self.store_path)
        service.append("code", "Login bug", "401 after token expiry")

        result = service.context("login token expiry")

        self.assertIn("Login bug", result["text"])
        self.assertGreater(len(result["hits"]), 0)

    def test_context_reinforce_persists(self) -> None:
        service = MemoryService(self.store_path)
        service.append("code", "Login bug", "401 after token expiry")
        mtime_before = self.store_path.stat().st_mtime_ns

        service.context("login token expiry", reinforce=True)

        self.assertNotEqual(self.store_path.stat().st_mtime_ns, mtime_before)

    def test_sqlite_services_observe_each_others_writes(self) -> None:
        sqlite_path = self.store_path.with_suffix(".db")
        first = MemoryService(sqlite_path)
        second = MemoryService(sqlite_path)

        a = first.append("code", "Login bug", "401 after token expiry")
        b = second.append("requirement", "Refresh token", "rotate on expiry")
        first.link(a["node_id"], b["node_id"], "related_to")

        found = second.search("login refresh token", top_k=10)
        found_ids = {hit["node_id"] for hit in found["results"]}
        self.assertIn(a["node_id"], found_ids)
        self.assertIn(b["node_id"], found_ids)

    def test_record_outcome_requires_a_named_source(self) -> None:
        service = MemoryService(self.store_path)
        node = service.append("code", "Login bug", "401 after token expiry")

        with self.assertRaises(ValueError):
            service.record_outcome(node["node_id"], "confirmed", "   ")
        with self.assertRaises(ValueError):
            service.record_outcome(node["node_id"], "probably", "pytest")

        updated = service.record_outcome(node["node_id"], "confirmed", "pytest tests/test_auth.py")
        self.assertGreater(updated["confidence"], 0.0)
        self.assertEqual(updated["confidence_source"], "pytest tests/test_auth.py")

    def test_outcome_persists_across_reload(self) -> None:
        service = MemoryService(self.store_path)
        node = service.append("code", "Login bug", "401 after token expiry")
        service.record_outcome(node["node_id"], "confirmed", "pytest")

        reloaded = MemoryService(self.store_path).search("login token")
        hit = next(
            item for item in reloaded["results"] if item["node_id"] == node["node_id"]
        )
        self.assertGreater(hit["confidence"], 0.0)

    def test_revise_needs_something_to_change(self) -> None:
        service = MemoryService(self.store_path)
        node = service.append("code", "Retry policy", "retry three times")

        with self.assertRaises(ValueError):
            service.revise(node["node_id"])

        revised = service.revise(node["node_id"], content="retry with backoff")
        self.assertEqual(revised["confidence"], 0.0)

    def test_mark_stale_hides_memory_and_explicit_status_can_find_it(self) -> None:
        service = MemoryService(self.store_path)
        node = service.append("code", "Legacy import", "use FastMCP")

        stale = service.mark_stale(node["node_id"], "replaced by MCPServer")

        self.assertEqual(stale["status"], "stale")
        self.assertEqual(service.search("FastMCP")["results"], [])
        maintenance = service.search("FastMCP", statuses=["stale"])
        self.assertEqual([hit["node_id"] for hit in maintenance["results"]], [node["node_id"]])

    def test_refuted_outcome_hides_memory_until_revised(self) -> None:
        service = MemoryService(self.store_path)
        node = service.append("code", "Retry policy", "retry forever")

        refuted = service.record_outcome(node["node_id"], "refuted", "load test")

        self.assertEqual(refuted["status"], "refuted")
        self.assertEqual(service.search("retry forever")["results"], [])
        revised = service.revise(node["node_id"], content="retry three times")
        self.assertEqual(revised["status"], "active")

    def test_forget_removes_the_memory_from_the_store(self) -> None:
        service = MemoryService(self.store_path)
        node = service.append("code", "Secret", "hunter2 is the production password")

        service.forget(node["node_id"], reason="user asked for removal")

        found = service.search("hunter2 password", top_k=10)
        self.assertEqual(found["results"], [])
        self.assertNotIn("hunter2", self.store_path.read_text(encoding="utf-8"))

    def test_append_refuses_to_write_knowledge_directly(self) -> None:
        """Shared claims must go through the promotion gate, not around it."""
        service = MemoryService(self.store_path)

        with self.assertRaises(ValueError):
            service.append("math", "Everyone skips this", "a general claim", scope="knowledge")
        with self.assertRaises(ValueError):
            service.append("math", "Bad scope", "content", scope="team")

    def test_promote_knowledge_enforces_supporting_evidence(self) -> None:
        service = MemoryService(self.store_path)
        supporting = [
            service.append(
                "math",
                f"Missed the discriminant #{index}",
                "skipped checking whether the discriminant is negative",
                owner="student-a",
            )["node_id"]
            for index in range(3)
        ]

        with self.assertRaises(ValueError):
            service.promote_knowledge("Claim", "content", [])
        with self.assertRaises(ValueError):
            service.promote_knowledge("Claim", "content", supporting[:2])

        claim = service.promote_knowledge(
            "Discriminant sign is routinely skipped",
            "Students skip checking whether the discriminant is negative",
            supporting,
        )
        self.assertEqual(claim["scope"], "knowledge")
        self.assertEqual(claim["confidence"], 0.0)

    def test_personal_memories_are_not_visible_to_other_owners(self) -> None:
        service = MemoryService(self.store_path)
        mine = service.append(
            "prefs", "Prefers terse answers", "wants short replies", owner="student-a"
        )

        visible = service.search("prefers replies", owner="student-b")

        self.assertNotIn(mine["node_id"], {hit["node_id"] for hit in visible["results"]})


if __name__ == "__main__":
    unittest.main()
