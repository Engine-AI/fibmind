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


if __name__ == "__main__":
    unittest.main()
