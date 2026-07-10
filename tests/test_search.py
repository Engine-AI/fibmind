"""Tests for FibMind.search (query-based recall over the whole forest)."""

import unittest

from fibmind import FibMind, NodeType


class FibMindSearchTests(unittest.TestCase):
    def _memory(self) -> FibMind:
        memory = FibMind()
        memory.append("code", "Login bug", "401 returned after token expiry")
        memory.append("code", "Cache layer", "LRU cache for search results")
        memory.append("requirement", "Refresh token", "Rotate refresh token on expiry")
        return memory

    def test_search_finds_relevant_node(self) -> None:
        memory = self._memory()

        hits = memory.search("login token expiry")

        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0].node.title, "Login bug")

    def test_search_excludes_root_nodes_by_default(self) -> None:
        memory = self._memory()

        hits = memory.search("code", include_roots=False)

        self.assertTrue(all(hit.node.node_type != NodeType.ROOT for hit in hits))

    def test_search_filters_by_category(self) -> None:
        memory = self._memory()

        hits = memory.search("token", categories={"requirement"})

        self.assertTrue(all(hit.node.category == "requirement" for hit in hits))

    def test_search_is_read_only(self) -> None:
        memory = self._memory()
        before = {nid: (n.access_count, n.importance) for nid, n in memory.nodes.items()}

        memory.search("login token expiry")

        after = {nid: (n.access_count, n.importance) for nid, n in memory.nodes.items()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
