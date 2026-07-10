"""Tests for the context-pack builder."""

import unittest

from fibmind import FibMind
from fibmind.context import ContextPack, build_context


class BuildContextTests(unittest.TestCase):
    def _memory(self) -> FibMind:
        memory = FibMind()
        login = memory.append("code", "Login bug", "401 returned after token expiry")
        req = memory.append("requirement", "Refresh token", "Rotate refresh token on expiry")
        memory.append("code", "Unrelated", "kubernetes deployment manifest")
        from fibmind import RelationType

        memory.link_nodes(login, req, RelationType.RELATED_TO)
        return memory

    def test_build_context_returns_pack_with_hits_and_text(self) -> None:
        memory = self._memory()

        pack = build_context(memory, "login token expiry")

        self.assertIsInstance(pack, ContextPack)
        self.assertGreater(len(pack.hits), 0)
        self.assertIn("Login bug", pack.text)

    def test_build_context_expands_related_nodes(self) -> None:
        memory = self._memory()

        pack = build_context(memory, "login token expiry", top_k=1, depth=1)

        titles = {hit.node.title for hit in pack.hits}
        # The directly-related requirement is pulled in via graph expansion.
        self.assertIn("Refresh token", titles)

    def test_build_context_respects_max_chars(self) -> None:
        memory = self._memory()

        pack = build_context(memory, "login token expiry", max_chars=80)

        self.assertLessEqual(len(pack.text), 80)

    def test_build_context_empty_when_no_match(self) -> None:
        memory = self._memory()

        pack = build_context(memory, "nonexistent zzzzz")

        self.assertEqual(pack.hits, [])
        self.assertEqual(pack.text, "")

    def test_build_context_is_read_only_by_default(self) -> None:
        memory = self._memory()
        before = {nid: (n.access_count, n.importance) for nid, n in memory.nodes.items()}

        build_context(memory, "login token expiry")

        after = {nid: (n.access_count, n.importance) for nid, n in memory.nodes.items()}
        self.assertEqual(before, after)

    def test_build_context_reinforces_when_requested(self) -> None:
        memory = self._memory()

        pack = build_context(memory, "login token expiry", reinforce=True)

        top_node_id = pack.hits[0].node.id
        self.assertGreater(memory.nodes[top_node_id].access_count, 0)

    def test_pack_to_dict_is_serializable(self) -> None:
        memory = self._memory()

        pack = build_context(memory, "login token expiry")
        data = pack.to_dict()

        self.assertIn("text", data)
        self.assertIn("hits", data)
        self.assertIsInstance(data["hits"], list)
        self.assertIn("node_id", data["hits"][0])


if __name__ == "__main__":
    unittest.main()
