"""Tests for the keyword ranking module."""

import unittest

from fibmind import FibMind
from fibmind.ranking import ScoredHit, rank_nodes, tokenize


class TokenizeTests(unittest.TestCase):
    def test_tokenize_lowercases_and_splits_on_non_alnum(self) -> None:
        self.assertEqual(tokenize("Login-Bug: 401!"), ["login", "bug", "401"])

    def test_tokenize_returns_empty_for_blank(self) -> None:
        self.assertEqual(tokenize("   "), [])


class RankNodesTests(unittest.TestCase):
    def _memory(self) -> FibMind:
        memory = FibMind()
        memory.append("code", "Login bug", "401 returned after token expiry")
        memory.append("code", "Cache layer", "LRU cache for search results")
        memory.append("requirement", "Refresh token", "Rotate refresh token on expiry")
        return memory

    def test_rank_returns_scored_hits_sorted_desc(self) -> None:
        memory = self._memory()

        hits = rank_nodes(memory.nodes.values(), "login token expiry")

        self.assertTrue(all(isinstance(hit, ScoredHit) for hit in hits))
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_ranks_relevant_node_first(self) -> None:
        memory = self._memory()

        hits = rank_nodes(memory.nodes.values(), "login token expiry")

        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0].node.title, "Login bug")

    def test_title_match_outranks_content_only_match(self) -> None:
        memory = FibMind()
        memory.append("code", "Cache", "handles requests")
        memory.append("code", "Request handler", "reads the cache once")

        hits = rank_nodes(memory.nodes.values(), "cache")

        self.assertEqual(hits[0].node.title, "Cache")

    def test_min_score_filters_irrelevant_nodes(self) -> None:
        memory = self._memory()

        hits = rank_nodes(memory.nodes.values(), "kubernetes deployment", min_score=0.01)

        self.assertEqual(hits, [])

    def test_top_k_limits_results(self) -> None:
        memory = self._memory()

        hits = rank_nodes(memory.nodes.values(), "token", top_k=1)

        self.assertLessEqual(len(hits), 1)

    def test_empty_query_returns_no_hits(self) -> None:
        memory = self._memory()

        self.assertEqual(rank_nodes(memory.nodes.values(), ""), [])

    def test_matched_terms_are_reported(self) -> None:
        memory = self._memory()

        hits = rank_nodes(memory.nodes.values(), "login expiry")

        top = hits[0]
        self.assertIn("login", top.matched_terms)
        self.assertIn("expiry", top.matched_terms)

    def test_rank_does_not_mutate_nodes(self) -> None:
        memory = self._memory()
        before = {nid: (n.access_count, n.importance) for nid, n in memory.nodes.items()}

        rank_nodes(memory.nodes.values(), "login token")

        after = {nid: (n.access_count, n.importance) for nid, n in memory.nodes.items()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
