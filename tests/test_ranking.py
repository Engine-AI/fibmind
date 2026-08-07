"""Tests for the keyword ranking module."""

import unittest

from fibmind import FibMind
from fibmind.ranking import ScoredHit, rank_nodes, tokenize


class TokenizeTests(unittest.TestCase):
    def test_tokenize_lowercases_and_splits_on_non_alnum(self) -> None:
        self.assertEqual(tokenize("Login-Bug: 401!"), ["login", "bug", "401"])

    def test_tokenize_returns_empty_for_blank(self) -> None:
        self.assertEqual(tokenize("   "), [])

    def test_tokenize_builds_cjk_bigrams(self) -> None:
        self.assertEqual(tokenize("登录故障"), ["登录故障", "登录", "录故", "故障"])

    def test_tokenize_normalizes_full_width_text(self) -> None:
        self.assertEqual(tokenize("ＡＰＩ ４０１"), ["api", "401"])

    def test_tokenize_drops_common_english_stop_words(self) -> None:
        self.assertEqual(tokenize("what is the current backend"), ["current", "backend"])


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
        before = {nid: (n.access_count, n.familiarity) for nid, n in memory.nodes.items()}

        rank_nodes(memory.nodes.values(), "login token")

        after = {nid: (n.access_count, n.familiarity) for nid, n in memory.nodes.items()}
        self.assertEqual(before, after)

    def test_familiarity_does_not_affect_ranking(self) -> None:
        """Repeated recall must not reorder results.

        This is the loop that would otherwise form: whatever gets read comes back
        higher, so the store converges on whatever is familiar rather than
        whatever is right.
        """
        memory = self._memory()
        baseline = [hit.node.id for hit in rank_nodes(memory.nodes.values(), "login token")]

        for node in memory.nodes.values():
            node.familiarity = 1.0
            node.access_count = 99
        memory.nodes[baseline[-1]].familiarity = 1.0

        after = [hit.node.id for hit in rank_nodes(memory.nodes.values(), "login token")]
        self.assertEqual(baseline, after)

    def test_confidence_lifts_a_memory(self) -> None:
        """Evidence is the one signal allowed to change the ordering."""
        memory = self._memory()
        ranked = rank_nodes(memory.nodes.values(), "login token")
        self.assertGreater(len(ranked), 1)
        runner_up = ranked[1].node

        before = ranked[1].score
        runner_up.confidence = 1.0
        after = next(
            hit.score
            for hit in rank_nodes(memory.nodes.values(), "login token")
            if hit.node.id == runner_up.id
        )

        self.assertGreater(after, before)

    def test_rank_supports_chinese_queries(self) -> None:
        memory = FibMind()
        memory.append("代码", "登录故障", "令牌过期后返回 401")

        hits = rank_nodes(memory.nodes.values(), "登录 令牌")

        self.assertEqual(hits[0].node.title, "登录故障")
        self.assertIn("登录", hits[0].matched_terms)
        self.assertIn("令牌", hits[0].matched_terms)


if __name__ == "__main__":
    unittest.main()
