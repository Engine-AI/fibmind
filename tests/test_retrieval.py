"""Hybrid retrieval: inverted index, BM25 fusion, vectors, explainability.

The invariants from ``ranking.py`` are re-asserted here against the new engine:
familiarity never moves a result, confidence does, identity filtering holds.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fibmind import (
    EmbeddingCache,
    FibMind,
    HashingEmbeddingProvider,
    MemoryService,
    MemoryStatus,
    Verdict,
)
from fibmind.embedding import cosine, provider_from_env
from fibmind.retrieval import LexicalIndex, retrieve


def _memory() -> FibMind:
    memory = FibMind()
    memory.append("code", "Login bug", "401 returned after token expiry", owner="a", workspace_id="w", project_id="p")
    memory.append("code", "Cache layer", "LRU cache for search results", owner="a", workspace_id="w", project_id="p")
    memory.append("requirement", "Refresh token", "Rotate refresh token on expiry", owner="a", workspace_id="w", project_id="p")
    return memory


IDENT = dict(owner="a", workspace_id="w", project_id="p")


class LexicalIndexTests(unittest.TestCase):
    def test_index_tracks_append_revise_and_forget(self) -> None:
        memory = _memory()
        self.assertEqual(len(memory.lexical), len(memory.nodes))
        node_id = memory.search("login", **IDENT)[0].node.id

        memory.revise(node_id, content="503 from the gateway")
        self.assertEqual(memory.search("token expiry 401", **IDENT)[0].node.title, "Refresh token")
        self.assertEqual(memory.search("gateway 503", **IDENT)[0].node.id, node_id)

        memory.forget(node_id, reason="test")
        self.assertNotIn(node_id, memory.lexical.terms)
        self.assertEqual([hit.node.id for hit in memory.search("gateway", **IDENT)], [])

    def test_rebuild_from_log_rebuilds_the_index(self) -> None:
        memory = _memory()
        replayed = FibMind.rebuild_from_log(memory.events)
        self.assertEqual(len(replayed.lexical), len(memory.lexical))
        self.assertEqual(
            [hit.node.title for hit in replayed.search("login token expiry", **IDENT)],
            [hit.node.title for hit in memory.search("login token expiry", **IDENT)],
        )

    def test_only_nodes_sharing_a_term_are_candidates(self) -> None:
        index = LexicalIndex()
        memory = _memory()
        index.rebuild(memory.nodes.values())
        self.assertEqual(index.candidates({"kubernetes"}), set())
        self.assertEqual(len(index.candidates({"token"})), 2)

    def test_bm25_rewards_rare_terms(self) -> None:
        memory = FibMind()
        for i in range(20):
            memory.append("code", f"note {i}", "the retry policy applies to uploads", owner="a")
        memory.append("code", "special", "the retry policy applies to kafka", owner="a")
        hits = memory.search("retry kafka", top_k=3, owner="a")
        self.assertEqual(hits[0].node.title, "special")


class RankingInvariantTests(unittest.TestCase):
    def test_relevant_node_first_and_title_bonus(self) -> None:
        memory = _memory()
        self.assertEqual(memory.search("login token expiry", **IDENT)[0].node.title, "Login bug")
        memory2 = FibMind()
        memory2.append("code", "Cache", "handles requests")
        memory2.append("code", "Request handler", "reads the cache once")
        self.assertEqual(memory2.search("cache")[0].node.title, "Cache")

    def test_familiarity_never_reorders(self) -> None:
        memory = _memory()
        before = [hit.node.id for hit in memory.search("login token", **IDENT)]
        for node in memory.nodes.values():
            node.familiarity = 1.0
            node.access_count = 99
        self.assertEqual(before, [hit.node.id for hit in memory.search("login token", **IDENT)])

    def test_confidence_lifts(self) -> None:
        memory = _memory()
        ranked = memory.search("login token", **IDENT)
        runner_up = ranked[1].node
        before = ranked[1].score
        memory.record_outcome(runner_up.id, Verdict.CONFIRMED, source="pytest")
        after = next(hit.score for hit in memory.search("login token", **IDENT) if hit.node.id == runner_up.id)
        self.assertGreater(after, before)

    def test_identity_and_status_still_filter(self) -> None:
        memory = _memory()
        self.assertEqual(memory.search("login token", owner="b", workspace_id="w", project_id="p"), [])
        self.assertEqual(memory.search("login token", owner="a", workspace_id="w", project_id="other"), [])
        node_id = memory.search("login", **IDENT)[0].node.id
        memory.mark_stale(node_id, "old")
        self.assertNotIn(node_id, [hit.node.id for hit in memory.search("login token", **IDENT)])
        self.assertIn(node_id, [hit.node.id for hit in memory.search("login token", statuses={MemoryStatus.STALE}, **IDENT)])

    def test_chinese_bigrams_still_match(self) -> None:
        memory = FibMind()
        memory.append("代码", "登录故障", "令牌过期后返回 401")
        hits = memory.search("登录 令牌")
        self.assertEqual(hits[0].node.title, "登录故障")
        self.assertIn("登录", hits[0].matched_terms)


class VectorTests(unittest.TestCase):
    def test_hashing_provider_is_deterministic_and_fuzzy(self) -> None:
        provider = HashingEmbeddingProvider()
        a, b = provider.embed(["upload retries with backoff", "upload retries with backoff"])
        self.assertEqual(a, b)
        near, far = provider.embed(["upload retrying with backoff", "kubernetes ingress tls"])
        self.assertGreater(cosine(a, near), cosine(a, far))

    def test_vector_candidates_are_fused_with_lexical(self) -> None:
        memory = FibMind()
        memory.append("code", "Upload retry", "uploads are retried with exponential backoff", owner="a")
        memory.append("code", "Kafka lag", "consumer lag alerts fire after five minutes", owner="a")
        memory.embeddings = EmbeddingCache(HashingEmbeddingProvider())
        # "retrying" shares no exact token with "retried"; trigram vectors bridge it.
        ranked, report = retrieve(memory, "retrying backoffs", owner="a", embeddings=memory.embeddings, explain=True)
        self.assertTrue(report["vector_enabled"])
        self.assertGreaterEqual(report["vector_candidates"], 1)
        self.assertEqual(ranked[0].node.title, "Upload retry")
        self.assertIn("vector", ranked[0].sources)

    def test_vectors_do_not_invent_results_for_unrelated_queries(self) -> None:
        memory = FibMind()
        memory.append("code", "Upload retry", "uploads are retried with exponential backoff", owner="a")
        memory.embeddings = EmbeddingCache(HashingEmbeddingProvider())
        self.assertEqual(memory.search("kubernetes ingress tls certificate", owner="a"), [])

    def test_provider_failure_degrades_to_lexical(self) -> None:
        class Broken:
            name = "broken"

            def embed(self, texts):
                raise OSError("connection refused")

        memory = _memory()
        memory.embeddings = EmbeddingCache(Broken())
        hits = memory.search("login token expiry", **IDENT)
        self.assertEqual(hits[0].node.title, "Login bug")
        self.assertEqual(memory.embeddings.failures, 1)
        self.assertIn("connection refused", memory.embeddings.last_error)

    def test_cache_embeds_each_text_once(self) -> None:
        calls = []

        class Counting(HashingEmbeddingProvider):
            def embed(self, texts):
                calls.append(len(texts))
                return super().embed(texts)

        cache = EmbeddingCache(Counting())
        cache.vectors_for(["a", "b"])
        cache.vectors_for(["a", "b", "c"])
        self.assertEqual(calls, [2, 1])
        self.assertEqual(len(cache), 3)

    def test_provider_from_env(self) -> None:
        self.assertIsNone(provider_from_env({}))
        self.assertIsNone(provider_from_env({"FIBMIND_EMBEDDING": "off"}))
        self.assertIsInstance(provider_from_env({"FIBMIND_EMBEDDING": "hashing"}), HashingEmbeddingProvider)
        with self.assertRaises(ValueError):
            provider_from_env({"FIBMIND_EMBEDDING": "openai"})
        with self.assertRaises(ValueError):
            provider_from_env({"FIBMIND_EMBEDDING": "quantum"})
        openai = provider_from_env(
            {"FIBMIND_EMBEDDING": "openai", "FIBMIND_EMBEDDING_URL": "http://localhost:11434/v1/", "FIBMIND_EMBEDDING_MODEL": "nomic"}
        )
        self.assertEqual(openai.name, "openai:nomic")
        self.assertEqual(openai.url, "http://localhost:11434/v1")


class ExplainTests(unittest.TestCase):
    def test_explain_reports_signals_and_exclusions(self) -> None:
        memory = _memory()
        stale_id = memory.search("cache", **IDENT)[0].node.id
        memory.mark_stale(stale_id, "superseded")
        memory.append("code", "Other owner cache", "cache for bob", owner="b", workspace_id="w", project_id="p")

        report = memory.explain_recall("cache search results", **IDENT)

        self.assertEqual(report["query_terms"], ["cache", "results", "search"])
        self.assertEqual(report["results"], [])
        reasons = {row["title"]: row["reason"] for row in report["excluded"]}
        self.assertEqual(reasons["Cache layer"], "status:stale")
        self.assertEqual(reasons["Other owner cache"], "not_visible")
        self.assertEqual(report["weights"]["familiarity"], 0.0)

        report = memory.explain_recall("login token expiry", **IDENT)
        top = report["results"][0]
        self.assertEqual(top["title"], "Login bug")
        self.assertEqual(top["sources"], ["lexical"])
        self.assertGreater(top["bm25"], 0)
        self.assertEqual(top["coverage"], 1.0)
        self.assertIsNone(top["vector"])

    def test_service_explain_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(Path(tmp) / "m.db")
            service.append("code", "Login bug", "401 after token expiry", owner="a")
            report = service.explain_recall("login 401", owner="a")
            self.assertEqual(report["results"][0]["title"], "Login bug")
            status = service.status()
            self.assertEqual(status["nodes"], 2)  # root + node
            self.assertFalse(status["embedding"]["enabled"])
            self.assertGreater(status["lexical_terms"], 0)


if __name__ == "__main__":
    unittest.main()
