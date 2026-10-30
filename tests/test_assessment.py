"""Unit and regression tests for FibBrain Metacognition (assessment.py, G0-G4)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fibmind.assessment import (
    DeterministicAssessor,
    classify,
    route_unknown,
)
from fibmind.brain import BrainState, FibBrain
from fibmind.distill import Aspect
from fibmind.graph import FibMind
from fibmind.models import MemoryKind, MemoryNode, MemoryStatus, NodeType
from fibmind.retrieval import LexicalIndex
from fibmind.service import MemoryService


class AssessmentUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lexical = LexicalIndex()

    def test_deterministic_assessor_extracts_cjk_and_symbols(self) -> None:
        assessor = DeterministicAssessor()
        goal = "fix login 401 in auth/session.py and 并发刷新"
        query_terms = {"auth", "session", "py", "并发", "刷新"}
        aspects = assessor.aspects(goal, query_terms)
        aspect_names = [a.aspect for a in aspects]

        self.assertIn("auth/session.py", aspect_names)
        self.assertIn("并发刷新", aspect_names)

    def test_relevance_floor_excludes_weak_memories(self) -> None:
        node = MemoryNode(
            id="node_weak",
            category="architecture",
            title="Recall memory from forest",
            content="General memory about recall and forest indexing",
            tags=("recall",),
            confidence=0.75,
            node_type=NodeType.RAW,
        )
        self.lexical.add(node)
        goal = "deploy kubernetes cluster with autoscaling pods"
        result = classify(
            goal,
            candidates=[node],
            lexical=self.lexical,
        )
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(len(result["known"]), 0)
        self.assertEqual(len(result["weak"]), 1)
        self.assertEqual(result["weak"][0]["node_id"], "node_weak")

    def test_contradicted_memory_routes_to_avoid(self) -> None:
        node = MemoryNode(
            id="node_refuted",
            category="decision",
            title="Retry login without token refresh",
            content="Immediately retry login on 401 without refreshing",
            status=MemoryStatus.REFUTED,
            status_reason="refuted by pytest integration tests",
            node_type=NodeType.RAW,
        )
        self.lexical.add(node)
        goal = "retry login on 401 without refreshing"
        result = classify(
            goal,
            candidates=[node],
            lexical=self.lexical,
        )
        self.assertEqual(len(result["contradicted"]), 1)
        self.assertEqual(result["contradicted"][0]["route"], "avoid")
        self.assertEqual(result["next"][0], {"route": "avoid", "target": "node_refuted"})

    def test_known_memory_requires_high_confidence_and_freshness(self) -> None:
        now = datetime(2026, 10, 20, 0, 0, 0, tzinfo=UTC)
        fresh_time = now - timedelta(days=10)
        node = MemoryNode(
            id="node_known",
            category="decision",
            title="Refresh access token before retrying login request",
            content="When login fails with 401, refresh tokens first before retrying",
            confidence=0.75,
            created_at=fresh_time,
            node_type=NodeType.RAW,
        )
        self.lexical.add(node)

        def outcomes(nid: str) -> dict[str, int]:
            return {"confirmed": 2, "refuted": 0, "sessions": 2}

        def last_conf(nid: str) -> datetime:
            return fresh_time

        goal = "refresh access token before retrying login request"
        result = classify(
            goal,
            candidates=[node],
            lexical=self.lexical,
            outcome_lookup=outcomes,
            last_confirmation_lookup=last_conf,
            now=now,
        )
        self.assertEqual(result["verdict"], "known")
        self.assertEqual(len(result["known"]), 1)
        self.assertIn("confirmed 2x in 2 sessions", result["known"][0]["evidence"])

    def test_uncertain_memory_reasons_and_routing(self) -> None:
        # 1. Unchecked with command -> route verify
        node_cmd = MemoryNode(
            id="node_cmd",
            category="procedure",
            title="Database migration script",
            content="Run migration script to upgrade database tables",
            confidence=0.0,
            metadata={"command": "pytest tests/test_db.py"},
            node_type=NodeType.RAW,
        )
        self.lexical.add(node_cmd)
        res_cmd = classify("database migration script upgrade", [node_cmd], lexical=self.lexical)
        self.assertEqual(len(res_cmd["uncertain"]), 1)
        self.assertIn("unchecked", res_cmd["uncertain"][0]["reasons"])
        self.assertEqual(res_cmd["uncertain"][0]["route"], "verify")
        self.assertEqual(res_cmd["uncertain"][0]["target"], "pytest tests/test_db.py")

        # 2. File drift -> route code
        with tempfile.TemporaryDirectory() as tmp_dir:
            drift_file = Path(tmp_dir) / "drift.py"
            drift_file.write_text("modified code", encoding="utf-8")
            node_file = MemoryNode(
                id="node_file",
                category="decision",
                title="Audit token validation module",
                content="Audit token validation in drift.py",
                confidence=0.75,
                metadata={
                    "files": ["drift.py"],
                    "file_hashes": {"drift.py": "original_hash_value_that_differs"},
                },
                node_type=NodeType.RAW,
            )
            self.lexical.add(node_file)
            res_file = classify(
                "audit token validation module in drift.py", [node_file], lexical=self.lexical, cwd=tmp_dir
            )
            self.assertEqual(len(res_file["uncertain"]), 1)
            self.assertTrue(any("file_hash_mismatch" in r for r in res_file["uncertain"][0]["reasons"]))
            self.assertEqual(res_file["uncertain"][0]["route"], "code")
            self.assertEqual(res_file["uncertain"][0]["target"], "drift.py")

        # 3. Dialectical conflict -> route resolve
        node_conf = MemoryNode(
            id="node_conf",
            category="decision",
            title="Synchronous webhook requests handler",
            content="Deliver webhooks synchronously",
            confidence=0.0,
            metadata={"conflicted": True},
            node_type=NodeType.RAW,
        )
        self.lexical.add(node_conf)
        res_conf = classify("synchronous webhook requests handler", [node_conf], lexical=self.lexical)
        self.assertEqual(res_conf["uncertain"][0]["route"], "resolve")
        self.assertEqual(res_conf["uncertain"][0]["target"], "node_conf")

        # 4. Preference without handle -> route ask
        node_pref = MemoryNode(
            id="node_pref",
            category="preference",
            title="Formatting convention for code style",
            content="Prefer black code formatting",
            confidence=0.0,
            memory_kind=MemoryKind.PREFERENCE,
            node_type=NodeType.RAW,
        )
        self.lexical.add(node_pref)
        res_pref = classify("formatting convention for code style", [node_pref], lexical=self.lexical)
        self.assertEqual(res_pref["uncertain"][0]["route"], "ask")
        self.assertEqual(res_pref["uncertain"][0]["target"], "user")

    def test_unknown_aspect_routing(self) -> None:
        # Symbol/path -> code
        r1, t1, _ = route_unknown("inspect src/auth.py", Aspect(aspect="src/auth.py", terms=("src", "auth", "py")))
        self.assertEqual(r1, "code")
        self.assertEqual(t1, "src/auth.py")

        # Preference intent -> ask
        r2, t2, _ = route_unknown("Should we prefer ruff or flake8?", Aspect(aspect="ruff", terms=("ruff",)))
        self.assertEqual(r2, "ask")
        self.assertEqual(t2, "user")

        # Time-sensitive -> search
        r3, t3, _ = route_unknown(
            "What is the latest release version of python?", Aspect(aspect="python", terms=("python",))
        )
        self.assertEqual(r3, "search")
        self.assertEqual(t3, "web search")

        # Open question match -> ask
        q_node = MemoryNode(
            id="q1",
            category="question",
            title="How to configure redis auth password?",
            content="Unresolved redis password authentication",
            node_type=NodeType.RAW,
        )
        r4, t4, _ = route_unknown(
            "configure redis password", Aspect(aspect="redis", terms=("redis",)), open_questions=[q_node]
        )
        self.assertEqual(r4, "ask")
        self.assertEqual(t4, "user")


class FibBrainMetacognitionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test_brain.json"
        self.service = MemoryService(self.db_path)
        self.brain = FibBrain(self.service)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_assess_emits_episode_event_when_session_id_present(self) -> None:
        state = BrainState(session_id="session-meta-1", workspace_id="ws1", project_id="fibmind")
        res = self.brain.assess("deploy kubernetes pod cluster", state=state)
        self.assertEqual(res["verdict"], "unknown")

        episodes = self.service.read(
            lambda mem: [
                n for n in mem.nodes.values() if n.category == "episode" and n.metadata.get("kind") == "assess"
            ]
        )
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].metadata["payload"]["verdict"], "unknown")

    def test_plan_and_coordinate_synthesize_assessment_steps(self) -> None:
        state = BrainState(session_id="session-plan-1", workspace_id="ws1", project_id="fibmind")
        # Add a refuted memory
        refuted_id = self.service.append(
            category="decision",
            title="Deploy directly to production without testing",
            content="Bypass staging and push straight to prod",
            workspace_id=state.workspace_id,
            project_id=state.project_id,
            session_id=state.session_id,
        )["node_id"]
        self.service.run(lambda mem: mem.set_status(refuted_id, MemoryStatus.REFUTED, "refuted by safety policy"))

        # Add an active checked memory
        known_id = self.service.append(
            category="decision",
            title="Verify staging health before prod promote",
            content="Check staging HTTP 200 before promote",
            workspace_id=state.workspace_id,
            project_id=state.project_id,
            session_id=state.session_id,
        )["node_id"]

        def set_known(mem: FibMind) -> None:
            mem.nodes[known_id].confidence = 0.75
            mem.outcome_index[known_id] = {
                "confirmed": 2,
                "refuted": 0,
                "inconclusive": 0,
                "sessions": {"session-plan-1"},
                "last_confirmed_at": datetime.now(UTC),
            }

        self.service.run(set_known)

        planned = self.brain.plan("deploy to production and verify staging health", state=state)
        step_titles = [s["title"] for s in planned["steps"]]
        self.assertTrue(any(t.startswith("Avoid: Deploy directly to production") for t in step_titles))
        self.assertTrue(any(t.startswith("Reuse: Verify staging health") for t in step_titles))

        # Coordinate incorporates capabilities
        coordinated = self.brain.coordinate("deploy to production and verify staging health", state=state)
        caps = [c["capability"] for c in coordinated["capabilities"]]
        self.assertIn("memory", caps)

    def test_recall_include_gaps_renders_gaps_section_in_budget(self) -> None:
        state = BrainState(session_id="session-gaps-1", workspace_id="ws1", project_id="fibmind")
        pack = self.brain.recall(
            "unseen goal for testing gap rendering", state=state, budget_tokens=200, include_gaps=True
        )
        self.assertIn("Gaps:", pack["text"])
        self.assertIn("gaps", pack["budget"]["sections"])
        self.assertGreater(pack["budget"]["sections"]["gaps"]["used"], 0)

    def test_review_session_extracts_open_questions(self) -> None:
        state = BrainState(session_id="session-q-1", workspace_id="ws1", project_id="fibmind")
        # Call assess directly to record an assess event with unknown aspects
        self.brain.assess("solve unknown concurrency race in cache_lock.py", state=state)

        # Session makes unrelated decision and verification
        self.brain.observe("decision", "optimized database indices", state=state)
        self.brain.observe("test", "test suite passed", {"command": "pytest"}, state=state)

        review = self.brain.review_session(state=state, mode="candidates")
        question_candidates = [c for c in review["written"] if c["category"] == "question"]
        self.assertGreater(len(question_candidates), 0)
        self.assertTrue(
            any("cache_lock.py" in c["title"] or "cache_lock.py" in c["content"] for c in question_candidates)
        )

    def test_status_metacognition_counts(self) -> None:
        state = BrainState(session_id="session-stat-1", workspace_id="ws1", project_id="fibmind")
        # Write memory and assess
        mem_id = self.service.append(
            category="decision",
            title="Fast query caching layer",
            content="Enable redis cache for queries",
            workspace_id=state.workspace_id,
            project_id=state.project_id,
            session_id=state.session_id,
        )["node_id"]

        def set_mem(mem: FibMind) -> None:
            mem.nodes[mem_id].confidence = 0.75
            mem.outcome_index[mem_id] = {
                "confirmed": 2,
                "refuted": 0,
                "inconclusive": 0,
                "sessions": {"session-stat-1"},
                "last_confirmed_at": datetime.now(UTC),
            }

        self.service.run(set_mem)
        self.brain.assess("fast query caching layer", state=state)

        # Now refute it in same session -> overconfident
        self.brain.reflect("refuted", "cache produced stale data", node_id=mem_id, state=state)

        status_report = self.service.status()
        self.assertIn("metacognition", status_report)
        meta = status_report["metacognition"]
        self.assertGreaterEqual(meta["assessments"], 1)
        self.assertGreaterEqual(meta["overconfident"], 1)
        self.assertGreater(meta["p_refuted_given_known"], 0.0)

    def test_contract_guarantees_visibility_determinism_familiarity_and_custom_assessor(self) -> None:
        state_alice = BrainState(owner="alice", workspace_id="ws1", project_id="proj_a")
        state_bob = BrainState(owner="bob", workspace_id="ws1", project_id="proj_a")

        alice_node_id = self.service.append(
            category="decision",
            title="Use Ed25519 signatures for webhook tokens",
            content="Always sign webhook payloads with Ed25519 keys",
            owner="alice",
            workspace_id="ws1",
            project_id="proj_a",
        )["node_id"]

        # 1. Visibility: Bob cannot see Alice's personal memory in assess (neither known, uncertain, nor weak)
        bob_res = self.brain.assess("Use Ed25519 signatures for webhook tokens", state=state_bob)
        all_bob_ids = {
            *(x["node_id"] for x in bob_res["known"]),
            *(x["node_id"] for x in bob_res["uncertain"]),
            *(x["node_id"] for x in bob_res["contradicted"]),
            *(x["node_id"] for x in bob_res["weak"]),
        }
        self.assertNotIn(alice_node_id, all_bob_ids)
        self.assertEqual(bob_res["verdict"], "unknown")

        # 2. Determinism & Familiarity invariance: boosting familiarity must not change any field of assess
        alice_res_before = self.brain.assess("Use Ed25519 signatures for webhook tokens", state=state_alice)

        def boost_familiarity(mem: FibMind) -> None:
            mem.nodes[alice_node_id].familiarity = 999.0
            mem.nodes[alice_node_id].access_count = 500

        self.service.run(boost_familiarity)
        alice_res_after = self.brain.assess("Use Ed25519 signatures for webhook tokens", state=state_alice)
        self.assertEqual(alice_res_before, alice_res_after)

        # 3. Custom GapAssessor can change unknown aspects/terms, but cannot change any memory's classification
        class CustomAssessor:
            name = "custom-v1"

            def aspects(self, goal: str, query_terms: set[str]) -> list[Aspect]:
                return [Aspect(aspect="custom_sub_question", terms=tuple(sorted(query_terms)))]

        custom_res = self.brain.assess(
            "Use Ed25519 signatures and rotate keys",
            state=state_alice,
            assessor=CustomAssessor(),
        )
        self.assertEqual(custom_res["assessor"], "custom-v1")
        self.assertEqual(custom_res["uncertain"][0]["node_id"], alice_node_id)
        self.assertEqual(custom_res["unknown"][0]["aspect"], "custom_sub_question")


if __name__ == "__main__":
    unittest.main()
