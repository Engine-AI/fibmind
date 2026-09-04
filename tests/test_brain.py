"""Tests for the FibBrain cognitive protocol."""

import tempfile
import unittest
from pathlib import Path

from fibmind import AdmitVerdict, AdviseVerdict, BrainState, FibBrain, GoalStatus, MemoryService, Verdict
from fibmind.models import MemoryStatus


def _state(**overrides: str) -> BrainState:
    base = {
        "owner": "alice",
        "workspace_id": "ws-1",
        "project_id": "fibmind",
        "session_id": "sess-1",
    }
    base.update(overrides)
    return BrainState(**base)


def _review_hits(pack: dict) -> list[dict]:
    """Hits written by session review; goals and hand-written memories are ignored."""
    return [hit for hit in pack["hits"] if hit["category"] in {"summary", "verification", "code", "decision", "error", "correction", "risk"}]


class FibBrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.brain = FibBrain(MemoryService(Path(self._tmp.name) / "memory.json"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_remember_writes_an_admitted_memory(self) -> None:
        result = self.brain.remember(
            "decision",
            "Retry policy",
            "retry three times with backoff",
            state=_state(),
        )

        self.assertEqual(result["verdict"], AdmitVerdict.WRITE.value)
        self.assertIn("node_id", result)
        recalled = self.brain.recall("retry policy", state=_state())
        self.assertIn("Retry policy", recalled["text"])
        self.assertEqual(recalled["state"]["project_id"], "fibmind")

    def test_remember_skips_duplicates(self) -> None:
        first = self.brain.remember(
            "decision",
            "Retry policy",
            "retry three times with backoff",
            state=_state(),
        )
        second = self.brain.remember(
            "decision",
            "Retry policy",
            "retry three times with backoff",
            state=_state(),
        )

        self.assertEqual(first["verdict"], AdmitVerdict.WRITE.value)
        self.assertEqual(second["verdict"], AdmitVerdict.SKIP.value)
        self.assertEqual(second["duplicate_node_id"], first["node_id"])
        self.assertEqual(self.brain.recall("retry policy", state=_state())["hits"][0]["node_id"], first["node_id"])

    def test_remember_skips_raw_dumps(self) -> None:
        dump = "\n".join(f"ERROR step {index} failed" for index in range(90))
        result = self.brain.remember("code", "Build log", dump, state=_state())

        self.assertEqual(result["verdict"], AdmitVerdict.SKIP.value)
        self.assertNotIn("node_id", result)
        self.assertEqual(self.brain.recall("build log failed", state=_state())["hits"], [])

    def test_remember_skips_empty_speculation(self) -> None:
        result = self.brain.remember(
            "scratch",
            "Guess",
            "maybe the cache is wrong",
            state=_state(),
        )

        self.assertEqual(result["verdict"], AdmitVerdict.SKIP.value)

    def test_observe_stays_in_the_episode_not_the_store(self) -> None:
        event = self.brain.observe("tool_result", "pytest passed", {"command": "pytest -q"}, state=_state())
        recalled = self.brain.recall("pytest passed", state=_state())

        self.assertEqual(event["kind"], "tool_result")
        self.assertEqual(recalled["hits"], [])
        self.assertEqual(recalled["episode"][0]["summary"], "pytest passed")
        self.assertEqual(recalled["episode"][0]["node_id"], event["node_id"])

    def test_observe_without_a_session_stays_in_process(self) -> None:
        event = self.brain.observe("note", "no session here")
        self.assertIsNone(event["node_id"])
        recalled = self.brain.recall("no session here", state=BrainState(owner="alice"))
        self.assertEqual(recalled["hits"], [])
        self.assertEqual([item["summary"] for item in recalled["episode"]], ["no session here"])

    def test_episode_survives_a_restart_and_stays_in_its_session(self) -> None:
        store = Path(self._tmp.name) / "memory.json"
        self.brain.observe("test", "142 passed", {"command": ".venv/bin/pytest -q"}, state=_state())

        reopened = FibBrain(MemoryService(store))
        same_session = reopened.recall("anything", state=_state())
        other_session = reopened.recall("anything", state=_state(session_id="sess-2"))

        self.assertEqual([item["summary"] for item in same_session["episode"]], ["142 passed"])
        self.assertEqual(other_session["episode"], [])

    def _observe_a_session(self) -> None:
        state = _state()
        self.brain.plan("Fix flaky upload retry test", state=state)
        self.brain.observe("decision", "cap retry backoff at 8 seconds so the test finishes under the CI timeout", state=state)
        self.brain.observe("tool_result", "edited files", {"files": ["src/upload.py", "tests/test_upload.py"]}, state=state)
        self.brain.observe("test", "142 passed", {"command": ".venv/bin/pytest -q"}, state=state)
        self.brain.observe("error", "first attempt timed out at 30s", state=state)

    def test_review_session_candidates_mode_writes_nothing(self) -> None:
        self._observe_a_session()
        before = self.brain.recall("upload retry backoff", state=_state(session_id="sess-2"))["hits"]

        report = self.brain.review_session(_state(), mode="candidates")
        after = self.brain.recall("upload retry backoff", state=_state(session_id="sess-2"))["hits"]

        self.assertEqual(report["objective"], "Fix flaky upload retry test")
        categories = sorted(item["category"] for item in report["written"])
        self.assertEqual(categories, ["code", "decision", "error", "summary", "verification"])
        self.assertTrue(all("node_id" not in item for item in report["written"]))
        self.assertEqual(before, after)

    def test_review_session_approve_mode_holds_memories_until_approved(self) -> None:
        self._observe_a_session()
        report = self.brain.review_session(_state(), mode="approve")
        next_session = _state(session_id="sess-2")

        self.assertTrue(all(item["status"] == MemoryStatus.PENDING.value for item in report["written"]))
        self.assertEqual(_review_hits(self.brain.recall("upload retry backoff CI timeout", state=next_session)), [])
        pending = self.brain.pending_reviews(_state())
        self.assertEqual(len(pending), len(report["written"]))

        summary = next(item for item in report["written"] if item["category"] == "summary")
        error = next(item for item in report["written"] if item["category"] == "error")
        self.brain.approve_memory(summary["node_id"])
        self.brain.reject_memory(error["node_id"], "not worth keeping")

        hits = _review_hits(self.brain.recall("upload retry backoff CI timeout", state=next_session))
        self.assertEqual([hit["node_id"] for hit in hits], [summary["node_id"]])
        timed_out = _review_hits(self.brain.recall("attempt timed out", state=next_session))
        self.assertNotIn(error["node_id"], [hit["node_id"] for hit in timed_out])
        with self.assertRaises(ValueError):
            self.brain.approve_memory(summary["node_id"])

    def test_review_session_auto_mode_is_idempotent(self) -> None:
        self._observe_a_session()
        first = self.brain.review_session(_state(), mode="auto")
        count_after_first = len(self.brain.recall("upload retry", state=_state(session_id="sess-2"), top_k=20)["hits"])

        second = self.brain.review_session(_state(), mode="auto")
        count_after_second = len(self.brain.recall("upload retry", state=_state(session_id="sess-2"), top_k=20)["hits"])

        self.assertGreater(len(first["written"]), 0)
        self.assertEqual(second["written"], [])
        self.assertEqual(len(second["skipped"]), len(first["written"]))
        self.assertTrue(all("duplicate" in item["admit"]["reason"] for item in second["skipped"]))
        self.assertEqual(count_after_first, count_after_second)

    def test_review_session_close_retires_the_episode(self) -> None:
        self._observe_a_session()
        report = self.brain.review_session(_state(), mode="auto", close=True)

        self.assertGreater(report["closed_episodes"], 0)
        self.assertEqual(self.brain.recall("anything", state=_state())["episode"], [])
        again = self.brain.review_session(_state(), mode="candidates")
        self.assertEqual(again["episode_count"], 0)

    def test_review_memories_are_recalled_next_session_but_episode_is_not(self) -> None:
        self._observe_a_session()
        self.brain.review_session(_state(), mode="auto")
        next_session = _state(session_id="sess-2")

        hits = self.brain.recall("which command verified the upload retry fix", state=next_session)["hits"]
        self.assertTrue(any(hit["category"] == "verification" for hit in hits))
        self.assertFalse(any(hit["category"] == "episode" for hit in hits))
        self.assertEqual(self.brain.review_session(next_session, mode="candidates")["candidates"], 0)

    def test_advise_allows_unknown_actions(self) -> None:
        decision = self.brain.advise("web_search", state=_state())

        self.assertEqual(decision["verdict"], AdviseVerdict.ALLOW.value)
        self.assertEqual(decision["memories"], [])

    def test_advise_rejects_well_evidenced_refutations(self) -> None:
        written = self.brain.remember(
            "decision",
            "Use web_search for the private API",
            "web_search is the right way to find this private API",
            state=_state(),
        )
        self.brain.reflect(
            Verdict.REFUTED.value,
            "user correction: public search cannot see the private API",
            node_id=written["node_id"],
        )

        decision = self.brain.advise("web_search", state=_state())

        self.assertEqual(decision["verdict"], AdviseVerdict.REJECT.value)
        self.assertGreaterEqual(len(decision["memories"]), 1)

    def test_reflect_can_resolve_a_node_from_the_goal(self) -> None:
        written = self.brain.remember(
            "error",
            "SQLite migration refuse overwrite",
            "migration refuses to overwrite an existing SQLite file",
            state=_state(),
        )

        updated = self.brain.reflect(
            Verdict.CONFIRMED.value,
            "pytest tests/test_migrate.py",
            goal="sqlite migration overwrite",
            state=_state(),
        )

        self.assertEqual(updated["node_id"], written["node_id"])
        self.assertGreater(updated["confidence"], 0.0)
        self.assertEqual(updated["status"], MemoryStatus.ACTIVE.value)

    def test_recall_does_not_cross_projects(self) -> None:
        self.brain.remember(
            "decision",
            "Retry policy",
            "retry three times with backoff",
            state=_state(project_id="fibmind"),
        )
        self.brain.remember(
            "decision",
            "Retry policy",
            "fail immediately and never retry",
            state=_state(project_id="other-app"),
        )

        pack = self.brain.recall("retry policy", state=_state(project_id="fibmind"))
        titles = {hit["title"] for hit in pack["hits"]}
        excerpts = {hit["excerpt"] for hit in pack["hits"]}

        self.assertIn("Retry policy", titles)
        self.assertTrue(any("three times" in excerpt for excerpt in excerpts))
        self.assertFalse(any("never retry" in excerpt for excerpt in excerpts))

    def test_plan_persists_a_goal_and_is_idempotent(self) -> None:
        first = self.brain.plan("Fix login 401 after token expiry", state=_state())
        second = self.brain.plan("Fix login 401 after token expiry", state=_state())

        self.assertEqual(first["goal_id"], second["goal_id"])
        self.assertEqual(first["status"], GoalStatus.ACTIVE.value)
        titles = [step["title"] for step in first["steps"]]
        self.assertEqual(titles[0], "Recall related history")
        self.assertTrue(any(step["title"].startswith("Do the work:") for step in first["steps"]))
        self.assertIn("code", first["capabilities"])
        inspected = self.brain.memory.inspect(first["goal_id"])
        self.assertIn("fibbrain-goal", inspected["tags"])

    def test_plan_adds_avoid_step_from_refuted_memory(self) -> None:
        written = self.brain.remember(
            "decision",
            "Use pytest for the private API",
            "pytest is the right way to exercise this private API",
            state=_state(),
        )
        self.brain.reflect(
            Verdict.REFUTED.value,
            "user correction: the private API is not covered by pytest",
            node_id=written["node_id"],
        )

        planned = self.brain.plan("Verify the private API with pytest", state=_state())
        avoid = [step["title"] for step in planned["steps"] if step["title"].startswith("Avoid:")]

        self.assertTrue(any("pytest" in title.lower() for title in avoid))

    def test_coordinate_blocks_a_refuted_capability(self) -> None:
        written = self.brain.remember(
            "decision",
            "Use pytest for the private API",
            "pytest is the right way to exercise this private API",
            state=_state(),
        )
        self.brain.reflect(
            Verdict.REFUTED.value,
            "user correction: the private API is not covered by pytest",
            node_id=written["node_id"],
        )

        coordinated = self.brain.coordinate(
            "Verify the private API with pytest",
            state=_state(),
        )
        test_item = next(item for item in coordinated["capabilities"] if item["capability"] == "test")

        self.assertTrue(coordinated["blocked"])
        self.assertEqual(coordinated["status"], GoalStatus.BLOCKED.value)
        self.assertEqual(test_item["advise"]["verdict"], AdviseVerdict.REJECT.value)
        self.assertEqual(test_item["action"], "pytest")

    def test_coordinate_starts_with_memory(self) -> None:
        coordinated = self.brain.coordinate("Document the identity model", state=_state())
        names = [item["capability"] for item in coordinated["capabilities"]]

        self.assertEqual(names[0], "memory")
        self.assertIn("docs", names)
        self.assertFalse(coordinated["blocked"])

    def test_complete_goal_keeps_the_node_recallable(self) -> None:
        planned = self.brain.plan("Ship the FibBrain protocol", state=_state())
        completed = self.brain.complete_goal(goal_id=planned["goal_id"], state=_state())
        recalled = self.brain.recall("FibBrain protocol", state=_state())

        self.assertEqual(completed["status"], GoalStatus.COMPLETE.value)
        self.assertTrue(any(hit["node_id"] == planned["goal_id"] for hit in recalled["hits"]))

    def test_plan_does_not_reuse_a_goal_from_another_project(self) -> None:
        first = self.brain.plan("Retry policy work", state=_state(project_id="fibmind"))
        second = self.brain.plan("Retry policy work", state=_state(project_id="other-app"))

        self.assertNotEqual(first["goal_id"], second["goal_id"])


if __name__ == "__main__":
    unittest.main()
