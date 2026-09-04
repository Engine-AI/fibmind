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
        event = self.brain.observe("tool_result", "pytest passed", {"command": "pytest -q"})
        recalled = self.brain.recall("pytest passed", state=_state())

        self.assertEqual(event["kind"], "tool_result")
        self.assertEqual(recalled["hits"], [])
        self.assertEqual(recalled["episode"][0]["summary"], "pytest passed")

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
