"""Token budget and L0 hot memory.

A pack must never exceed its token budget, must say what it spent per
section and why anything was cut, and the hot preamble must stay frozen for a
session while the next session can see a refreshed one.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fibmind import (
    BrainState,
    CallableCounter,
    FibBrain,
    FibMind,
    MemoryService,
    build_context,
    estimate_tokens,
)
from fibmind.context import SOURCE_EXPAND, SOURCE_HOT, SOURCE_SEARCH

IDENT = dict(owner="a", workspace_id="w", project_id="p")


def _state(**over: str) -> BrainState:
    base = {"owner": "a", "workspace_id": "w", "project_id": "p", "session_id": "s1"}
    base.update(over)
    return BrainState(**base)


def _forest(n: int = 12) -> FibMind:
    memory = FibMind()
    for i in range(n):
        memory.append(
            "code",
            f"Login retry note {i}",
            f"Attempt {i}: the login endpoint retried the token refresh and logged a 401 with trace id {i:04d}. " * 2,
            **IDENT,
        )
    return memory


class BudgetTests(unittest.TestCase):
    def test_pack_never_exceeds_the_token_budget(self) -> None:
        memory = _forest()
        for budget in (10, 40, 120, 400):
            pack = build_context(memory, "login retry token 401", top_k=10, depth=1, budget_tokens=budget, **IDENT)
            self.assertLessEqual(estimate_tokens(pack.text), budget, f"budget {budget}")
            self.assertEqual(pack.budget.budget_tokens, budget)
            self.assertLessEqual(pack.budget.used_tokens, budget)

    def test_budget_report_names_sections_and_truncation(self) -> None:
        memory = _forest()
        pack = build_context(memory, "login retry token 401", top_k=10, depth=1, budget_tokens=60, **IDENT)
        report = pack.budget.to_dict()
        self.assertIn(SOURCE_SEARCH, report["sections"])
        self.assertGreater(report["sections"][SOURCE_SEARCH]["offered"], report["sections"][SOURCE_SEARCH]["kept"])
        reasons = {row["reason"] for row in report["truncated"]}
        self.assertTrue(reasons & {"over_budget", "excerpt_shortened"})
        self.assertEqual(report["counter"], "estimate")
        self.assertEqual(report["remaining_tokens"], report["budget_tokens"] - report["used_tokens"])

    def test_generous_budget_keeps_everything(self) -> None:
        memory = _forest(3)
        pack = build_context(memory, "login retry token 401", top_k=10, depth=1, budget_tokens=5000, **IDENT)
        self.assertEqual(pack.budget.truncated, [])
        self.assertEqual(sum(section["kept"] for section in pack.budget.sections.values()), len(pack.hits))

    def test_max_chars_still_bounds_when_no_budget_given(self) -> None:
        memory = _forest()
        pack = build_context(memory, "login retry token 401", top_k=10, max_chars=80, **IDENT)
        self.assertLessEqual(len(pack.text), 80)
        self.assertEqual(pack.budget.budget_tokens, 20)

    def test_custom_counter_is_used_and_reported(self) -> None:
        memory = _forest(4)
        chars = CallableCounter(name="chars", fn=len)
        pack = build_context(memory, "login retry", top_k=4, budget_tokens=200, counter=chars, **IDENT)
        self.assertLessEqual(len(pack.text), 200)
        self.assertEqual(pack.budget.counter, "chars")

    def test_hot_section_renders_first_and_only_once(self) -> None:
        memory = _forest(4)
        hot_node = next(iter(n for n in memory.nodes.values() if n.title == "Login retry note 0"))
        pack = build_context(memory, "login retry note 0", top_k=4, budget_tokens=400, hot=[hot_node], **IDENT)
        self.assertTrue(pack.text.startswith("- [code] Login retry note 0"))
        self.assertEqual(pack.text.count("Login retry note 0:"), 1)
        self.assertEqual([hit.source for hit in pack.hot], [SOURCE_HOT])
        self.assertIn(hot_node.id, [hit.node.id for hit in pack.hits])
        self.assertEqual(pack.budget.sections[SOURCE_HOT]["kept"], 1)

    def test_service_drops_invisible_hot_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(Path(tmp) / "m.db")
            mine = service.append("preference", "Tabs", "use tabs", owner="a", workspace_id="w", project_id="p")
            theirs = service.append("preference", "Spaces", "use spaces", owner="b", workspace_id="w", project_id="p")
            pack = service.context("indentation", hot_node_ids=[mine["node_id"], theirs["node_id"]], budget_tokens=200, **IDENT)
            self.assertEqual([hit["node_id"] for hit in pack["hot"]], [mine["node_id"]])
            self.assertNotIn("Spaces", pack["text"])


class HotSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.brain = FibBrain(MemoryService(Path(self._tmp.name) / "memory.db"))
        self.pref = self.brain.remember("preference", "Editor", "PyCharm for Python work", state=_state())
        self.rule = self.brain.remember("requirement", "Tests before commit", "run pytest before every commit", state=_state())
        self.brain.remember("error", "Flaky upload", "upload test flaked once on CI", state=_state())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_snapshot_selects_hot_kinds_only(self) -> None:
        snap = self.brain.hot_snapshot(_state())
        self.assertEqual(set(snap["node_ids"]), {self.pref["node_id"], self.rule["node_id"]})
        self.assertTrue(snap["frozen"])
        self.assertIsNotNone(snap["snapshot_id"])

    def test_snapshot_is_frozen_within_a_session(self) -> None:
        first = self.brain.hot_snapshot(_state())
        self.brain.remember("preference", "Shell", "zsh with starship", state=_state())
        second = self.brain.hot_snapshot(_state())
        self.assertEqual(first["node_ids"], second["node_ids"])
        self.assertEqual(first["snapshot_id"], second["snapshot_id"])

        refreshed = self.brain.hot_snapshot(_state(), refresh=True)
        self.assertEqual(len(refreshed["node_ids"]), 3)
        self.assertNotEqual(refreshed["snapshot_id"], first["snapshot_id"])

    def test_next_session_gets_a_fresh_snapshot(self) -> None:
        self.brain.hot_snapshot(_state())
        self.brain.remember("preference", "Shell", "zsh with starship", state=_state())
        nxt = self.brain.hot_snapshot(_state(session_id="s2"))
        self.assertEqual(len(nxt["node_ids"]), 3)

    def test_evidence_orders_the_snapshot(self) -> None:
        self.brain.reflect("confirmed", "user said so", node_id=self.rule["node_id"], state=_state())
        snap = self.brain.hot_snapshot(_state(session_id="s3"))
        self.assertEqual(snap["node_ids"][0], self.rule["node_id"])

    def test_recall_renders_hot_first_within_budget_and_hides_snapshot_nodes(self) -> None:
        pack = self.brain.recall("upload test", state=_state(), budget_tokens=150)
        self.assertTrue(pack["text"].startswith("- [preference]") or pack["text"].startswith("- [requirement]"))
        self.assertIn("Flaky upload", pack["text"])
        self.assertLessEqual(pack["budget"]["used_tokens"], 150)
        self.assertLessEqual(pack["budget"]["sections"]["hot"]["used"], 50)
        self.assertFalse(any(hit["category"] == "hot" for hit in pack["hits"]))

    def test_recall_can_skip_hot(self) -> None:
        pack = self.brain.recall("upload test", state=_state(), include_hot=False)
        self.assertEqual(pack["hot"], [])
        self.assertEqual(pack["budget"]["sections"]["hot"]["budget"], 0)

    def test_snapshot_does_not_cross_identity(self) -> None:
        other = self.brain.hot_snapshot(_state(owner="b"))
        self.assertEqual(other["node_ids"], [])
        other_project = self.brain.hot_snapshot(_state(project_id="q"))
        self.assertEqual(other_project["node_ids"], [])

    def test_review_reports_next_hot(self) -> None:
        self.brain.observe("decision", "keep PyCharm", state=_state())
        report = self.brain.review_session(_state(), mode="auto")
        self.assertIsInstance(report["next_hot"], list)
        self.assertIn(self.pref["node_id"], report["next_hot"])


if __name__ == "__main__":
    unittest.main()
