"""Cross-implementation contract tests.

These pin two promises documented in ``docs/contract.md``:

1. The event log is versioned, and a store written at a newer version is
   refused rather than misread.
2. ``data/demo-brain.json`` is a conformance fixture: any implementation that
   replays its events must produce the same nodes, edges, trees, and the same
   answers to a fixed query set. The expected answers live in
   ``tests/fixtures/demo-brain.expected.json`` so a port in another language can
   run the same check without importing this package.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fibmind import LOG_VERSION, FibMind, JsonStore, MemoryEvent, SqliteStore
from fibmind.mcp_server import build_server
from fibmind.service import MemoryService

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "demo-brain.json"
EXPECTED = ROOT / "tests" / "fixtures" / "demo-brain.expected.json"


def _replayed_fixture() -> FibMind:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    events = [MemoryEvent.from_dict(item) for item in payload["events"]]
    return FibMind.rebuild_from_log(events)


def _answer(memory: FibMind, query: dict) -> list[str]:
    """Run one fixed query the way ``fibbrain_recall`` does: active, visible,
    episode scratch excluded."""
    hits = memory.search(
        query["query"],
        top_k=query.get("top_k", 5),
        owner=query.get("owner"),
        workspace_id=query.get("workspace_id"),
        project_id=query.get("project_id"),
        session_id=query.get("session_id"),
        exclude_categories=set(query.get("exclude_categories", ["episode"])),
    )
    return [hit.node.title for hit in hits]


class LogVersionTests(unittest.TestCase):
    def test_events_carry_the_log_version(self) -> None:
        memory = FibMind()
        memory.append("code", "Login 401", "token expiry causes 401 on login")
        for event in memory.events:
            self.assertEqual(event.to_dict()["log_version"], LOG_VERSION)

    def test_json_store_writes_and_reads_log_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.json"
            memory = FibMind()
            memory.append("code", "Login 401", "token expiry causes 401 on login")
            JsonStore(path).save(memory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["log_version"], LOG_VERSION)
            self.assertEqual(len(JsonStore(path).load().nodes), len(memory.nodes))

    def test_json_store_refuses_a_newer_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.json"
            path.write_text(json.dumps({"log_version": LOG_VERSION + 1, "nodes": []}))
            with self.assertRaises(ValueError):
                JsonStore(path).load()

    def test_sqlite_store_records_and_refuses_newer_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.db"
            SqliteStore(path)
            with sqlite3.connect(path) as connection:
                row = connection.execute(
                    "SELECT value FROM meta WHERE key = 'log_version'"
                ).fetchone()
                self.assertEqual(int(row[0]), LOG_VERSION)
                connection.execute(
                    "UPDATE meta SET value = ? WHERE key = 'log_version'",
                    (str(LOG_VERSION + 1),),
                )
            with self.assertRaises(ValueError):
                SqliteStore(path)

    def test_unversioned_store_is_read_as_version_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.json"
            memory = FibMind()
            memory.append("code", "Login 401", "token expiry causes 401 on login")
            payload = {
                "nodes": [node.to_dict() for node in memory.nodes.values()],
                "edges": [],
                "trees": [tree.to_dict() for tree in memory.trees.values()],
                "category_roots": memory.category_roots,
                "events": [event.to_dict() for event in memory.events],
            }
            path.write_text(json.dumps(payload))
            self.assertEqual(len(JsonStore(path).load().nodes), len(memory.nodes))


class DemoBrainFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

    def test_fixture_declares_its_log_version(self) -> None:
        self.assertEqual(self.payload["log_version"], self.expected["log_version"])
        self.assertLessEqual(self.payload["log_version"], LOG_VERSION)

    def test_replay_matches_materialized_state(self) -> None:
        replayed = _replayed_fixture()
        materialized = JsonStore(FIXTURE).load()
        self.assertEqual(set(replayed.nodes), set(materialized.nodes))
        self.assertEqual(set(replayed.edges), set(materialized.edges))
        self.assertEqual(set(replayed.trees), set(materialized.trees))
        self.assertEqual(replayed.category_roots, materialized.category_roots)
        for node_id, node in replayed.nodes.items():
            other = materialized.nodes[node_id]
            self.assertEqual(node.title, other.title)
            self.assertEqual(node.content, other.content)
            self.assertEqual(node.status, other.status)
            self.assertEqual(node.scope, other.scope)
            self.assertEqual(node.owner, other.owner)
            self.assertEqual(node.project_id, other.project_id)

    def test_replay_matches_expected_shape(self) -> None:
        replayed = _replayed_fixture()
        shape = self.expected["shape"]
        self.assertEqual(len(replayed.nodes), shape["nodes"])
        self.assertEqual(len(replayed.edges), shape["edges"])
        self.assertEqual(len(replayed.trees), shape["trees"])
        self.assertEqual(sorted(replayed.category_roots), shape["categories"])

    def test_fixed_queries_return_expected_titles(self) -> None:
        replayed = _replayed_fixture()
        for case in self.expected["queries"]:
            with self.subTest(query=case["query"], state=case.get("project_id")):
                self.assertEqual(_answer(replayed, case), case["expect_titles"])


class ToolSurfaceTests(unittest.TestCase):
    def test_mcp_tool_names_match_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = build_server(MemoryService(Path(tmp) / "store.db"))
            registered = set(server._tool_manager._tools)  # noqa: SLF001
        expected = set(json.loads(EXPECTED.read_text())["tools"])
        self.assertEqual(registered, expected)


if __name__ == "__main__":
    unittest.main()
