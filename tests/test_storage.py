"""Tests for JSON/SQLite storage backends."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fibmind import (
    FibMind,
    JsonStore,
    MemoryScope,
    MemoryStatus,
    RelationType,
    SqliteStore,
    open_store,
)

# A minimal v1 database: one node carrying the old single ``importance`` column,
# one folded node rewritten in place the way v1 did it, and no event log.
V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE nodes (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
    category TEXT NOT NULL, node_type TEXT NOT NULL, layer TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, metadata_json TEXT NOT NULL,
    importance REAL NOT NULL, access_count INTEGER NOT NULL, memory_weight INTEGER NOT NULL
);
CREATE TABLE trees (
    id TEXT PRIMARY KEY, category TEXT NOT NULL, title TEXT NOT NULL,
    root_node_id TEXT NOT NULL, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE edges (
    id TEXT PRIMARY KEY, from_node_id TEXT NOT NULL, to_node_id TEXT NOT NULL,
    relation_type TEXT NOT NULL, weight REAL NOT NULL, direction TEXT NOT NULL,
    created_at TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE node_tags (node_id TEXT NOT NULL, tag TEXT NOT NULL, PRIMARY KEY (node_id, tag));
CREATE TABLE node_tree_memberships (
    node_id TEXT NOT NULL, tree_id TEXT NOT NULL, PRIMARY KEY (node_id, tree_id)
);
CREATE TABLE category_roots (category TEXT PRIMARY KEY, tree_id TEXT NOT NULL);
"""

_TS = "2026-01-01T00:00:00+00:00"


def _write_v1_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(V1_SCHEMA)
        connection.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        connection.execute("INSERT INTO meta VALUES ('revision', '3')")
        rows = [
            ("node_kept", "Kept", "still visible", "raw", 0.8),
            ("node_folded", "Folded", "absorbed but preserved", "folded:raw", 0.4),
            ("node_summary", "Summary", "- Folded: absorbed", "compressed", 0.4),
        ]
        for node_id, title, content, layer, importance in rows:
            connection.execute(
                "INSERT INTO nodes VALUES (?, ?, ?, 'code', 'raw', ?, ?, ?, '{}', ?, 7, 1)",
                (node_id, title, content, layer, _TS, _TS, importance),
            )
        connection.execute(
            "INSERT INTO edges VALUES "
            "('edge_1', 'node_summary', 'node_folded', 'summary_of', 0.8, 'directed', ?, '{}')",
            (_TS,),
        )
        connection.commit()
    finally:
        connection.close()


class SqliteMigrationTests(unittest.TestCase):
    def test_v1_importance_becomes_familiarity_not_confidence(self) -> None:
        """v1 mixed recall frequency into one trust number.

        That number was access-driven, so it maps onto familiarity; confidence
        has to start at zero because nothing external ever backed those values.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            _write_v1_database(path)

            restored = SqliteStore(path).load()

        kept = restored.nodes["node_kept"]
        self.assertEqual(kept.familiarity, 0.8)
        self.assertEqual(kept.confidence, 0.0)
        self.assertIsNone(kept.confidence_source)
        self.assertEqual(kept.scope, MemoryScope.PERSONAL)
        self.assertEqual(kept.status, MemoryStatus.ACTIVE)

    def test_v1_folded_nodes_recover_their_pointer(self) -> None:
        """v1 rewrote folded nodes' layer in place; recover it as a pointer."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            _write_v1_database(path)

            restored = SqliteStore(path).load()

        folded = restored.nodes["node_folded"]
        self.assertEqual(folded.layer, "raw")
        self.assertEqual(folded.folded_into, "node_summary")
        self.assertEqual(folded.content, "absorbed but preserved")

    def test_migration_is_idempotent_and_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            _write_v1_database(path)

            SqliteStore(path)
            store = SqliteStore(path)
            with store.transaction() as memory:
                new_id = memory.append("code", "Fresh", "written after migrating")

            restored = store.load()

        self.assertEqual(restored.nodes["node_kept"].familiarity, 0.8)
        self.assertIn(new_id, restored.nodes)
        self.assertTrue(restored.events)


class SqliteStoreTests(unittest.TestCase):
    def test_round_trip_preserves_complete_graph(self) -> None:
        memory = FibMind()
        first = memory.append(
            "代码",
            "登录故障",
            "令牌过期后返回 401",
            tags={"认证", "错误"},
            metadata={"priority": 1},
        )
        second = memory.append("需求", "刷新令牌", "自动刷新")
        memory.mark_stale(second, "superseded requirement")
        memory.link_nodes(first, second, RelationType.RELATED_TO, metadata={"source": "test"})

        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteStore(Path(tmp) / "memory.db")
            store.save(memory)
            restored = store.load()

        self.assertEqual(set(memory.nodes), set(restored.nodes))
        self.assertEqual(set(memory.edges), set(restored.edges))
        self.assertEqual(set(memory.trees), set(restored.trees))
        self.assertEqual(restored.nodes[first].tags, {"认证", "错误"})
        self.assertEqual(restored.nodes[first].metadata, {"priority": 1})
        self.assertEqual(restored.nodes[second].status, MemoryStatus.STALE)
        self.assertEqual(restored.nodes[second].status_reason, "superseded requirement")
        related = next(
            edge for edge in restored.edges.values() if edge.relation_type == RelationType.RELATED_TO
        )
        self.assertEqual(related.metadata, {"source": "test"})

    def test_transaction_rolls_back_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteStore(Path(tmp) / "memory.db")
            with self.assertRaises(RuntimeError):
                with store.transaction() as memory:
                    memory.append("code", "Bug", "content")
                    raise RuntimeError("stop")

            self.assertEqual(store.load().nodes, {})
            self.assertEqual(store.load().events, [])

    def test_json_store_reads_legacy_importance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            path.write_text(
                json.dumps(
                    {
                        "nodes": [
                            {
                                "id": "node_legacy",
                                "title": "Legacy",
                                "content": "written before the split",
                                "category": "code",
                                "node_type": "raw",
                                "layer": "raw",
                                "created_at": _TS,
                                "updated_at": _TS,
                                "importance": 0.9,
                                "access_count": 4,
                                "memory_weight": 1,
                            }
                        ],
                        "edges": [],
                        "trees": [],
                        "category_roots": {},
                    }
                ),
                encoding="utf-8",
            )

            restored = JsonStore(path).load()

        node = restored.nodes["node_legacy"]
        self.assertEqual(node.familiarity, 0.9)
        self.assertEqual(node.confidence, 0.0)

    def test_open_store_uses_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsInstance(open_store(Path(tmp) / "memory.json"), JsonStore)
            self.assertIsInstance(open_store(Path(tmp) / "memory.db"), SqliteStore)
            with self.assertRaises(ValueError):
                open_store(Path(tmp) / "memory.txt")


if __name__ == "__main__":
    unittest.main()
