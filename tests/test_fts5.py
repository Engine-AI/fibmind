"""Unit tests for SQLite FTS5 incremental full-text indexing and search."""

import sqlite3
import tempfile
import unittest

from fibmind import FibMind, MemoryNode, NodeType, SqliteStore
from fibmind.service import MemoryService


class Fts5StorageTests(unittest.TestCase):
    def test_schema_version_is_six(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = SqliteStore(f.name)
            self.assertGreaterEqual(store.SCHEMA_VERSION, 6)
            with store._connect() as conn:
                version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
                self.assertEqual(version[0], str(store.SCHEMA_VERSION))
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertIn("nodes_fts", tables)

    def test_fts5_incremental_insert_update_delete(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = SqliteStore(f.name)
            node = MemoryNode(
                id="mem_1",
                title="Distributed Transaction Lock",
                content="How to avoid deadlocks in two-phase commit protocols",
                category="architecture",
                node_type=NodeType.CONCEPT,
                tags=("distributed", "lock", "transaction"),
            )
            forest = FibMind()
            forest.nodes[node.id] = node
            store.save(forest)

            # Query via search_fts
            hits = store.search_fts("deadlocks")
            self.assertIn("mem_1", hits)

            hits_tag = store.search_fts("distributed")
            self.assertIn("mem_1", hits_tag)

            # Update node
            node2 = MemoryNode(
                id="mem_1",
                title="Distributed Consensus Protocol",
                content="Raft algorithm leader election mechanism",
                category="consensus",
                node_type=NodeType.CONCEPT,
                tags=("raft", "consensus"),
            )
            forest2 = FibMind()
            forest2.nodes[node2.id] = node2
            store.save(forest2)

            # Old keywords should no longer match
            self.assertEqual(store.search_fts("deadlocks"), [])
            # New keywords should match
            self.assertIn("mem_1", store.search_fts("raft"))
            self.assertIn("mem_1", store.search_fts("consensus"))

            # Delete node
            forest3 = FibMind()
            store.save(forest3)
            self.assertEqual(store.search_fts("raft"), [])

    def test_fts5_cjk_support_and_special_chars(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = SqliteStore(f.name)
            node = MemoryNode(
                id="mem_cjk",
                title="SQLite 并发锁死处理策略",
                content="通过开启 WAL 模式与增加 busy_timeout 解决多进程争用",
                category="存储优化",
                node_type=NodeType.CONCEPT,
                tags=("数据库", "锁优化"),
            )
            forest = FibMind()
            forest.nodes[node.id] = node
            store.save(forest)

            # CJK searches
            self.assertIn("mem_cjk", store.search_fts("并发锁死"))
            self.assertIn("mem_cjk", store.search_fts("争用"))
            self.assertIn("mem_cjk", store.search_fts("存储优化"))
            self.assertIn("mem_cjk", store.search_fts("数据库"))

            # Special characters in query should not crash
            results = store.search_fts("WAL: busy_timeout? (锁优化) *")
            self.assertIn("mem_cjk", results)

    def test_migration_from_v5_backfills_fts5(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            # Manually create a v5 database without nodes_fts
            conn = sqlite3.connect(f.name)
            conn.executescript(
                """
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO meta(key, value) VALUES ('schema_version', '5');
                INSERT INTO meta(key, value) VALUES ('log_version', '1');
                INSERT INTO meta(key, value) VALUES ('revision', '1');
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
                    category TEXT NOT NULL, node_type TEXT NOT NULL, layer TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, metadata_json TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'personal', owner TEXT, workspace_id TEXT,
                    project_id TEXT, session_id TEXT, task_id TEXT, status TEXT NOT NULL DEFAULT 'active',
                    status_reason TEXT, memory_kind TEXT, familiarity REAL NOT NULL DEFAULT 0.0,
                    access_count INTEGER NOT NULL, confidence REAL NOT NULL DEFAULT 0.0,
                    confidence_source TEXT, folded_into TEXT, memory_weight INTEGER NOT NULL
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
                CREATE TABLE node_tags (node_id TEXT NOT NULL, tag TEXT NOT NULL, PRIMARY KEY(node_id, tag));
                CREATE TABLE node_tree_memberships (node_id TEXT NOT NULL, tree_id TEXT NOT NULL, PRIMARY KEY(node_id, tree_id));
                CREATE TABLE category_roots (category TEXT PRIMARY KEY, tree_id TEXT NOT NULL);
                CREATE TABLE event_log (id TEXT PRIMARY KEY, seq INTEGER NOT NULL, op TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL);

                INSERT INTO nodes(id, title, content, category, node_type, layer, created_at, updated_at, metadata_json, access_count, memory_weight)
                VALUES ('legacy_1', 'Legacy System Architecture', 'Microservices with gRPC communication', 'arch', 'factual', 'core', '2025-01-01T00:00:00', '2025-01-01T00:00:00', '{}', 1, 1);
                INSERT INTO node_tags(node_id, tag) VALUES ('legacy_1', 'microservices');
                """
            )
            conn.commit()
            conn.close()

            # Now open with SqliteStore - triggers _upgrade
            store = SqliteStore(f.name)
            self.assertGreaterEqual(store.SCHEMA_VERSION, 6)

            # nodes_fts must be populated with legacy_1
            hits = store.search_fts("microservices")
            self.assertIn("legacy_1", hits)
            hits_grpc = store.search_fts("gRPC")
            self.assertIn("legacy_1", hits_grpc)

    def test_memory_service_search_fts(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            service = MemoryService(f.name)
            node = MemoryNode(
                id="mem_svc",
                title="Service Layer Routing",
                content="Routes FTS queries directly to SqliteStore",
                category="service",
                node_type=NodeType.CONCEPT,
            )

            def _add(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_add)

            hits = service.search_fts("routing")
            self.assertIn("mem_svc", hits)


if __name__ == "__main__":
    unittest.main()
