"""Tests for workspace / project / session / task identity isolation."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fibmind import FibMind, MemoryScope, RelationType, SqliteStore
from fibmind.context import build_context
from fibmind.graph import DEFAULT_PROMOTION_THRESHOLD

_TS = "2026-01-01T00:00:00+00:00"

V3_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE nodes (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
    category TEXT NOT NULL, node_type TEXT NOT NULL, layer TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, metadata_json TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'personal', owner TEXT,
    status TEXT NOT NULL DEFAULT 'active', status_reason TEXT,
    familiarity REAL NOT NULL DEFAULT 0.0, access_count INTEGER NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0, confidence_source TEXT,
    folded_into TEXT, memory_weight INTEGER NOT NULL
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
CREATE TABLE event_log (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL, op TEXT NOT NULL,
    payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


def _write_v3_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(V3_SCHEMA)
        connection.execute("INSERT INTO meta VALUES ('schema_version', '3')")
        connection.execute("INSERT INTO meta VALUES ('revision', '1')")
        connection.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, 'code', 'raw', 'raw', ?, ?, '{}', "
            "'personal', NULL, 'active', NULL, 0.0, 0, 0.0, NULL, NULL, 1)",
            ("node_legacy", "Legacy retry", "retry three times", _TS, _TS),
        )
        connection.commit()
    finally:
        connection.close()


class IdentityVisibilityTests(unittest.TestCase):
    def test_personal_memories_do_not_cross_projects(self) -> None:
        memory = FibMind()
        mine = memory.append(
            "decision",
            "Retry policy",
            "retry three times with backoff",
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
        )
        theirs = memory.append(
            "decision",
            "Retry policy",
            "fail immediately and never retry",
            owner="alice",
            workspace_id="ws-1",
            project_id="other-app",
        )

        visible = {
            hit.node.id
            for hit in memory.search(
                "retry policy",
                owner="alice",
                workspace_id="ws-1",
                project_id="fibmind",
            )
        }

        self.assertIn(mine, visible)
        self.assertNotIn(theirs, visible)

    def test_personal_memories_do_not_cross_workspaces(self) -> None:
        memory = FibMind()
        mine = memory.append(
            "prefs",
            "Editor preference",
            "prefers PyCharm",
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
        )
        theirs = memory.append(
            "prefs",
            "Editor preference",
            "prefers Neovim",
            owner="alice",
            workspace_id="ws-2",
            project_id="fibmind",
        )

        visible = {
            hit.node.id
            for hit in memory.search(
                "editor preference",
                owner="alice",
                workspace_id="ws-1",
                project_id="fibmind",
            )
        }

        self.assertIn(mine, visible)
        self.assertNotIn(theirs, visible)

    def test_personal_memories_are_visible_across_sessions(self) -> None:
        memory = FibMind()
        earlier = memory.append(
            "error",
            "SQLite migration refuse overwrite",
            "migration refuses to overwrite an existing SQLite file",
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
            session_id="sess-old",
        )

        visible = {
            hit.node.id
            for hit in memory.search(
                "sqlite migration overwrite",
                owner="alice",
                workspace_id="ws-1",
                project_id="fibmind",
                session_id="sess-new",
            )
        }

        self.assertIn(earlier, visible)

    def test_session_scratch_is_invisible_in_another_session(self) -> None:
        memory = FibMind()
        scratch = memory.append(
            "scratch",
            "Draft retry note",
            "maybe uploads should retry forever",
            scope=MemoryScope.SESSION,
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
            session_id="sess-old",
        )

        visible = {
            hit.node.id
            for hit in memory.search(
                "retry forever",
                owner="alice",
                workspace_id="ws-1",
                project_id="fibmind",
                session_id="sess-new",
            )
        }

        self.assertNotIn(scratch, visible)

    def test_omitting_workspace_hides_labelled_memories(self) -> None:
        memory = FibMind()
        labelled = memory.append(
            "decision",
            "Retry policy",
            "retry three times",
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
        )

        visible = {hit.node.id for hit in memory.search("retry policy", owner="alice")}

        self.assertNotIn(labelled, visible)

    def test_knowledge_ignores_owner_but_keeps_project_isolation(self) -> None:
        memory = FibMind()
        supporting = [
            memory.append(
                "math",
                f"Missed the discriminant #{index}",
                "skipped checking whether the discriminant is negative",
                owner="alice",
                workspace_id="ws-1",
                project_id="fibmind",
            )
            for index in range(DEFAULT_PROMOTION_THRESHOLD)
        ]
        claim = memory.promote_to_knowledge(
            title="Discriminant sign is routinely skipped",
            content="Students skip checking whether the discriminant is negative",
            supporting_node_ids=supporting,
        )

        same_project = {
            hit.node.id
            for hit in memory.search(
                "discriminant",
                owner="bob",
                workspace_id="ws-1",
                project_id="fibmind",
            )
        }
        other_project = {
            hit.node.id
            for hit in memory.search(
                "discriminant",
                owner="bob",
                workspace_id="ws-1",
                project_id="other-app",
            )
        }

        self.assertIn(claim, same_project)
        self.assertNotIn(claim, other_project)
        self.assertTrue(same_project.isdisjoint(set(supporting)))

    def test_unscoped_knowledge_stays_visible_to_labelled_callers(self) -> None:
        memory = FibMind()
        supporting = [
            memory.append(
                "math",
                f"Missed the discriminant #{index}",
                "skipped checking whether the discriminant is negative",
                owner="alice",
            )
            for index in range(DEFAULT_PROMOTION_THRESHOLD)
        ]
        claim = memory.promote_to_knowledge(
            title="Discriminant sign is routinely skipped",
            content="Students skip checking whether the discriminant is negative",
            supporting_node_ids=supporting,
        )

        visible = {
            hit.node.id
            for hit in memory.search(
                "discriminant",
                owner="bob",
                workspace_id="ws-1",
                project_id="fibmind",
            )
        }

        self.assertIn(claim, visible)

    def test_graph_expansion_cannot_cross_projects(self) -> None:
        memory = FibMind()
        left = memory.append(
            "decision",
            "Retry policy",
            "retry three times with backoff",
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
        )
        right = memory.append(
            "decision",
            "Retry policy",
            "fail immediately and never retry",
            owner="alice",
            workspace_id="ws-1",
            project_id="other-app",
        )
        memory.link_nodes(left, right, RelationType.RELATED_TO)

        pack = build_context(
            memory,
            "retry policy",
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
        )
        visible = {hit.node.id for hit in pack.hits}

        self.assertIn(left, visible)
        self.assertNotIn(right, visible)

    def test_auto_link_does_not_cross_projects(self) -> None:
        memory = FibMind()
        left = memory.append(
            "decision",
            "Retry policy for uploads",
            "retry three times with backoff before failing",
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
        )
        right = memory.append(
            "decision",
            "Retry policy for uploads",
            "retry three times with backoff before failing",
            owner="alice",
            workspace_id="ws-1",
            project_id="other-app",
        )

        linked = {
            (edge.from_node_id, edge.to_node_id)
            for edge in memory.edges.values()
            if edge.relation_type == RelationType.SIMILAR_TO
        }

        self.assertNotIn((left, right), linked)
        self.assertNotIn((right, left), linked)

    def test_rebuild_from_log_keeps_identity_fields(self) -> None:
        memory = FibMind()
        node_id = memory.append(
            "decision",
            "Retry policy",
            "retry three times",
            owner="alice",
            workspace_id="ws-1",
            project_id="fibmind",
            session_id="sess-1",
            task_id="task-9",
        )

        replayed = FibMind.rebuild_from_log(memory.events)
        node = replayed.nodes[node_id]

        self.assertEqual(node.workspace_id, "ws-1")
        self.assertEqual(node.project_id, "fibmind")
        self.assertEqual(node.session_id, "sess-1")
        self.assertEqual(node.task_id, "task-9")

    def test_session_scope_requires_session_id(self) -> None:
        memory = FibMind()

        with self.assertRaises(ValueError):
            memory.append(
                "scratch",
                "Draft",
                "temporary note",
                scope=MemoryScope.SESSION,
                owner="alice",
            )

    def test_promotion_rejects_mixed_project_supporters(self) -> None:
        memory = FibMind()
        supporting = [
            memory.append(
                "math",
                f"Missed the discriminant #{index}",
                "skipped checking whether the discriminant is negative",
                owner="alice",
                workspace_id="ws-1",
                project_id="fibmind" if index < 2 else "other-app",
            )
            for index in range(DEFAULT_PROMOTION_THRESHOLD)
        ]

        with self.assertRaises(ValueError):
            memory.promote_to_knowledge(
                title="Discriminant sign is routinely skipped",
                content="Students skip checking whether the discriminant is negative",
                supporting_node_ids=supporting,
            )

    def test_schema_v3_upgrades_identity_fields_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            _write_v3_database(path)

            store = SqliteStore(path)
            restored = store.load()
            node = restored.nodes["node_legacy"]

            self.assertIsNone(node.workspace_id)
            self.assertIsNone(node.project_id)
            self.assertIsNone(node.session_id)
            self.assertIsNone(node.task_id)
            hits = restored.search("legacy retry")
            self.assertEqual(hits[0].node.id, "node_legacy")
            self.assertGreaterEqual(store.SCHEMA_VERSION, 4)
            self.assertIsNone(node.memory_kind)


if __name__ == "__main__":
    unittest.main()
