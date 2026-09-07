"""Persistence helpers for FibMind."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from fibmind.graph import FibMind
from fibmind.models import LOG_VERSION, Edge, EventOp, MemoryEvent, MemoryNode, MemoryTree


class MemoryStore(Protocol):
    """Common persistence interface used by :class:`MemoryService`."""

    path: Path

    def load(self) -> FibMind: ...

    def save(self, memory: FibMind) -> None: ...

    def transaction(self) -> Iterator[FibMind]: ...


class JsonStore:
    """Atomic JSON persistence retained for demos and compatibility."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, memory: FibMind) -> None:
        payload: dict[str, Any] = {
            "log_version": LOG_VERSION,
            "nodes": [node.to_dict() for node in memory.nodes.values()],
            "edges": [edge.to_dict() for edge in memory.edges.values()],
            "trees": [tree.to_dict() for tree in memory.trees.values()],
            "category_roots": memory.category_roots,
            "events": [event.to_dict() for event in memory.events],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        self._atomic_write(serialized)

    def _atomic_write(self, serialized: str) -> None:
        fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def load(self) -> FibMind:
        if not self.path.exists():
            return FibMind()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        _check_log_version(payload.get("log_version"))
        memory = FibMind()
        memory.nodes = {item["id"]: MemoryNode.from_dict(item) for item in payload.get("nodes", [])}
        memory.edges = {item["id"]: Edge.from_dict(item) for item in payload.get("edges", [])}
        memory.trees = {item["id"]: MemoryTree.from_dict(item) for item in payload.get("trees", [])}
        memory.category_roots = dict(payload.get("category_roots", {}))
        memory.events = [MemoryEvent.from_dict(item) for item in payload.get("events", [])]
        memory.rebuild_indices()
        return memory

    @contextmanager
    def transaction(self) -> Iterator[FibMind]:
        memory = self.load()
        yield memory
        self.save(memory)


class SqliteStore:
    """Transactional SQLite persistence for the complete memory graph."""

    SCHEMA_VERSION = 5

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'personal',
                    owner TEXT,
                    workspace_id TEXT,
                    project_id TEXT,
                    session_id TEXT,
                    task_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    status_reason TEXT,
                    memory_kind TEXT,
                    familiarity REAL NOT NULL DEFAULT 0.0,
                    access_count INTEGER NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    confidence_source TEXT,
                    folded_into TEXT,
                    memory_weight INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trees (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    root_node_id TEXT NOT NULL REFERENCES nodes(id),
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    from_node_id TEXT NOT NULL REFERENCES nodes(id),
                    to_node_id TEXT NOT NULL REFERENCES nodes(id),
                    relation_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    direction TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS node_tags (
                    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (node_id, tag)
                );
                CREATE TABLE IF NOT EXISTS node_tree_memberships (
                    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
                    PRIMARY KEY (node_id, tree_id)
                );
                CREATE TABLE IF NOT EXISTS category_roots (
                    category TEXT PRIMARY KEY,
                    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE
                );
                -- The log is the source of truth; the tables above are a cache
                -- that FibMind.rebuild_from_log can regenerate. Rows are only
                -- ever appended, except for redaction on forget.
                CREATE TABLE IF NOT EXISTS event_log (
                    id TEXT PRIMARY KEY,
                    seq INTEGER NOT NULL,
                    op TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )
            connection.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('revision', '0')")
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('log_version', ?)",
                (str(LOG_VERSION),),
            )
            # Indices come after the upgrade: on a v1 database the columns they
            # cover do not exist until the ALTER TABLEs have run.
            self._upgrade(connection)
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_nodes_category_layer
                    ON nodes(category, layer);
                CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
                CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(scope, owner);
                CREATE INDEX IF NOT EXISTS idx_nodes_workspace
                    ON nodes(workspace_id, project_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
                CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(memory_kind);
                CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node_id);
                CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_node_id);
                CREATE INDEX IF NOT EXISTS idx_node_tags_tag ON node_tags(tag);
                CREATE INDEX IF NOT EXISTS idx_event_log_seq ON event_log(seq);
                """
            )
            version = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if version is None or int(version["value"]) != self.SCHEMA_VERSION:
                raise ValueError("Unsupported FibMind SQLite schema version")
            log_version = connection.execute(
                "SELECT value FROM meta WHERE key = 'log_version'"
            ).fetchone()
            _check_log_version(int(log_version["value"]) if log_version else None)

    def _upgrade(self, connection: sqlite3.Connection) -> None:
        """Bring older databases up to the current schema.

        v1 stored a single ``importance`` column mixing recall frequency with
        trust. It maps onto ``familiarity``; ``confidence`` starts at zero
        because nothing external ever backed those values. v3 adds lifecycle
        status so stale/refuted memories remain auditable without entering
        normal recall. v4 adds workspace / project / session / task identity
        so recall can isolate one working context from another. v5 adds
        ``memory_kind`` so procedural memories can be told from declarative
        ones; older rows keep ``NULL``.
        """
        row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None or int(row["value"]) >= self.SCHEMA_VERSION:
            return

        columns = {
            info["name"] for info in connection.execute("PRAGMA table_info(nodes)")
        }
        additions = {
            "scope": "TEXT NOT NULL DEFAULT 'personal'",
            "owner": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "status_reason": "TEXT",
            "familiarity": "REAL NOT NULL DEFAULT 0.0",
            "confidence": "REAL NOT NULL DEFAULT 0.0",
            "confidence_source": "TEXT",
            "folded_into": "TEXT",
            "workspace_id": "TEXT",
            "project_id": "TEXT",
            "session_id": "TEXT",
            "task_id": "TEXT",
            "memory_kind": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE nodes ADD COLUMN {name} {definition}")
        if "importance" in columns:
            connection.execute("UPDATE nodes SET familiarity = importance")
            # It has to go, not just stop being read: the v1 column is NOT NULL
            # without a default, so leaving it in place makes every subsequent
            # insert fail.
            connection.execute("ALTER TABLE nodes DROP COLUMN importance")
        # Nodes folded by v1 were rewritten in place; recover the pointer from
        # the layer marker so their originals become visible again.
        connection.execute(
            """
            UPDATE nodes SET folded_into = (
                SELECT e.from_node_id FROM edges e
                WHERE e.to_node_id = nodes.id AND e.relation_type = 'summary_of'
                LIMIT 1
            )
            WHERE layer LIKE 'folded:%'
            """
        )
        connection.execute(
            "UPDATE nodes SET layer = REPLACE(layer, 'folded:', '') WHERE layer LIKE 'folded:%'"
        )
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(self.SCHEMA_VERSION),),
        )

    def load(self) -> FibMind:
        with self._connect() as connection:
            return self._load(connection)

    def save(self, memory: FibMind) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._write(connection, memory)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    @contextmanager
    def transaction(self) -> Iterator[FibMind]:
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            memory = self._load(connection)
            yield memory
            self._write(connection, memory)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _load(self, connection: sqlite3.Connection) -> FibMind:
        tag_map: dict[str, set[str]] = {}
        for row in connection.execute("SELECT node_id, tag FROM node_tags"):
            tag_map.setdefault(row["node_id"], set()).add(row["tag"])

        tree_map: dict[str, set[str]] = {}
        for row in connection.execute("SELECT node_id, tree_id FROM node_tree_memberships"):
            tree_map.setdefault(row["node_id"], set()).add(row["tree_id"])

        memory = FibMind()
        memory.nodes = {}
        for row in connection.execute("SELECT * FROM nodes"):
            data = dict(row)
            data["metadata"] = json.loads(data.pop("metadata_json"))
            data["tags"] = sorted(tag_map.get(data["id"], set()))
            data["tree_ids"] = sorted(tree_map.get(data["id"], set()))
            memory.nodes[data["id"]] = MemoryNode.from_dict(data)

        memory.trees = {}
        for row in connection.execute("SELECT * FROM trees"):
            data = dict(row)
            data["metadata"] = json.loads(data.pop("metadata_json"))
            memory.trees[data["id"]] = MemoryTree.from_dict(data)

        memory.edges = {}
        for row in connection.execute("SELECT * FROM edges"):
            data = dict(row)
            data["metadata"] = json.loads(data.pop("metadata_json"))
            memory.edges[data["id"]] = Edge.from_dict(data)

        memory.category_roots = {
            row["category"]: row["tree_id"]
            for row in connection.execute("SELECT category, tree_id FROM category_roots")
        }
        memory.events = [
            MemoryEvent(
                id=row["id"],
                seq=row["seq"],
                op=EventOp(row["op"]),
                payload=json.loads(row["payload_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in connection.execute("SELECT * FROM event_log ORDER BY seq")
        ]
        memory.rebuild_indices()
        return memory

    def _write(self, connection: sqlite3.Connection, memory: FibMind) -> None:
        connection.executemany(
            """
            INSERT INTO nodes(
                id, title, content, category, node_type, layer, created_at, updated_at,
                metadata_json, scope, owner, workspace_id, project_id, session_id,
                task_id, status, status_reason, memory_kind, familiarity,
                access_count, confidence, confidence_source, folded_into, memory_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                content=excluded.content,
                category=excluded.category,
                node_type=excluded.node_type,
                layer=excluded.layer,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at,
                metadata_json=excluded.metadata_json,
                scope=excluded.scope,
                owner=excluded.owner,
                workspace_id=excluded.workspace_id,
                project_id=excluded.project_id,
                session_id=excluded.session_id,
                task_id=excluded.task_id,
                status=excluded.status,
                status_reason=excluded.status_reason,
                memory_kind=excluded.memory_kind,
                familiarity=excluded.familiarity,
                access_count=excluded.access_count,
                confidence=excluded.confidence,
                confidence_source=excluded.confidence_source,
                folded_into=excluded.folded_into,
                memory_weight=excluded.memory_weight
            """,
            [
                (
                    node.id,
                    node.title,
                    node.content,
                    node.category,
                    node.node_type.value,
                    node.layer,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                    json.dumps(node.metadata, ensure_ascii=False, separators=(",", ":")),
                    node.scope.value,
                    node.owner,
                    node.workspace_id,
                    node.project_id,
                    node.session_id,
                    node.task_id,
                    node.status.value,
                    node.status_reason,
                    node.memory_kind.value if node.memory_kind else None,
                    node.familiarity,
                    node.access_count,
                    node.confidence,
                    node.confidence_source,
                    node.folded_into,
                    node.memory_weight,
                )
                for node in memory.nodes.values()
            ],
        )
        connection.executemany(
            """
            INSERT INTO trees(id, category, title, root_node_id, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                category=excluded.category,
                title=excluded.title,
                root_node_id=excluded.root_node_id,
                created_at=excluded.created_at,
                metadata_json=excluded.metadata_json
            """,
            [
                (
                    tree.id,
                    tree.category,
                    tree.title,
                    tree.root_node_id,
                    tree.created_at.isoformat(),
                    json.dumps(tree.metadata, ensure_ascii=False, separators=(",", ":")),
                )
                for tree in memory.trees.values()
            ],
        )
        connection.executemany(
            """
            INSERT INTO edges(
                id, from_node_id, to_node_id, relation_type, weight, direction,
                created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                from_node_id=excluded.from_node_id,
                to_node_id=excluded.to_node_id,
                relation_type=excluded.relation_type,
                weight=excluded.weight,
                direction=excluded.direction,
                created_at=excluded.created_at,
                metadata_json=excluded.metadata_json
            """,
            [
                (
                    edge.id,
                    edge.from_node_id,
                    edge.to_node_id,
                    edge.relation_type.value,
                    edge.weight,
                    edge.direction.value,
                    edge.created_at.isoformat(),
                    json.dumps(edge.metadata, ensure_ascii=False, separators=(",", ":")),
                )
                for edge in memory.edges.values()
            ],
        )

        self._delete_missing(connection, "edges", set(memory.edges))

        connection.execute("DELETE FROM category_roots")
        connection.executemany(
            "INSERT INTO category_roots(category, tree_id) VALUES (?, ?)",
            sorted(memory.category_roots.items()),
        )

        connection.execute("DELETE FROM node_tags")
        connection.executemany(
            "INSERT INTO node_tags(node_id, tag) VALUES (?, ?)",
            [(node.id, tag) for node in memory.nodes.values() for tag in sorted(node.tags)],
        )

        connection.execute("DELETE FROM node_tree_memberships")
        connection.executemany(
            "INSERT INTO node_tree_memberships(node_id, tree_id) VALUES (?, ?)",
            [
                (node.id, tree_id)
                for node in memory.nodes.values()
                for tree_id in sorted(node.tree_ids)
            ],
        )

        self._delete_missing(connection, "trees", set(memory.trees))
        self._delete_missing(connection, "nodes", set(memory.nodes))

        # Append-only in normal operation. The upsert exists for one case:
        # ``forget`` redacts content out of earlier entries, and that redaction
        # has to reach disk or a replay would resurrect it.
        connection.executemany(
            """
            INSERT INTO event_log(id, seq, op, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json
            """,
            [
                (
                    event.id,
                    event.seq,
                    event.op.value,
                    json.dumps(event.payload, ensure_ascii=False, separators=(",", ":")),
                    event.created_at.isoformat(),
                )
                for event in memory.events
            ],
        )
        connection.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'"
        )

    @staticmethod
    def _delete_missing(
        connection: sqlite3.Connection, table: str, current_ids: set[str]
    ) -> None:
        existing_ids = {row["id"] for row in connection.execute(f"SELECT id FROM {table}")}
        connection.executemany(
            f"DELETE FROM {table} WHERE id = ?",
            [(item_id,) for item_id in sorted(existing_ids - current_ids)],
        )


def _check_log_version(found: int | str | None) -> None:
    """Refuse a log written at a format version this build cannot replay.

    A missing version means the store predates versioning and is read as
    version 1. A newer version is refused rather than guessed at: the log is the
    source of truth, and misreading it silently corrupts everything derived.
    """
    if found is None:
        return
    if int(found) > LOG_VERSION:
        raise ValueError(
            f"Memory log is version {found}; this build reads up to {LOG_VERSION}"
        )


def open_store(path: str | Path) -> MemoryStore:
    """Open a store selected by its filename suffix."""
    store_path = Path(path)
    suffix = store_path.suffix.lower()
    if suffix == ".json":
        return JsonStore(store_path)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return SqliteStore(store_path)
    valid = ".json, .db, .sqlite, .sqlite3"
    raise ValueError(f"Unsupported store format for {store_path}. Expected one of: {valid}")
