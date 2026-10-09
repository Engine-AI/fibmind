"""Persistence helpers for FibMind."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from fibmind.graph import FibMind
from fibmind.models import LOG_VERSION, Edge, EventOp, MemoryEvent, MemoryNode, MemoryTree
from fibmind.ranking import tokenize


class MemoryStore(Protocol):
    """Common persistence interface used by :class:`MemoryService`."""

    path: Path

    def load(self) -> FibMind: ...

    def save(self, memory: FibMind) -> None: ...

    def transaction(self) -> Iterator[FibMind]: ...

    def read_setting(self, key: str) -> Any | None: ...

    def write_setting(self, key: str, value: Any) -> None: ...

    def revision(self) -> str: ...


def _snapshot_value(value: dict[str, Any]) -> str:
    """Freeze mutable metadata and event payloads for a transaction diff."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class _StoreSnapshot:
    nodes: dict[str, str]
    trees: dict[str, str]
    edges: dict[str, str]
    events: dict[str, str]
    category_roots: dict[str, str]

    @classmethod
    def capture(cls, memory: FibMind) -> _StoreSnapshot:
        return cls(
            nodes={key: _snapshot_value(node.to_dict()) for key, node in memory.nodes.items()},
            trees={key: _snapshot_value(tree.to_dict()) for key, tree in memory.trees.items()},
            edges={key: _snapshot_value(edge.to_dict()) for key, edge in memory.edges.items()},
            events={event.id: _snapshot_value(event.to_dict()) for event in memory.events},
            category_roots=dict(memory.category_roots),
        )


class JsonStore:
    """Atomic JSON persistence retained for demos and compatibility."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, memory: FibMind) -> None:
        settings: dict[str, Any] = {}
        if self.path.exists():
            try:
                settings = json.loads(self.path.read_text(encoding="utf-8")).get("settings") or {}
            except (json.JSONDecodeError, OSError):
                settings = {}
        payload: dict[str, Any] = {
            "log_version": LOG_VERSION,
            "settings": settings,
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
        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def read_setting(self, key: str) -> Any | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return (payload.get("settings") or {}).get(key)

    def write_setting(self, key: str, value: Any) -> None:
        payload: dict[str, Any] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload.setdefault("settings", {})[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(json.dumps(payload, ensure_ascii=False, indent=2))

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

    def revision(self) -> str:
        """A token that changes whenever the file does.

        There is no counter to read here, so the file's own mtime and size
        stand in. Coarser than SQLite's ``meta.revision`` — two writes inside
        one mtime tick look identical — which is why callers must also drop
        their cache around their own writes rather than relying on this alone.
        """
        try:
            stat = self.path.stat()
        except OSError:
            return "0"
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    @contextmanager
    def transaction(self) -> Iterator[FibMind]:
        memory = self.load()
        yield memory
        self.save(memory)


class SqliteStore:
    """Transactional SQLite persistence for the complete memory graph."""

    SCHEMA_VERSION = 6

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
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    id UNINDEXED,
                    title,
                    content,
                    tags,
                    category
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
            version = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if version is None or int(version["value"]) != self.SCHEMA_VERSION:
                raise ValueError("Unsupported FibMind SQLite schema version")
            log_version = connection.execute("SELECT value FROM meta WHERE key = 'log_version'").fetchone()
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

        columns = {info["name"] for info in connection.execute("PRAGMA table_info(nodes)")}
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
        connection.execute("UPDATE nodes SET layer = REPLACE(layer, 'folded:', '') WHERE layer LIKE 'folded:%'")

        # v6 adds nodes_fts FTS5 virtual table for incremental full-text search
        if int(row["value"]) < 6:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    id UNINDEXED,
                    title,
                    content,
                    tags,
                    category
                );
                """
            )
            count = connection.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
            if count == 0:
                tag_map: dict[str, list[str]] = {}
                for tag_row in connection.execute("SELECT node_id, tag FROM node_tags"):
                    tag_map.setdefault(tag_row[0], []).append(tag_row[1])
                nodes_data = connection.execute("SELECT id, title, content, category FROM nodes").fetchall()
                fts_rows = [
                    (
                        node_row[0],
                        " ".join(tokenize(node_row[1])),
                        " ".join(tokenize(node_row[2])),
                        " ".join(tokenize(" ".join(sorted(tag_map.get(node_row[0], []))))),
                        " ".join(tokenize(node_row[3])),
                    )
                    for node_row in nodes_data
                ]
                connection.executemany(
                    "INSERT INTO nodes_fts(id, title, content, tags, category) VALUES (?, ?, ?, ?, ?)",
                    fts_rows,
                )

        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(self.SCHEMA_VERSION),),
        )

    def read_setting(self, key: str) -> Any | None:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key = ?", (f"setting:{key}",)).fetchone()
        return json.loads(row["value"]) if row else None

    def write_setting(self, key: str, value: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (f"setting:{key}", json.dumps(value, ensure_ascii=False)),
            )

    def load(self) -> FibMind:
        with self._connect() as connection:
            return self._load(connection)

    def revision(self) -> str:
        """The write counter ``_write`` bumps; changes on every commit.

        Read outside a transaction on purpose: the point is to notice writes
        from *other* connections, including other processes.
        """
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()
        return str(row["value"]) if row else "0"

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
            before = _StoreSnapshot.capture(memory)
            yield memory
            self._write(connection, memory, before)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _load(self, connection: sqlite3.Connection) -> FibMind:
        tag_map: dict[str, set[str]] = {}
        for row in connection.execute("SELECT node_id, tag FROM node_tags"):
            tag_map.setdefault(row[0], set()).add(row[1])

        tree_map: dict[str, set[str]] = {}
        for row in connection.execute("SELECT node_id, tree_id FROM node_tree_memberships"):
            tree_map.setdefault(row[0], set()).add(row[1])

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
            row[0]: row[1] for row in connection.execute("SELECT category, tree_id FROM category_roots")
        }
        memory.events = [
            MemoryEvent(
                id=row[0],
                seq=row[1],
                op=EventOp(row[2]),
                payload=json.loads(row[3]),
                created_at=datetime.fromisoformat(row[4]),
            )
            for row in connection.execute("SELECT id, seq, op, payload_json, created_at FROM event_log ORDER BY seq")
        ]
        memory.rebuild_indices()
        return memory

    def _write(self, connection: sqlite3.Connection, memory: FibMind, before: _StoreSnapshot | None = None) -> None:
        changed_nodes = [
            node
            for node in memory.nodes.values()
            if before is None or _snapshot_value(node.to_dict()) != before.nodes.get(node.id)
        ]
        changed_trees = [
            tree
            for tree in memory.trees.values()
            if before is None or _snapshot_value(tree.to_dict()) != before.trees.get(tree.id)
        ]
        changed_edges = [
            edge
            for edge in memory.edges.values()
            if before is None or _snapshot_value(edge.to_dict()) != before.edges.get(edge.id)
        ]
        changed_events = [
            event
            for event in memory.events
            if before is None or _snapshot_value(event.to_dict()) != before.events.get(event.id)
        ]
        deleted_nodes = set(before.nodes) - set(memory.nodes) if before else set()
        deleted_trees = set(before.trees) - set(memory.trees) if before else set()
        deleted_edges = set(before.edges) - set(memory.edges) if before else set()
        roots_changed = before is None or before.category_roots != memory.category_roots
        if before is not None and not any(
            (
                changed_nodes,
                changed_trees,
                changed_edges,
                changed_events,
                deleted_nodes,
                deleted_trees,
                deleted_edges,
                roots_changed,
            )
        ):
            return
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
                for node in changed_nodes
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
                for tree in changed_trees
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
                for edge in changed_edges
            ],
        )

        self._delete_missing(connection, "edges", set(memory.edges), set(before.edges) if before else None)

        if roots_changed:
            connection.execute("DELETE FROM category_roots")
            connection.executemany(
                "INSERT INTO category_roots(category, tree_id) VALUES (?, ?)",
                sorted(memory.category_roots.items()),
            )

        changed_node_ids = {node.id for node in changed_nodes} | deleted_nodes
        connection.executemany("DELETE FROM node_tags WHERE node_id = ?", [(key,) for key in changed_node_ids])
        connection.executemany(
            "INSERT INTO node_tags(node_id, tag) VALUES (?, ?)",
            [(node.id, tag) for node in changed_nodes for tag in sorted(node.tags)],
        )

        connection.executemany(
            "DELETE FROM node_tree_memberships WHERE node_id = ?", [(key,) for key in changed_node_ids]
        )
        connection.executemany(
            "INSERT INTO node_tree_memberships(node_id, tree_id) VALUES (?, ?)",
            [(node.id, tree_id) for node in changed_nodes for tree_id in sorted(node.tree_ids)],
        )

        self._delete_missing(connection, "trees", set(memory.trees), set(before.trees) if before else None)
        missing_node_ids = self._delete_missing(
            connection, "nodes", set(memory.nodes), set(before.nodes) if before else None
        )

        # Maintain nodes_fts incrementally
        all_affected = {node.id for node in changed_nodes} | deleted_nodes | missing_node_ids
        if all_affected:
            connection.executemany("DELETE FROM nodes_fts WHERE id = ?", [(nid,) for nid in all_affected])
            if changed_nodes:
                connection.executemany(
                    "INSERT INTO nodes_fts(id, title, content, tags, category) VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            node.id,
                            " ".join(tokenize(node.title)),
                            " ".join(tokenize(node.content)),
                            " ".join(tokenize(" ".join(sorted(node.tags)))),
                            " ".join(tokenize(node.category)),
                        )
                        for node in changed_nodes
                    ],
                )

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
                for event in changed_events
            ],
        )
        connection.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'")

    def search_fts(self, query: str, limit: int = 50) -> list[str]:
        """Perform fast full-text search directly via SQLite FTS5 BM25 ranking.

        Returns matching node IDs ordered by relevance without full graph deserialization.
        """
        terms = tokenize(query)
        if not terms:
            return []
        escaped_terms = [t.replace('"', '""') for t in terms]
        match_expr = " OR ".join(f'"{t}"' for t in escaped_terms)
        with self._connect() as connection:
            try:
                rows = connection.execute(
                    "SELECT id, rank FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match_expr, limit),
                ).fetchall()
                return [row[0] for row in rows]
            except sqlite3.OperationalError:
                return []

    @staticmethod
    def _delete_missing(
        connection: sqlite3.Connection, table: str, current_ids: set[str], existing_ids: set[str] | None = None
    ) -> set[str]:
        if existing_ids is None:
            existing_ids = {row["id"] for row in connection.execute(f"SELECT id FROM {table}")}
        missing = existing_ids - current_ids
        connection.executemany(
            f"DELETE FROM {table} WHERE id = ?",
            [(item_id,) for item_id in sorted(missing)],
        )
        return missing


def _check_log_version(found: int | str | None) -> None:
    """Refuse a log written at a format version this build cannot replay.

    A missing version means the store predates versioning and is read as
    version 1. A newer version is refused rather than guessed at: the log is the
    source of truth, and misreading it silently corrupts everything derived.
    """
    if found is None:
        return
    if int(found) > LOG_VERSION:
        raise ValueError(f"Memory log is version {found}; this build reads up to {LOG_VERSION}")


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
