"""Domain models for the FibMind memory graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


# Version of the event-log format. Any implementation — in any language — that
# can replay a log at this version must rebuild the same nodes, edges, and trees
# (see docs/contract.md). Bump it only when an existing event payload changes
# meaning; adding a new optional field does not require a bump.
LOG_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def optional_id(value: str | None) -> str | None:
    """Normalize an identity field: blank strings become ``None``."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class NodeType(StrEnum):
    RAW = "raw"
    COMPRESSED = "compressed"
    SUMMARY = "summary"
    CONCEPT = "concept"
    ROOT = "root"
    ARCHIVE = "archive"


class RelationType(StrEnum):
    TREE_CHILD = "tree_child"
    CONTAINS = "contains"
    SUMMARY_OF = "summary_of"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    REFERENCES = "references"
    CAUSED_BY = "caused_by"
    VERSION_OF = "version_of"
    DERIVED_FROM = "derived_from"
    ANSWER_TO = "answer_to"
    DEPENDS_ON = "depends_on"


class EdgeDirection(StrEnum):
    DIRECTED = "directed"
    BIDIRECTIONAL = "bidirectional"


class TraversalDirection(StrEnum):
    OUT = "out"
    IN = "in"
    BOTH = "both"


class MemoryScope(StrEnum):
    """Who a memory belongs to, and therefore whether it may be shared.

    ``PERSONAL`` memories are observations about one owner and must never be
    shared: doing so both leaks privacy and pollutes other owners' preferences.
    ``KNOWLEDGE`` memories are claims about the world, distilled from several
    personal observations, and are the only scope that is shareable.
    ``SESSION`` memories are scratch state that is expected to be forgotten.
    """

    SESSION = "session"
    PERSONAL = "personal"
    KNOWLEDGE = "knowledge"


class MemoryStatus(StrEnum):
    """Whether a memory may participate in normal recall.

    Inactive memories stay persisted for provenance and maintenance, but normal
    search/context only sees ``ACTIVE`` nodes. ``PENDING`` is a memory written
    by an automatic review that a human has not approved yet; it stays out of
    recall until approved. ``FORGOTTEN`` is an audit state; forgotten nodes
    themselves are deleted and their event payload is redacted.
    """

    ACTIVE = "active"
    PENDING = "pending"
    STALE = "stale"
    REFUTED = "refuted"
    FORGOTTEN = "forgotten"


class Verdict(StrEnum):
    """The outcome of checking a memory against something outside the model."""

    CONFIRMED = "confirmed"
    REFUTED = "refuted"


class EventOp(StrEnum):
    """The append-only operations that make up the memory event log."""

    CREATE_NODE = "create_node"
    LINK = "link"
    FOLD = "fold"
    REVISE = "revise"
    FORGET = "forget"
    OBSERVE = "observe"
    PROMOTE = "promote"
    CREATE_TREE = "create_tree"
    SET_STATUS = "set_status"


@dataclass(slots=True)
class MemoryEvent:
    """One immutable entry in the memory log.

    The log — not the materialized node/edge tables — is the source of truth.
    Everything else is a cache that :meth:`FibMind.rebuild_from_log` can
    reconstruct, which is what makes the store replayable across model
    generations and mergeable across devices.
    """

    op: EventOp
    payload: dict[str, Any]
    seq: int = 0
    id: str = field(default_factory=lambda: new_id("event"))
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seq": self.seq,
            "op": self.op.value,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "log_version": LOG_VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEvent":
        return cls(
            id=data["id"],
            seq=int(data.get("seq", 0)),
            op=EventOp(data["op"]),
            payload=data.get("payload", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass(slots=True)
class MemoryNode:
    title: str
    content: str
    category: str
    node_type: NodeType = NodeType.RAW
    layer: str = "raw"
    id: str = field(default_factory=lambda: new_id("node"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    tree_ids: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    scope: MemoryScope = MemoryScope.PERSONAL
    owner: str | None = None
    workspace_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    status_reason: str | None = None
    # How often this node has been recalled. A cache derived from access
    # patterns, deliberately kept OUT of relevance scoring: rewarding recall
    # frequency creates a recalled -> ranked-higher -> recalled-more loop that
    # promotes whatever is familiar rather than whatever is true.
    familiarity: float = 0.0
    access_count: int = 0
    # How much external evidence says this node is correct. Only
    # ``record_outcome`` moves it, and only with a stated source. This is the
    # single signal relevance scoring is allowed to trust.
    confidence: float = 0.0
    confidence_source: str | None = None
    # Set when a higher layer has absorbed this node. Folded nodes stay intact
    # and searchable-by-provenance instead of being overwritten in place, so the
    # raw material survives for re-distillation later.
    folded_into: str | None = None
    memory_weight: int = 1

    def touch(self) -> None:
        self.updated_at = utc_now()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "node_type": self.node_type.value,
            "layer": self.layer,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tree_ids": sorted(self.tree_ids),
            "tags": sorted(self.tags),
            "metadata": self.metadata,
            "scope": self.scope.value,
            "owner": self.owner,
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "familiarity": self.familiarity,
            "access_count": self.access_count,
            "confidence": self.confidence,
            "confidence_source": self.confidence_source,
            "folded_into": self.folded_into,
            "memory_weight": self.memory_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        # Stores written before familiarity/confidence were split carry a single
        # ``importance`` field. It was access-frequency driven, so it maps onto
        # familiarity; confidence starts at zero because no external check ever
        # backed those numbers.
        familiarity = data.get("familiarity", data.get("importance", 0.0))
        node = cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            category=data["category"],
            node_type=NodeType(data["node_type"]),
            layer=data["layer"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tree_ids=set(data.get("tree_ids", [])),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
            scope=MemoryScope(data.get("scope", MemoryScope.PERSONAL.value)),
            owner=data.get("owner"),
            workspace_id=optional_id(data.get("workspace_id")),
            project_id=optional_id(data.get("project_id")),
            session_id=optional_id(data.get("session_id")),
            task_id=optional_id(data.get("task_id")),
            status=MemoryStatus(data.get("status", MemoryStatus.ACTIVE.value)),
            status_reason=data.get("status_reason"),
            familiarity=float(familiarity),
            access_count=int(data.get("access_count", 0)),
            confidence=float(data.get("confidence", 0.0)),
            confidence_source=data.get("confidence_source"),
            folded_into=data.get("folded_into"),
            memory_weight=int(data.get("memory_weight", 1)),
        )
        return node


@dataclass(slots=True)
class Edge:
    from_node_id: str
    to_node_id: str
    relation_type: RelationType
    weight: float = 1.0
    direction: EdgeDirection = EdgeDirection.DIRECTED
    id: str = field(default_factory=lambda: new_id("edge"))
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def connects(self, node_id: str) -> bool:
        if self.from_node_id == node_id:
            return True
        return self.direction == EdgeDirection.BIDIRECTIONAL and self.to_node_id == node_id

    def neighbor_of(self, node_id: str) -> str | None:
        if self.from_node_id == node_id:
            return self.to_node_id
        if self.direction == EdgeDirection.BIDIRECTIONAL and self.to_node_id == node_id:
            return self.from_node_id
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "direction": self.direction.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Edge":
        return cls(
            id=data["id"],
            from_node_id=data["from_node_id"],
            to_node_id=data["to_node_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=float(data.get("weight", 1.0)),
            direction=EdgeDirection(data.get("direction", EdgeDirection.DIRECTED.value)),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class MemoryTree:
    category: str
    title: str
    root_node_id: str
    id: str = field(default_factory=lambda: new_id("tree"))
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "root_node_id": self.root_node_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryTree":
        return cls(
            id=data["id"],
            category=data["category"],
            title=data["title"],
            root_node_id=data["root_node_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )
