"""Domain models for the FibMind memory graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


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
    importance: float = 0.0
    access_count: int = 0
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
            "importance": self.importance,
            "access_count": self.access_count,
            "memory_weight": self.memory_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
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
            importance=float(data.get("importance", 0.0)),
            access_count=int(data.get("access_count", 0)),
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
