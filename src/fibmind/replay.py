"""Rebuild forest state from the event log.

The log is the source of truth; nodes, edges, and trees are a cache. This
module is the *only* place that knows how each ``EventOp`` changes that cache
on replay, so adding an op means adding one handler here and one ``_record``
call in :class:`fibmind.graph.FibMind`.

Handlers write directly into the forest's tables. Recall counters
(``familiarity``, ``access_count``) are deliberately reset: they describe how
a store was used, not what it knows.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fibmind.models import Edge, EventOp, MemoryEvent, MemoryNode, MemoryStatus, MemoryTree, NodeType, Verdict

if TYPE_CHECKING:
    from fibmind.graph import FibMind

Handler = Callable[["FibMind", MemoryEvent, dict[str, Any]], None]


def apply_event(memory: FibMind, event: MemoryEvent) -> None:
    """Fold one event into ``memory``. Unknown ops are ignored, so an older
    implementation can replay a newer log without failing."""
    handler = _HANDLERS.get(event.op)
    if handler is not None:
        handler(memory, event, event.payload)


def _apply_create(memory: FibMind, event: MemoryEvent, payload: dict[str, Any]) -> None:
    node_payload = payload.get("node") or payload.get("root")
    if node_payload is not None:
        node = MemoryNode.from_dict(node_payload)
        node.familiarity = 0.0
        node.access_count = 0
        memory.nodes[node.id] = node
    tree_id = payload.get("tree_id")
    if tree_id is None:
        return
    root_id = payload.get("existing_root_node_id") or (node_payload["id"] if node_payload else None)
    if root_id is None:
        return
    created_at = payload.get("created_at")
    memory.trees[tree_id] = MemoryTree(
        id=tree_id,
        category=payload["category"],
        title=payload.get("title", payload["category"]),
        root_node_id=root_id,
        created_at=(datetime.fromisoformat(created_at) if created_at is not None else event.created_at),
    )
    memory.category_roots[payload["category"]] = tree_id
    root = memory.nodes.get(root_id)
    if root is not None:
        root.tree_ids.add(tree_id)
        # ``expand_node_to_tree`` promotes an existing node to CONCEPT as it
        # hands it a tree; replay it here.
        node_type = payload.get("node_type")
        if node_type is not None:
            root.node_type = NodeType(node_type)


def _apply_link(memory: FibMind, event: MemoryEvent, payload: dict[str, Any]) -> None:
    edge = Edge.from_dict(payload["edge"])
    memory.edges[edge.id] = edge


def _apply_fold(memory: FibMind, event: MemoryEvent, payload: dict[str, Any]) -> None:
    for node_id in payload.get("source_node_ids", []):
        node = memory.nodes.get(node_id)
        if node is not None:
            node.folded_into = payload["into_node_id"]


def _apply_revise(memory: FibMind, event: MemoryEvent, payload: dict[str, Any]) -> None:
    node = memory.nodes.get(payload["node_id"])
    after = payload.get("after")
    if node is None or not isinstance(after, dict) or "title" not in after:
        return
    node.title = after["title"]
    node.content = after["content"]
    node.tags = set(after.get("tags", []))
    memory.lexical.add(node)
    # Old evidence does not vouch for new content.
    node.confidence = 0.0
    node.confidence_source = None
    node.status = MemoryStatus(after.get("status", MemoryStatus.ACTIVE.value))
    node.status_reason = None


def _apply_observe(memory: FibMind, event: MemoryEvent, payload: dict[str, Any]) -> None:
    node = memory.nodes.get(payload["node_id"])
    if node is None:
        return
    node.confidence = float(payload["confidence"])
    node.confidence_source = payload.get("source")
    status = payload.get("status")
    if status is not None:
        node.status = MemoryStatus(status)
        node.status_reason = payload.get("status_reason")
    elif payload.get("verdict") == Verdict.REFUTED.value:
        node.status = MemoryStatus.REFUTED
        node.status_reason = payload.get("note") or f"refuted by {payload.get('source')}"


def _apply_set_status(memory: FibMind, event: MemoryEvent, payload: dict[str, Any]) -> None:
    node = memory.nodes.get(payload["node_id"])
    if node is not None:
        node.status = MemoryStatus(payload["status"])
        node.status_reason = payload.get("reason")


def _apply_promote(memory: FibMind, event: MemoryEvent, payload: dict[str, Any]) -> None:
    node = memory.nodes.get(payload["node_id"])
    if node is None:
        return
    node_type = payload.get("node_type")
    if node_type is not None:
        node.node_type = NodeType(node_type)
    # Archiving also moves the node between layers; the forest owns the
    # category/layer index, so let it do the move.
    layer = payload.get("layer")
    if layer is not None:
        memory.move_node_to_layer(node, layer)


def _apply_forget(memory: FibMind, event: MemoryEvent, payload: dict[str, Any]) -> None:
    node_id = payload["node_id"]
    memory.nodes.pop(node_id, None)
    for edge_id in payload.get("removed_edges", []):
        memory.edges.pop(edge_id, None)
    for tree_id, tree in list(memory.trees.items()):
        if tree.root_node_id == node_id:
            memory.trees.pop(tree_id)
            memory.category_roots.pop(tree.category, None)


_HANDLERS: dict[EventOp, Handler] = {
    EventOp.CREATE_NODE: _apply_create,
    EventOp.CREATE_TREE: _apply_create,
    EventOp.LINK: _apply_link,
    EventOp.FOLD: _apply_fold,
    EventOp.REVISE: _apply_revise,
    EventOp.OBSERVE: _apply_observe,
    EventOp.SET_STATUS: _apply_set_status,
    EventOp.PROMOTE: _apply_promote,
    EventOp.FORGET: _apply_forget,
}
