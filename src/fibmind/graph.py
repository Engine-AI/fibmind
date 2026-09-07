"""FibMind memory forest graph engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from fibmind.fibonacci import DEFAULT_LAYER_POLICY, FibonacciLayerPolicy
from fibmind.models import (
    MemoryKind,
    infer_memory_kind,
    Edge,
    EdgeDirection,
    EventOp,
    MemoryEvent,
    MemoryNode,
    MemoryScope,
    MemoryStatus,
    MemoryTree,
    NodeType,
    RelationType,
    TraversalDirection,
    Verdict,
    optional_id,
    utc_now,
)
from fibmind.ranking import ScoredHit, rank_nodes

# How much a single confirmation/refutation moves a node's confidence. Evidence
# accumulates rather than deciding outright, so one lucky pass does not make a
# memory authoritative and one flake does not erase it.
CONFIDENCE_STEP = 0.25

# Content stored in the log in place of a forgotten node's text. Replay must not
# resurrect what a user asked to have deleted.
TOMBSTONE = "[forgotten]"

# Minimum number of distinct personal observations required before a claim may
# be promoted to shared knowledge. A statistical bar, not a model's opinion:
# "this generalizes" is a claim about a population, so it needs a population.
DEFAULT_PROMOTION_THRESHOLD = 3

# Similarity edges are only auto-created above this relevance score, so that
# appending a memory does not wire it to everything sharing a common word.
AUTO_LINK_MIN_SCORE = 0.5
AUTO_LINK_TOP_K = 3


@dataclass(frozen=True, slots=True)
class SearchHit:
    node: MemoryNode
    depth: int
    via_relation: RelationType | None = None
    via_direction: TraversalDirection | None = None


class FibMind:
    """A prototype Fibonacci forest graph for long-term AI memory."""

    def __init__(self, layer_policy: FibonacciLayerPolicy = DEFAULT_LAYER_POLICY) -> None:
        self.layer_policy = layer_policy
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: dict[str, Edge] = {}
        self.trees: dict[str, MemoryTree] = {}
        self.category_roots: dict[str, str] = {}
        self.events: list[MemoryEvent] = []
        self.adjacency: dict[str, set[str]] = {}
        self.reverse_adjacency: dict[str, set[str]] = {}
        self.by_cat_layer: dict[tuple[str, str], set[str]] = {}

    def _record(self, op: EventOp, payload: dict) -> MemoryEvent:
        """Append one entry to the log. Every semantic mutation goes through here."""
        event = MemoryEvent(op=op, payload=payload, seq=len(self.events) + 1)
        self.events.append(event)
        return event

    def create_tree(self, category: str, title: str | None = None, content: str = "") -> str:
        if category in self.category_roots:
            return self.category_roots[category]

        root = MemoryNode(
            title=title or f"{category} root",
            content=content,
            category=category,
            node_type=NodeType.ROOT,
            layer="long_term",
            confidence=1.0,
            confidence_source="structural",
            memory_weight=0,
        )
        tree = MemoryTree(category=category, title=title or category, root_node_id=root.id)
        root.tree_ids.add(tree.id)

        self._add_node(root)
        self.trees[tree.id] = tree
        self.category_roots[category] = tree.id
        self._record(
            EventOp.CREATE_TREE,
            {
                "tree_id": tree.id,
                "category": category,
                "title": tree.title,
                "created_at": tree.created_at.isoformat(),
                "root": root.to_dict(),
            },
        )
        return tree.id

    def append(
        self,
        category: str,
        title: str,
        content: str,
        tags: Iterable[str] | None = None,
        metadata: dict | None = None,
        scope: MemoryScope = MemoryScope.PERSONAL,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        auto_link: bool = True,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        compress: bool = True,
        memory_kind: MemoryKind | str | None = None,
    ) -> str:
        parsed_scope = MemoryScope(scope)
        workspace_id = optional_id(workspace_id)
        project_id = optional_id(project_id)
        session_id = optional_id(session_id)
        task_id = optional_id(task_id)
        if parsed_scope == MemoryScope.SESSION and session_id is None:
            raise ValueError("session-scoped memories require session_id")

        tree_id = self.create_tree(category)
        tree = self.trees[tree_id]

        node = MemoryNode(
            title=title,
            content=content,
            category=category,
            node_type=NodeType.RAW,
            layer="raw",
            tags=set(tags or []),
            metadata=metadata or {},
            scope=parsed_scope,
            owner=owner,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
            status=MemoryStatus(status),
            memory_kind=(
                MemoryKind(memory_kind) if memory_kind else infer_memory_kind(category, parsed_scope)
            ),
        )
        node.tree_ids.add(tree_id)
        self._add_node(node)
        self._record(EventOp.CREATE_NODE, {"node": node.to_dict()})

        self.link_nodes(
            tree.root_node_id,
            node.id,
            RelationType.TREE_CHILD,
            weight=1.0,
            direction=EdgeDirection.DIRECTED,
        )
        if auto_link:
            self._auto_link_similar(node)
        if compress:
            self.compress_overflow(category)
        return node.id

    def set_status(self, node_id: str, status: MemoryStatus, reason: str) -> MemoryNode:
        """Move a node between lifecycle states, recording why.

        ``mark_stale`` is the common case; this is the general form used by the
        review flow to approve (``pending`` → ``active``) or reject
        (``pending`` → ``stale``) an automatically extracted memory.
        """
        node = self._require_node(node_id)
        if not reason or not reason.strip():
            raise ValueError("reason must explain the status change")
        node.status = MemoryStatus(status)
        node.status_reason = reason
        node.updated_at = utc_now()
        self._record(
            EventOp.SET_STATUS,
            {"node_id": node_id, "status": node.status.value, "reason": reason},
        )
        return node

    def _auto_link_similar(self, node: MemoryNode) -> None:
        """Connect a new node to its nearest existing neighbours.

        Without this the forest is a star: every node hangs off its tree root and
        nothing else, so a one-hop expansion reaches only the root — which
        context building filters out — and yields nothing.
        """
        candidates = rank_nodes(
            (
                other
                for other in self.nodes.values()
                if other.id != node.id
                and self.is_visible(
                    other,
                    scopes={node.scope},
                    owner=node.owner,
                    workspace_id=node.workspace_id,
                    project_id=node.project_id,
                    session_id=node.session_id,
                )
            ),
            f"{node.title} {node.content}",
            top_k=AUTO_LINK_TOP_K,
            min_score=AUTO_LINK_MIN_SCORE,
        )
        for hit in candidates:
            self.link_nodes(
                node.id,
                hit.node.id,
                RelationType.SIMILAR_TO,
                weight=min(1.0, round(hit.score / 2, 4)),
                direction=EdgeDirection.BIDIRECTIONAL,
                metadata={"auto": True, "score": round(hit.score, 4)},
            )

    def link_nodes(
        self,
        from_node_id: str,
        to_node_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        direction: EdgeDirection = EdgeDirection.DIRECTED,
        metadata: dict | None = None,
    ) -> str:
        self._require_node(from_node_id)
        self._require_node(to_node_id)
        edge = Edge(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relation_type=relation_type,
            weight=weight,
            direction=direction,
            metadata=metadata or {},
        )
        self._add_edge(edge)
        self._record(EventOp.LINK, {"edge": edge.to_dict()})
        return edge.id

    def record_outcome(
        self,
        node_id: str,
        verdict: Verdict,
        source: str,
        note: str | None = None,
        session_id: str | None = None,
    ) -> MemoryNode:
        """Attach external evidence about whether a memory is correct.

        This is the only path that moves ``confidence``, and it demands a
        ``source`` — the point of the field is that something outside the model
        checked the claim. Recall alone never qualifies, which is what keeps
        familiar-but-wrong memories from rising.
        """
        node = self._require_node(node_id)
        verdict = Verdict(verdict)
        if not source or not source.strip():
            raise ValueError("source must name what produced the verdict")

        step = CONFIDENCE_STEP if verdict == Verdict.CONFIRMED else -CONFIDENCE_STEP
        node.confidence = max(0.0, min(1.0, node.confidence + step))
        node.confidence_source = source
        if verdict == Verdict.REFUTED:
            node.status = MemoryStatus.REFUTED
            node.status_reason = note or f"refuted by {source}"
        node.updated_at = utc_now()
        self._record(
            EventOp.OBSERVE,
            {
                "node_id": node_id,
                "verdict": verdict.value,
                "source": source,
                "note": note,
                "confidence": node.confidence,
                "status": node.status.value,
                "status_reason": node.status_reason,
                "session_id": optional_id(session_id),
            },
        )
        return node

    def outcome_counts(self, node_id: str) -> dict[str, int]:
        """Tally the evidence recorded against one node, from the log.

        Derived from ``observe`` events rather than kept as a counter, so it
        survives replay and cannot drift from the log. ``sessions`` counts the
        distinct sessions that supplied a confirmation; evidence from one
        session repeated three times is weaker than from three sessions.
        """
        confirmed = refuted = 0
        sessions: set[str] = set()
        for event in self.events:
            if event.op != EventOp.OBSERVE or event.payload.get("node_id") != node_id:
                continue
            verdict = event.payload.get("verdict")
            if verdict == Verdict.CONFIRMED.value:
                confirmed += 1
                session = event.payload.get("session_id")
                if session:
                    sessions.add(str(session))
            elif verdict == Verdict.REFUTED.value:
                refuted += 1
        return {"confirmed": confirmed, "refuted": refuted, "sessions": len(sessions)}

    def mark_stale(self, node_id: str, reason: str) -> MemoryNode:
        """Retire an outdated memory without deleting its provenance."""
        node = self._require_node(node_id)
        if not reason or not reason.strip():
            raise ValueError("reason must explain why the memory is stale")
        node.status = MemoryStatus.STALE
        node.status_reason = reason
        node.updated_at = utc_now()
        self._record(
            EventOp.SET_STATUS,
            {
                "node_id": node_id,
                "status": node.status.value,
                "reason": reason,
            },
        )
        return node

    def revise(
        self,
        node_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> MemoryNode:
        """Correct a memory in place, resetting the evidence that backed it.

        Confidence is earned by the old content, so it does not transfer: a
        revised claim starts unproven again.
        """
        node = self._require_node(node_id)
        before = {"title": node.title, "content": node.content, "tags": sorted(node.tags)}
        if title is not None:
            node.title = title
        if content is not None:
            node.content = content
        if tags is not None:
            node.tags = set(tags)
        node.confidence = 0.0
        node.confidence_source = None
        node.status = MemoryStatus.ACTIVE
        node.status_reason = None
        node.updated_at = utc_now()
        self._record(
            EventOp.REVISE,
            {
                "node_id": node_id,
                "before": before,
                "after": {
                    "title": node.title,
                    "content": node.content,
                    "tags": sorted(node.tags),
                    "status": node.status.value,
                },
            },
        )
        return node

    def forget(self, node_id: str, reason: str) -> dict:
        """Delete a memory and redact it from the log.

        Dropping the row is not enough: the log is the source of truth, so any
        content left in earlier ``create_node``/``revise`` entries would come
        back on the next replay. Those payloads are tombstoned here, which is
        what makes deletion actually hold.
        """
        node = self._require_node(node_id)
        if not reason or not reason.strip():
            raise ValueError("reason must explain why the memory is being deleted")

        removed_edges = [
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.from_node_id == node_id or edge.to_node_id == node_id
        ]
        for edge_id in removed_edges:
            self._remove_edge(edge_id)

        for other in self.nodes.values():
            if other.folded_into == node_id:
                other.folded_into = None

        self._unindex_node(node)
        self.nodes.pop(node_id)
        self.adjacency.pop(node_id, None)
        self.reverse_adjacency.pop(node_id, None)
        for tree_id, tree in list(self.trees.items()):
            if tree.root_node_id == node_id:
                self.trees.pop(tree_id)
                self.category_roots.pop(tree.category, None)

        self._redact_log(node_id)
        self._record(
            EventOp.FORGET,
            {"node_id": node_id, "reason": reason, "removed_edges": removed_edges},
        )
        return {"node_id": node_id, "reason": reason, "removed_edges": len(removed_edges)}

    def _redact_log(self, node_id: str) -> None:
        """Overwrite a forgotten node's text everywhere it appears in the log."""
        for event in self.events:
            payload = event.payload
            node_payload = payload.get("node")
            if isinstance(node_payload, dict) and node_payload.get("id") == node_id:
                node_payload["title"] = TOMBSTONE
                node_payload["content"] = TOMBSTONE
            if payload.get("node_id") == node_id:
                for section in ("before", "after"):
                    if isinstance(payload.get(section), dict):
                        payload[section] = {"redacted": True}

    def promote_to_knowledge(
        self,
        title: str,
        content: str,
        supporting_node_ids: list[str],
        category: str = "knowledge",
        threshold: int = DEFAULT_PROMOTION_THRESHOLD,
        tags: Iterable[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """Distil several personal observations into one shareable claim.

        The abstraction text is the caller's; what this enforces is the bar for
        making it shared. "This generalizes" is a claim about a population, so it
        needs at least ``threshold`` distinct observations behind it — a model
        asserting that it feels general is exactly the self-assessment this is
        meant to replace.

        Every supporting observation is linked with ``DERIVED_FROM``, so a claim
        that later turns out wrong can be traced back to what produced it.
        """
        unique_ids = list(dict.fromkeys(supporting_node_ids))
        if len(unique_ids) < threshold:
            raise ValueError(
                f"promotion needs at least {threshold} distinct supporting memories, "
                f"got {len(unique_ids)}"
            )
        supporters = [self._require_node(node_id) for node_id in unique_ids]
        inactive = [supporter.id for supporter in supporters if supporter.status != MemoryStatus.ACTIVE]
        if inactive:
            raise ValueError(f"promotion supporters must be active memories: {inactive}")
        workspace_id, project_id = self._promotion_identity(
            supporters, workspace_id=workspace_id, project_id=project_id
        )

        node_id = self.append(
            category=category,
            title=title,
            content=content,
            tags=tags,
            metadata={"supporting_node_ids": unique_ids},
            scope=MemoryScope.KNOWLEDGE,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
            auto_link=False,
        )
        for supporter in supporters:
            self.link_nodes(
                node_id,
                supporter.id,
                RelationType.DERIVED_FROM,
                weight=0.9,
            )
        self._record(
            EventOp.PROMOTE,
            {
                "node_id": node_id,
                "supporting_node_ids": unique_ids,
                "threshold": threshold,
            },
        )
        return node_id

    def promote_node(self, node_id: str, target_type: NodeType = NodeType.CONCEPT) -> str:
        """Mark a memory as a concept, keeping it retrievable.

        ``ROOT`` is reserved for the structural anchors that ``create_tree``
        creates; promoting a real memory to ROOT used to hide it, because
        ranking skips root nodes and layer accounting skips them too.
        """
        if target_type == NodeType.ROOT:
            raise ValueError("ROOT is reserved for tree anchors; promote to CONCEPT instead")
        node = self._require_node(node_id)
        node.node_type = target_type
        self._reinforce_node(node, amount=0.05)
        self._record(
            EventOp.PROMOTE, {"node_id": node_id, "node_type": target_type.value}
        )
        return node_id

    def expand_node_to_tree(self, node_id: str, title: str | None = None) -> str:
        """Give a concept its own tree without turning it into a hidden anchor."""
        node = self._require_node(node_id)
        category = f"{node.category}:{node.title}".lower().replace(" ", "_")
        original_category = category
        suffix = 2
        while category in self.category_roots:
            category = f"{original_category}_{suffix}"
            suffix += 1

        node.node_type = NodeType.CONCEPT
        tree = MemoryTree(category=category, title=title or node.title, root_node_id=node.id)
        node.tree_ids.add(tree.id)
        self.trees[tree.id] = tree
        self.category_roots[category] = tree.id
        self._record(
            EventOp.CREATE_TREE,
            {
                "tree_id": tree.id,
                "category": category,
                "title": tree.title,
                "created_at": tree.created_at.isoformat(),
                "existing_root_node_id": node.id,
            },
        )
        return tree.id

    def merge_trees(self, source_tree_id: str, target_tree_id: str) -> None:
        source = self._require_tree(source_tree_id)
        target = self._require_tree(target_tree_id)

        for node in self.nodes.values():
            if source.id in node.tree_ids:
                node.tree_ids.add(target.id)
                node.tree_ids.discard(source.id)

        self.link_nodes(target.root_node_id, source.root_node_id, RelationType.CONTAINS, weight=0.9)
        self.trees.pop(source.id)
        self.category_roots.pop(source.category, None)

    def split_tree(self, root_node_id: str, category: str, title: str | None = None) -> str:
        self._require_node(root_node_id)
        if category in self.category_roots:
            raise ValueError(f"Category already exists: {category}")

        tree = MemoryTree(category=category, title=title or category, root_node_id=root_node_id)
        self.trees[tree.id] = tree
        self.category_roots[category] = tree.id
        self.nodes[root_node_id].tree_ids.add(tree.id)

        for hit in self.search_from(
            root_node_id,
            depth=10,
            relation_types={RelationType.TREE_CHILD},
            direction=TraversalDirection.OUT,
        ):
            hit.node.tree_ids.add(tree.id)
        return tree.id

    def compress_overflow(self, category: str) -> None:
        for layer in self.layer_policy.order:
            capacity = self.layer_policy.capacity_for(layer)
            nodes = self._nodes_in_category_layer(category, layer)
            total_weight = sum(node.memory_weight for node in nodes)
            if total_weight <= capacity:
                continue

            compact_weight = total_weight - self.layer_policy.low_watermark_for(layer)
            overflow = self._select_overflow_nodes(nodes, compact_weight)
            next_layer = self.layer_policy.next_layer(layer)
            groups: dict[tuple, list[MemoryNode]] = {}
            for node in overflow:
                groups.setdefault(self._identity_key(node), []).append(node)
            for group in groups.values():
                if next_layer is None:
                    self._archive_nodes(group)
                else:
                    self._compress_nodes(category, layer, next_layer, group)

    def search_from(
        self,
        node_id: str,
        depth: int = 2,
        relation_types: set[RelationType] | None = None,
        min_weight: float = 0.0,
        reinforce: bool = False,
        direction: TraversalDirection = TraversalDirection.BOTH,
        scopes: set[MemoryScope] | None = None,
        owner: str | None = None,
        include_folded: bool = False,
        statuses: set[MemoryStatus] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> list[SearchHit]:
        self._require_node(node_id)
        direction = TraversalDirection(direction)
        queue: deque[
            tuple[str, int, RelationType | None, TraversalDirection | None]
        ] = deque([(node_id, 0, None, None)])
        visited = {node_id}
        hits: list[SearchHit] = []
        visibility = {
            "scopes": scopes,
            "owner": owner,
            "include_folded": include_folded,
            "statuses": statuses,
            "workspace_id": workspace_id,
            "project_id": project_id,
            "session_id": session_id,
        }

        while queue:
            current_id, current_depth, via_relation, via_direction = queue.popleft()
            node = self.nodes[current_id]
            if reinforce:
                self._reinforce_node(node)
            hits.append(
                SearchHit(
                    node=node,
                    depth=current_depth,
                    via_relation=via_relation,
                    via_direction=via_direction,
                )
            )

            if current_depth >= depth:
                continue

            for edge, neighbor_id, step_direction in self._traversable_edges(current_id, direction):
                if edge.weight < min_weight:
                    continue
                if relation_types is not None and edge.relation_type not in relation_types:
                    continue
                if neighbor_id in visited:
                    continue
                neighbor = self.nodes[neighbor_id]
                if not self.is_visible(neighbor, **visibility):
                    visited.add(neighbor_id)
                    continue
                visited.add(neighbor_id)
                queue.append(
                    (neighbor_id, current_depth + 1, edge.relation_type, step_direction)
                )

        return hits

    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: set[str] | None = None,
        min_score: float = 0.0,
        include_roots: bool = False,
        scopes: set[MemoryScope] | None = None,
        owner: str | None = None,
        include_folded: bool = False,
        statuses: set[MemoryStatus] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        exclude_categories: set[str] | None = None,
    ) -> list[ScoredHit]:
        """Rank all memory nodes by relevance to ``query``.

        Read-only: unlike ``search_from`` it never reinforces nodes. ``scopes``
        and ``owner`` keep one owner's personal memories out of another's
        results; folded nodes are excluded unless asked for, since their content
        is represented by the node that absorbed them. Workspace and project
        are exact-match labels: omitting them only sees unlabelled memories.
        """
        return rank_nodes(
            self._visible_nodes(
                scopes=scopes,
                owner=owner,
                include_folded=include_folded,
                statuses=statuses,
                workspace_id=workspace_id,
                project_id=project_id,
                session_id=session_id,
            ),
            query,
            top_k=top_k,
            categories=categories,
            exclude_categories=exclude_categories,
            min_score=min_score,
            include_roots=include_roots,
        )

    def _visible_nodes(
        self,
        scopes: set[MemoryScope] | None = None,
        owner: str | None = None,
        include_folded: bool = False,
        statuses: set[MemoryStatus] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryNode]:
        """Nodes a given caller is allowed to recall.

        Knowledge is shared by definition, so it stays visible regardless of
        ``owner``; personal memories are only visible to their own owner.
        """
        allowed_statuses = statuses if statuses is not None else {MemoryStatus.ACTIVE}
        visible: list[MemoryNode] = []
        for node in self.nodes.values():
            if self.is_visible(
                node,
                scopes=scopes,
                owner=owner,
                include_folded=include_folded,
                statuses=allowed_statuses,
                workspace_id=workspace_id,
                project_id=project_id,
                session_id=session_id,
            ):
                visible.append(node)
        return visible

    def is_visible(
        self,
        node: MemoryNode,
        *,
        scopes: set[MemoryScope] | None = None,
        owner: str | None = None,
        include_folded: bool = False,
        statuses: set[MemoryStatus] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Return whether ``node`` may be recalled by this caller."""
        allowed_statuses = statuses if statuses is not None else {MemoryStatus.ACTIVE}
        if node.status not in allowed_statuses:
            return False
        if not include_folded and node.folded_into is not None:
            return False
        if scopes is not None and node.scope not in scopes:
            return False
        if node.scope != MemoryScope.KNOWLEDGE and node.owner != owner:
            return False
        if node.scope == MemoryScope.SESSION:
            caller_session = optional_id(session_id)
            if not node.session_id or not caller_session or node.session_id != caller_session:
                return False
        if node.scope == MemoryScope.KNOWLEDGE:
            if node.workspace_id is not None and node.workspace_id != optional_id(workspace_id):
                return False
            if node.project_id is not None and node.project_id != optional_id(project_id):
                return False
            return True
        if node.workspace_id != optional_id(workspace_id):
            return False
        if node.project_id != optional_id(project_id):
            return False
        return True

    def _identity_key(self, node: MemoryNode) -> tuple:
        return (node.scope, node.owner, node.workspace_id, node.project_id)

    def _promotion_identity(
        self,
        supporters: list[MemoryNode],
        workspace_id: str | None,
        project_id: str | None,
    ) -> tuple[str | None, str | None]:
        workspaces = {supporter.workspace_id for supporter in supporters}
        projects = {supporter.project_id for supporter in supporters}
        if len(workspaces) != 1 or len(projects) != 1:
            raise ValueError(
                "promotion supporters must share one workspace_id and one project_id"
            )
        inferred_workspace = next(iter(workspaces))
        inferred_project = next(iter(projects))
        requested_workspace = optional_id(workspace_id)
        requested_project = optional_id(project_id)
        if requested_workspace is not None and requested_workspace != inferred_workspace:
            raise ValueError("promotion workspace_id does not match the supporting memories")
        if requested_project is not None and requested_project != inferred_project:
            raise ValueError("promotion project_id does not match the supporting memories")
        return inferred_workspace, inferred_project

    def nodes_by_tree(self, tree_id: str) -> list[MemoryNode]:
        self._require_tree(tree_id)
        return sorted(
            (node for node in self.nodes.values() if tree_id in node.tree_ids),
            key=lambda node: node.created_at,
        )

    def _compress_nodes(
        self,
        category: str,
        source_layer: str,
        target_layer: str,
        source_nodes: list[MemoryNode],
    ) -> str:
        if source_nodes and len({self._identity_key(node) for node in source_nodes}) != 1:
            return ""
        title = f"{category} {source_layer} compression ({len(source_nodes)} nodes)"
        summary = self._summarize(source_nodes)
        node_type = NodeType.COMPRESSED if target_layer == "compressed" else NodeType.SUMMARY
        if target_layer == "long_term":
            node_type = NodeType.ARCHIVE

        compressed = MemoryNode(
            title=title,
            content=summary,
            category=category,
            node_type=node_type,
            layer=target_layer,
            tags=set().union(*(node.tags for node in source_nodes)) if source_nodes else set(),
            metadata={
                "source_layer": source_layer,
                "source_node_ids": [node.id for node in source_nodes],
                "source_memory_weight": sum(node.memory_weight for node in source_nodes),
            },
            scope=source_nodes[0].scope if source_nodes else MemoryScope.PERSONAL,
            owner=source_nodes[0].owner if source_nodes else None,
            workspace_id=source_nodes[0].workspace_id if source_nodes else None,
            project_id=source_nodes[0].project_id if source_nodes else None,
            session_id=source_nodes[0].session_id if source_nodes else None,
            task_id=source_nodes[0].task_id if source_nodes else None,
            confidence=max((node.confidence for node in source_nodes), default=0.0),
            memory_weight=sum(node.memory_weight for node in source_nodes),
        )

        tree_id = self.create_tree(category)
        compressed.tree_ids.add(tree_id)
        self._add_node(compressed)
        self._record(EventOp.CREATE_NODE, {"node": compressed.to_dict()})

        folded_ids: list[str] = []
        for source in source_nodes:
            self.link_nodes(compressed.id, source.id, RelationType.SUMMARY_OF, weight=0.8)
            if source.layer == source_layer:
                # Mark the source as absorbed rather than rewriting its layer and
                # type. The original stays intact so it can be re-distilled by a
                # later, better summarizer; only its visibility changes.
                source.folded_into = compressed.id
                folded_ids.append(source.id)

        self._record(
            EventOp.FOLD,
            {
                "into_node_id": compressed.id,
                "source_node_ids": folded_ids,
                "source_layer": source_layer,
                "target_layer": target_layer,
            },
        )
        return compressed.id

    def _archive_nodes(self, nodes: list[MemoryNode]) -> None:
        for node in nodes:
            node.node_type = NodeType.ARCHIVE
            self._set_node_layer(node, "archive")
            node.touch()

    def _summarize(self, nodes: list[MemoryNode]) -> str:
        """Concatenate truncated excerpts of the folded nodes.

        This is a placeholder, not distillation: the output sits at the same
        level of abstraction as its inputs, merely shorter. Real promotion of
        specifics into general claims goes through ``promote_to_knowledge``,
        which requires supporting evidence and keeps provenance pointers.
        """
        lines = []
        for node in nodes:
            excerpt = node.content.strip().replace("\n", " ")
            if len(excerpt) > 120:
                excerpt = f"{excerpt[:117]}..."
            lines.append(f"- {node.title}: {excerpt}")
        return "\n".join(lines)

    def _nodes_in_category_layer(self, category: str, layer: str) -> list[MemoryNode]:
        node_ids = self.by_cat_layer.get((category, layer), set())
        return sorted(
            (
                self.nodes[node_id]
                for node_id in node_ids
                if self.nodes[node_id].node_type != NodeType.ROOT
                and self.nodes[node_id].status == MemoryStatus.ACTIVE
                and self.nodes[node_id].folded_into is None
            ),
            key=self._retention_key,
        )

    def _select_overflow_nodes(self, nodes: list[MemoryNode], overflow_weight: int) -> list[MemoryNode]:
        selected: list[MemoryNode] = []
        selected_weight = 0
        for node in nodes:
            selected.append(node)
            selected_weight += node.memory_weight
            if selected_weight >= overflow_weight:
                break
        return selected

    def _traversable_edges(
        self, node_id: str, direction: TraversalDirection
    ) -> list[tuple[Edge, str, TraversalDirection]]:
        candidates: dict[str, tuple[Edge, str, TraversalDirection]] = {}
        if direction in {TraversalDirection.OUT, TraversalDirection.BOTH}:
            for edge_id in self.adjacency.get(node_id, set()):
                edge = self.edges.get(edge_id)
                if edge is None:
                    continue
                neighbor_id = (
                    edge.to_node_id if edge.from_node_id == node_id else edge.from_node_id
                )
                candidates[edge_id] = (edge, neighbor_id, TraversalDirection.OUT)
        if direction in {TraversalDirection.IN, TraversalDirection.BOTH}:
            for edge_id in self.reverse_adjacency.get(node_id, set()):
                edge = self.edges.get(edge_id)
                if edge is None or edge_id in candidates:
                    continue
                neighbor_id = (
                    edge.from_node_id if edge.to_node_id == node_id else edge.to_node_id
                )
                candidates[edge_id] = (edge, neighbor_id, TraversalDirection.IN)
        return sorted(
            candidates.values(),
            key=lambda item: (-item[0].weight, item[0].created_at, item[0].id),
        )

    def _add_node(self, node: MemoryNode) -> None:
        self.nodes[node.id] = node
        self._index_node(node)

    def _add_edge(self, edge: Edge) -> None:
        self.edges[edge.id] = edge
        self.adjacency.setdefault(edge.from_node_id, set()).add(edge.id)
        self.reverse_adjacency.setdefault(edge.to_node_id, set()).add(edge.id)
        if edge.direction == EdgeDirection.BIDIRECTIONAL:
            self.adjacency.setdefault(edge.to_node_id, set()).add(edge.id)
            self.reverse_adjacency.setdefault(edge.from_node_id, set()).add(edge.id)

    def _remove_edge(self, edge_id: str) -> None:
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return
        for index in (self.adjacency, self.reverse_adjacency):
            for node_id in (edge.from_node_id, edge.to_node_id):
                bucket = index.get(node_id)
                if bucket is None:
                    continue
                bucket.discard(edge_id)
                if not bucket:
                    index.pop(node_id, None)

    def _index_node(self, node: MemoryNode) -> None:
        self.by_cat_layer.setdefault((node.category, node.layer), set()).add(node.id)

    def _unindex_node(self, node: MemoryNode) -> None:
        bucket = self.by_cat_layer.get((node.category, node.layer))
        if bucket is None:
            return
        bucket.discard(node.id)
        if not bucket:
            self.by_cat_layer.pop((node.category, node.layer), None)

    def _set_node_layer(self, node: MemoryNode, layer: str) -> None:
        if node.layer == layer:
            return
        self._unindex_node(node)
        node.layer = layer
        self._index_node(node)

    def _retention_key(self, node: MemoryNode) -> tuple[float, int, float, float]:
        """Fold order: least-evidenced first, familiarity only as a tiebreak.

        Refuted memories are folded away before unproven ones, and heavily
        recalled memories get no protection from evidence they never earned.
        """
        return (
            node.confidence,
            node.familiarity,
            node.updated_at.timestamp(),
            node.created_at.timestamp(),
        )

    def _reinforce_node(self, node: MemoryNode, amount: float = 0.02) -> None:
        """Register that a memory was recalled.

        Bumps familiarity only. Confidence is deliberately untouched: recall is
        not evidence, and treating it as such is what turns a memory store into
        an echo chamber.
        """
        node.touch()
        node.familiarity = min(1.0, node.familiarity + amount)

    def rebuild_indices(self) -> None:
        self.adjacency = {}
        self.reverse_adjacency = {}
        self.by_cat_layer = {}
        for node in self.nodes.values():
            self._index_node(node)
        for edge in self.edges.values():
            self.adjacency.setdefault(edge.from_node_id, set()).add(edge.id)
            self.reverse_adjacency.setdefault(edge.to_node_id, set()).add(edge.id)
            if edge.direction == EdgeDirection.BIDIRECTIONAL:
                self.adjacency.setdefault(edge.to_node_id, set()).add(edge.id)
                self.reverse_adjacency.setdefault(edge.from_node_id, set()).add(edge.id)

    @classmethod
    def rebuild_from_log(
        cls,
        events: Iterable[MemoryEvent],
        layer_policy: FibonacciLayerPolicy = DEFAULT_LAYER_POLICY,
    ) -> "FibMind":
        """Reconstruct a forest by replaying its event log.

        This is what makes the log the source of truth rather than a side
        record: node and edge tables are a cache that can be thrown away and
        rebuilt. Recall counters (``familiarity``, ``access_count``) are not
        replayed — they describe how the store was used, not what it knows, and
        are the one part that is expected to be lost.
        """
        memory = cls(layer_policy=layer_policy)
        for event in sorted(events, key=lambda item: item.seq):
            memory._apply(event)
        memory.events = sorted(events, key=lambda item: item.seq)
        memory.rebuild_indices()
        return memory

    def _apply(self, event: MemoryEvent) -> None:
        payload = event.payload
        if event.op in {EventOp.CREATE_NODE, EventOp.CREATE_TREE}:
            node_payload = payload.get("node") or payload.get("root")
            if node_payload is not None:
                node = MemoryNode.from_dict(node_payload)
                node.familiarity = 0.0
                node.access_count = 0
                self.nodes[node.id] = node
            tree_id = payload.get("tree_id")
            if tree_id is not None:
                root_id = payload.get("existing_root_node_id") or (
                    node_payload["id"] if node_payload else None
                )
                if root_id is not None:
                    created_at = payload.get("created_at")
                    self.trees[tree_id] = MemoryTree(
                        id=tree_id,
                        category=payload["category"],
                        title=payload.get("title", payload["category"]),
                        root_node_id=root_id,
                        created_at=(
                            datetime.fromisoformat(created_at)
                            if created_at is not None
                            else event.created_at
                        ),
                    )
                    self.category_roots[payload["category"]] = tree_id
                    if root_id in self.nodes:
                        self.nodes[root_id].tree_ids.add(tree_id)
        elif event.op == EventOp.LINK:
            edge = Edge.from_dict(payload["edge"])
            self.edges[edge.id] = edge
        elif event.op == EventOp.FOLD:
            for node_id in payload.get("source_node_ids", []):
                if node_id in self.nodes:
                    self.nodes[node_id].folded_into = payload["into_node_id"]
        elif event.op == EventOp.REVISE:
            node = self.nodes.get(payload["node_id"])
            after = payload.get("after")
            if node is not None and isinstance(after, dict) and "title" in after:
                node.title = after["title"]
                node.content = after["content"]
                node.tags = set(after.get("tags", []))
                node.confidence = 0.0
                node.confidence_source = None
                node.status = MemoryStatus(after.get("status", MemoryStatus.ACTIVE.value))
                node.status_reason = None
        elif event.op == EventOp.OBSERVE:
            node = self.nodes.get(payload["node_id"])
            if node is not None:
                node.confidence = float(payload["confidence"])
                node.confidence_source = payload.get("source")
                status = payload.get("status")
                if status is not None:
                    node.status = MemoryStatus(status)
                    node.status_reason = payload.get("status_reason")
                elif payload.get("verdict") == Verdict.REFUTED.value:
                    node.status = MemoryStatus.REFUTED
                    node.status_reason = payload.get("note") or f"refuted by {payload.get('source')}"
        elif event.op == EventOp.SET_STATUS:
            node = self.nodes.get(payload["node_id"])
            if node is not None:
                node.status = MemoryStatus(payload["status"])
                node.status_reason = payload.get("reason")
        elif event.op == EventOp.PROMOTE:
            node = self.nodes.get(payload["node_id"])
            node_type = payload.get("node_type")
            if node is not None and node_type is not None:
                node.node_type = NodeType(node_type)
        elif event.op == EventOp.FORGET:
            node_id = payload["node_id"]
            self.nodes.pop(node_id, None)
            for edge_id in payload.get("removed_edges", []):
                self.edges.pop(edge_id, None)
            for tree_id, tree in list(self.trees.items()):
                if tree.root_node_id == node_id:
                    self.trees.pop(tree_id)
                    self.category_roots.pop(tree.category, None)

    def _tree_for_root(self, node_id: str) -> MemoryTree | None:
        for tree in self.trees.values():
            if tree.root_node_id == node_id:
                return tree
        return None

    def _require_node(self, node_id: str) -> MemoryNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown node: {node_id}") from exc

    def _require_tree(self, tree_id: str) -> MemoryTree:
        try:
            return self.trees[tree_id]
        except KeyError as exc:
            raise KeyError(f"Unknown tree: {tree_id}") from exc
