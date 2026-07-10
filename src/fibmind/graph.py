"""FibMind memory forest graph engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from fibmind.fibonacci import DEFAULT_LAYER_POLICY, FibonacciLayerPolicy
from fibmind.models import Edge, EdgeDirection, MemoryNode, MemoryTree, NodeType, RelationType
from fibmind.ranking import ScoredHit, rank_nodes


@dataclass(frozen=True, slots=True)
class SearchHit:
    node: MemoryNode
    depth: int
    via_relation: RelationType | None = None


class FibMind:
    """A prototype Fibonacci forest graph for long-term AI memory."""

    def __init__(self, layer_policy: FibonacciLayerPolicy = DEFAULT_LAYER_POLICY) -> None:
        self.layer_policy = layer_policy
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: dict[str, Edge] = {}
        self.trees: dict[str, MemoryTree] = {}
        self.category_roots: dict[str, str] = {}
        self.adjacency: dict[str, set[str]] = {}
        self.by_cat_layer: dict[tuple[str, str], set[str]] = {}

    def create_tree(self, category: str, title: str | None = None, content: str = "") -> str:
        if category in self.category_roots:
            return self.category_roots[category]

        root = MemoryNode(
            title=title or f"{category} root",
            content=content,
            category=category,
            node_type=NodeType.ROOT,
            layer="long_term",
            importance=1.0,
            memory_weight=0,
        )
        tree = MemoryTree(category=category, title=title or category, root_node_id=root.id)
        root.tree_ids.add(tree.id)

        self._add_node(root)
        self.trees[tree.id] = tree
        self.category_roots[category] = tree.id
        return tree.id

    def append(
        self,
        category: str,
        title: str,
        content: str,
        tags: Iterable[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
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
        )
        node.tree_ids.add(tree_id)
        self._add_node(node)

        self.link_nodes(
            tree.root_node_id,
            node.id,
            RelationType.TREE_CHILD,
            weight=1.0,
            direction=EdgeDirection.DIRECTED,
        )
        self.compress_overflow(category)
        return node.id

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
        return edge.id

    def promote_node(self, node_id: str, target_type: NodeType = NodeType.CONCEPT) -> str:
        node = self._require_node(node_id)
        node.node_type = target_type
        node.importance = max(node.importance, 0.75)
        self._reinforce_node(node, amount=0.05)

        if target_type in {NodeType.CONCEPT, NodeType.ROOT} and not self._tree_for_root(node_id):
            tree_id = self.expand_node_to_tree(node_id)
            return tree_id
        return node_id

    def expand_node_to_tree(self, node_id: str, title: str | None = None) -> str:
        node = self._require_node(node_id)
        category = f"{node.category}:{node.title}".lower().replace(" ", "_")
        original_category = category
        suffix = 2
        while category in self.category_roots:
            category = f"{original_category}_{suffix}"
            suffix += 1

        node.node_type = NodeType.ROOT
        node.importance = max(node.importance, 0.9)
        tree = MemoryTree(category=category, title=title or node.title, root_node_id=node.id)
        node.tree_ids.add(tree.id)
        self.trees[tree.id] = tree
        self.category_roots[category] = tree.id
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

        for hit in self.search_from(root_node_id, depth=10, relation_types={RelationType.TREE_CHILD}):
            hit.node.tree_ids.add(tree.id)
        return tree.id

    def compress_overflow(self, category: str) -> None:
        for layer in self.layer_policy.order:
            capacity = self.layer_policy.capacity_for(layer)
            nodes = self._nodes_in_category_layer(category, layer)
            total_weight = sum(node.memory_weight for node in nodes)
            if total_weight <= capacity:
                continue

            overflow = self._select_overflow_nodes(nodes, total_weight - capacity)
            next_layer = self.layer_policy.next_layer(layer)
            if next_layer is None:
                self._archive_nodes(overflow)
            else:
                self._compress_nodes(category, layer, next_layer, overflow)

    def search_from(
        self,
        node_id: str,
        depth: int = 2,
        relation_types: set[RelationType] | None = None,
        min_weight: float = 0.0,
        reinforce: bool = False,
    ) -> list[SearchHit]:
        self._require_node(node_id)
        queue: deque[tuple[str, int, RelationType | None]] = deque([(node_id, 0, None)])
        visited = {node_id}
        hits: list[SearchHit] = []

        while queue:
            current_id, current_depth, via_relation = queue.popleft()
            node = self.nodes[current_id]
            if reinforce:
                self._reinforce_node(node)
            hits.append(SearchHit(node=node, depth=current_depth, via_relation=via_relation))

            if current_depth >= depth:
                continue

            for edge in self._outgoing_edges(current_id):
                if edge.weight < min_weight:
                    continue
                if relation_types is not None and edge.relation_type not in relation_types:
                    continue
                neighbor_id = edge.neighbor_of(current_id)
                if neighbor_id is None or neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, current_depth + 1, edge.relation_type))

        return hits

    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: set[str] | None = None,
        min_score: float = 0.0,
        include_roots: bool = False,
    ) -> list[ScoredHit]:
        """Rank all memory nodes by keyword relevance to ``query``.

        This is read-only: unlike ``search_from`` it never reinforces nodes.
        """
        return rank_nodes(
            self.nodes.values(),
            query,
            top_k=top_k,
            categories=categories,
            min_score=min_score,
            include_roots=include_roots,
        )

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
            importance=max((node.importance for node in source_nodes), default=0.0),
            memory_weight=sum(node.memory_weight for node in source_nodes),
        )

        tree_id = self.create_tree(category)
        compressed.tree_ids.add(tree_id)
        self._add_node(compressed)

        for source in source_nodes:
            self.link_nodes(compressed.id, source.id, RelationType.SUMMARY_OF, weight=0.8)
            if source.layer == source_layer:
                self._set_node_layer(source, f"folded:{source_layer}")
                source.node_type = NodeType.ARCHIVE

        return compressed.id

    def _archive_nodes(self, nodes: list[MemoryNode]) -> None:
        for node in nodes:
            node.node_type = NodeType.ARCHIVE
            self._set_node_layer(node, "archive")
            node.touch()

    def _summarize(self, nodes: list[MemoryNode]) -> str:
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

    def _outgoing_edges(self, node_id: str) -> list[Edge]:
        edge_ids = self.adjacency.get(node_id, set())
        return [self.edges[edge_id] for edge_id in edge_ids if edge_id in self.edges]

    def _add_node(self, node: MemoryNode) -> None:
        self.nodes[node.id] = node
        self._index_node(node)

    def _add_edge(self, edge: Edge) -> None:
        self.edges[edge.id] = edge
        self.adjacency.setdefault(edge.from_node_id, set()).add(edge.id)
        if edge.direction == EdgeDirection.BIDIRECTIONAL:
            self.adjacency.setdefault(edge.to_node_id, set()).add(edge.id)

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
        return (
            node.importance,
            node.access_count,
            node.updated_at.timestamp(),
            node.created_at.timestamp(),
        )

    def _reinforce_node(self, node: MemoryNode, amount: float = 0.02) -> None:
        node.touch()
        node.importance = min(1.0, node.importance + amount)

    def rebuild_indices(self) -> None:
        self.adjacency = {}
        self.by_cat_layer = {}
        for node in self.nodes.values():
            self._index_node(node)
        for edge in self.edges.values():
            self.adjacency.setdefault(edge.from_node_id, set()).add(edge.id)
            if edge.direction == EdgeDirection.BIDIRECTIONAL:
                self.adjacency.setdefault(edge.to_node_id, set()).add(edge.id)

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
