"""Deterministic write gate for memories.

This is the Brain ``admit`` policy: not everything that happened is worth
keeping. The checks are conservative and do not call a model. A later
summarizer can still fold what does get written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fibmind.graph import FibMind
from fibmind.models import MemoryNode, MemoryScope, MemoryStatus, optional_id
from fibmind.safety import scan_candidate

# Raw dumps belong in logs, not long-term memory.
MAX_CONTENT_CHARS = 12_000
MAX_CONTENT_LINES = 80
LOG_LINE = re.compile(
    r"^(?:\s*(?:at |\t)|\[\d{2}:|Traceback|DEBUG |INFO |ERROR |WARN |WARNING )",
    re.IGNORECASE,
)
# Very short hedges with no conclusion are not memories.
SPECULATION = re.compile(
    r"^(?:maybe|perhaps|probably|might |i think|也许|可能|大概|感觉)\b",
    re.IGNORECASE,
)
MIN_SPECULATION_KEEP_CHARS = 80


class AdmitVerdict(StrEnum):
    WRITE = "write"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    category: str
    title: str
    content: str
    tags: tuple[str, ...] = ()
    scope: MemoryScope = MemoryScope.PERSONAL
    owner: str | None = None
    workspace_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class AdmitDecision:
    verdict: AdmitVerdict
    reason: str
    duplicate_node_id: str | None = None
    findings: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        data = {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "duplicate_node_id": self.duplicate_node_id,
        }
        if self.findings:
            data["findings"] = list(self.findings)
        return data


def fingerprint(title: str, content: str) -> str:
    """Normalize title+body so exact restatements collapse."""
    return " ".join(f"{title}\n{content}".split()).casefold()


def decide_admission(memory: FibMind, candidate: MemoryCandidate) -> AdmitDecision:
    """Return whether ``candidate`` should be written into ``memory``."""
    if candidate.scope == MemoryScope.KNOWLEDGE:
        return AdmitDecision(
            AdmitVerdict.SKIP,
            "knowledge is created through promote_to_knowledge, not remember",
        )
    if (
        candidate.scope == MemoryScope.SESSION
        and optional_id(candidate.session_id) is None
        and optional_id(candidate.task_id) is None
    ):
        return AdmitDecision(AdmitVerdict.SKIP, "session-scoped memories require session_id or task_id")

    dump_reason = _dump_reason(candidate.content)
    if dump_reason is not None:
        return AdmitDecision(AdmitVerdict.SKIP, dump_reason)

    if _is_empty_speculation(candidate):
        return AdmitDecision(AdmitVerdict.SKIP, "short speculation without a conclusion")

    safety = scan_candidate(candidate.title, candidate.content, candidate.tags)
    if safety.blocked:
        return AdmitDecision(AdmitVerdict.SKIP, safety.reason(), findings=tuple(f.to_dict() for f in safety.findings))

    match = _duplicate_of(memory, candidate)
    if match is not None:
        return AdmitDecision(
            AdmitVerdict.SKIP,
            "duplicate of an active memory in this working context",
            duplicate_node_id=match.id,
        )
    return AdmitDecision(AdmitVerdict.WRITE, "admitted")


def _dump_reason(content: str) -> str | None:
    if len(content) > MAX_CONTENT_CHARS:
        return "content looks like a raw dump (too long for long-term memory)"
    lines = content.splitlines()
    if len(lines) < MAX_CONTENT_LINES:
        return None
    log_like = sum(1 for line in lines if LOG_LINE.match(line))
    if log_like / len(lines) >= 0.4:
        return "content looks like a log or stack trace"
    return None


def _is_empty_speculation(candidate: MemoryCandidate) -> bool:
    text = candidate.content.strip()
    if len(text) >= MIN_SPECULATION_KEEP_CHARS:
        return False
    return bool(SPECULATION.match(text) or SPECULATION.match(candidate.title.strip()))


NEAR_DUPLICATE_JACCARD = 0.85
_NEGATION_TOKENS = frozenset(
    {
        "not",
        "never",
        "avoid",
        "dont",
        "don't",
        "don",
        "no",
        "without",
        "disable",
        "disabled",
        "deny",
        "refute",
        "禁止",
        "不要",
        "切勿",
        "不",
        "避免",
        "禁用",
        "关闭",
    }
)


def detect_potential_conflicts(
    memory: FibMind,
    candidate: MemoryCandidate,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Sniff potential semantic conflicts between candidate and visible active memories."""
    from fibmind.ranking import tokenize
    from fibmind.visibility import is_visible

    cand_tokens = set(tokenize(f"{candidate.title} {candidate.content}"))
    cand_neg = bool(cand_tokens & _NEGATION_TOKENS)
    cand_core = cand_tokens - _NEGATION_TOKENS
    if not cand_core:
        return []

    cand_files: set[str] = set()
    if isinstance(metadata, dict):
        raw_files = metadata.get("files") or metadata.get("changed_files") or ()
        if isinstance(raw_files, (list, tuple, set)):
            cand_files = {str(f).strip() for f in raw_files if f and str(f).strip()}

    workspace_id = optional_id(candidate.workspace_id)
    project_id = optional_id(candidate.project_id)
    session_id = optional_id(candidate.session_id)
    task_id = optional_id(candidate.task_id)

    conflicts: list[str] = []
    for node in memory.nodes.values():
        if node.status != MemoryStatus.ACTIVE or node.folded_into is not None:
            continue
        exact_context = node.workspace_id == workspace_id and node.project_id == project_id
        broader_visible = is_visible(
            node,
            scopes={candidate.scope, node.scope},
            owner=candidate.owner,
            statuses={MemoryStatus.ACTIVE},
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
        )
        if not (exact_context or broader_visible):
            continue

        node_files: set[str] = set()
        if isinstance(node.metadata, dict):
            raw_node_files = node.metadata.get("files") or node.metadata.get("changed_files") or ()
            if isinstance(raw_node_files, (list, tuple, set)):
                node_files = {str(f).strip() for f in raw_node_files if f and str(f).strip()}

        shared_files = cand_files & node_files
        if node.category != candidate.category and not shared_files:
            continue

        node_tokens = set(tokenize(f"{node.title} {node.content}"))
        node_neg = bool(node_tokens & _NEGATION_TOKENS)
        if cand_neg == node_neg:
            continue

        node_core = node_tokens - _NEGATION_TOKENS
        intersection = len(cand_core & node_core)
        union = len(cand_core | node_core)
        jaccard = (intersection / union) if union > 0 else 0.0

        if jaccard >= 0.40 or (len(shared_files) >= 1 and intersection >= 2):
            conflicts.append(node.id)

    return conflicts


def detect_version_predecessors(
    memory: FibMind,
    candidate: MemoryCandidate,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Detect active memories that candidate appears to supersede as a newer version."""
    from fibmind.ranking import tokenize
    from fibmind.visibility import is_visible

    cand_fp = fingerprint(candidate.title, candidate.content)
    cand_tokens = set(tokenize(f"{candidate.title} {candidate.content}"))
    cand_neg = bool(cand_tokens & _NEGATION_TOKENS)
    cand_core = cand_tokens - _NEGATION_TOKENS
    cand_title_norm = " ".join(candidate.title.split()).casefold()

    cand_files: set[str] = set()
    if isinstance(metadata, dict):
        raw_files = metadata.get("files") or metadata.get("changed_files") or ()
        if isinstance(raw_files, (list, tuple, set)):
            cand_files = {str(f).strip() for f in raw_files if f and str(f).strip()}

    workspace_id = optional_id(candidate.workspace_id)
    project_id = optional_id(candidate.project_id)
    session_id = optional_id(candidate.session_id)
    task_id = optional_id(candidate.task_id)

    predecessors: list[str] = []
    for node in memory.nodes.values():
        if node.status != MemoryStatus.ACTIVE or node.folded_into is not None:
            continue
        if node.scope != candidate.scope or node.owner != candidate.owner:
            continue
        if node.category != candidate.category:
            continue

        exact_context = node.workspace_id == workspace_id and node.project_id == project_id
        broader_visible = is_visible(
            node,
            scopes={candidate.scope, node.scope},
            owner=candidate.owner,
            statuses={MemoryStatus.ACTIVE},
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
        )
        if not (exact_context or broader_visible):
            continue

        node_fp = fingerprint(node.title, node.content)
        if node_fp == cand_fp:
            continue

        node_tokens = set(tokenize(f"{node.title} {node.content}"))
        node_neg = bool(node_tokens & _NEGATION_TOKENS)
        if cand_neg != node_neg:
            continue

        node_title_norm = " ".join(node.title.split()).casefold()
        matched_a = bool(cand_title_norm and cand_title_norm == node_title_norm and node_fp != cand_fp)

        node_files: set[str] = set()
        if isinstance(node.metadata, dict):
            raw_node_files = node.metadata.get("files") or node.metadata.get("changed_files") or ()
            if isinstance(raw_node_files, (list, tuple, set)):
                node_files = {str(f).strip() for f in raw_node_files if f and str(f).strip()}

        shared_files = cand_files & node_files
        node_core = node_tokens - _NEGATION_TOKENS
        union = len(cand_core | node_core)
        jaccard = (len(cand_core & node_core) / union) if union > 0 else 0.0
        matched_b = bool(shared_files) and (jaccard >= 0.45)

        if matched_a or matched_b:
            predecessors.append(node.id)

    return sorted(predecessors)


def discover_knowledge_clusters(
    memory: FibMind,
    *,
    owner: str | None = None,
    workspace_id: str | None = None,
    project_id: str | None = None,
    min_supporters: int = 3,
    min_sessions: int = 2,
) -> list[dict[str, Any]]:
    """Discover cross-session candidate clusters for promotion to shared knowledge."""
    from fibmind.models import RelationType
    from fibmind.ranking import tokenize

    already_promoted: set[str] = set()
    for k_node in memory.nodes.values():
        if k_node.scope == MemoryScope.KNOWLEDGE and k_node.status == MemoryStatus.ACTIVE:
            supp_ids = k_node.metadata.get("supporting_node_ids")
            if isinstance(supp_ids, (list, tuple, set)):
                already_promoted.update(str(s) for s in supp_ids)
            for edge_id in memory.adjacency.get(k_node.id, set()):
                edge = memory.edges.get(edge_id)
                if edge and edge.relation_type == RelationType.DERIVED_FROM and edge.from_node_id == k_node.id:
                    already_promoted.add(edge.to_node_id)

    excluded_categories = {"episode", "hot", "goal", "procedure", "profile", "preference"}
    target_workspace = optional_id(workspace_id)
    target_project = optional_id(project_id)

    def _extract_anchors(node: MemoryNode) -> set[str]:
        anchors: set[str] = set()
        if isinstance(node.metadata, dict):
            for k in ("files", "changed_files", "symbols"):
                val = node.metadata.get(k)
                if isinstance(val, (list, tuple, set)):
                    anchors.update(str(x).strip() for x in val if x and str(x).strip())
                elif isinstance(val, str) and val.strip():
                    anchors.add(val.strip())
        return anchors

    def _extract_core_tokens(node: MemoryNode) -> set[str]:
        tokens = set(tokenize(f"{node.title} {node.content}"))
        return tokens - _NEGATION_TOKENS

    groups: dict[tuple[str, str | None, str | None], list[MemoryNode]] = {}
    for node in memory.nodes.values():
        if node.status != MemoryStatus.ACTIVE or node.folded_into is not None:
            continue
        if node.scope != MemoryScope.PERSONAL:
            continue
        if node.category in excluded_categories:
            continue
        if node.node_type.value == "root":
            continue
        if node.confidence <= 0.0:
            continue
        if node.id in already_promoted:
            continue
        if owner is not None and node.owner != owner:
            continue
        if target_workspace is not None and node.workspace_id != target_workspace:
            continue
        if target_project is not None and node.project_id != target_project:
            continue

        key = (node.category, node.workspace_id, node.project_id)
        groups.setdefault(key, []).append(node)

    results: list[dict[str, Any]] = []

    for (cat, _ws, _prj), candidate_nodes in groups.items():
        sorted_nodes = sorted(candidate_nodes, key=lambda n: n.id)
        clusters: list[list[MemoryNode]] = []

        for cand in sorted_nodes:
            cand_anchors = _extract_anchors(cand)
            cand_core = _extract_core_tokens(cand)

            placed = False
            for cluster in clusters:
                seed = cluster[0]
                seed_anchors = _extract_anchors(seed)
                if cand_anchors and seed_anchors and bool(cand_anchors & seed_anchors):
                    cluster.append(cand)
                    placed = True
                    break
                seed_core = _extract_core_tokens(seed)
                union = len(seed_core | cand_core)
                jaccard = (len(seed_core & cand_core) / union) if union > 0 else 0.0
                if jaccard >= 0.30:
                    cluster.append(cand)
                    placed = True
                    break

            if not placed:
                clusters.append([cand])

        for cluster_nodes in clusters:
            if len(cluster_nodes) < min_supporters:
                continue

            distinct_sessions = {str(n.session_id) for n in cluster_nodes if n.session_id}
            distinct_hosts = {str(n.metadata.get("host")) for n in cluster_nodes if n.metadata.get("host")}

            if len(distinct_sessions) < min_sessions and len(distinct_hosts) < min_sessions:
                continue

            all_anchors = [_extract_anchors(n) for n in cluster_nodes]
            all_union = set.union(*all_anchors) if all_anchors else set()
            shared_anchors = {a for a in all_union if sum(1 for s in all_anchors if a in s) >= 2}

            results.append(
                {
                    "category": cat,
                    "supporting_node_ids": [n.id for n in cluster_nodes],
                    "sessions": sorted(distinct_sessions),
                    "shared_anchors": sorted(shared_anchors),
                    "suggested_title": f"Distilled {cat} pattern: {cluster_nodes[0].title[:60]}",
                    "suggested_content": " / ".join(n.content.strip() for n in cluster_nodes[:3]),
                }
            )

    return results


def propose_conflict_synthesis(
    node_a_or_memory: FibMind | MemoryNode,
    node_a_or_b: MemoryNode,
    node_b: MemoryNode | None = None,
) -> dict[str, Any]:
    """Propose distinguishing conditions and synthesis content for two conflicting memories."""
    if node_b is not None:
        node_a = node_a_or_b
        actual_b = node_b
    else:
        node_a = node_a_or_memory  # type: ignore[assignment]
        actual_b = node_a_or_b
    node_b = actual_b
    internal_tags = {"fibbrain-review", "synthesized", "fibbrain-episode"}

    def _extract_conditions(target: MemoryNode, other: MemoryNode) -> list[str]:
        conds: list[str] = []
        # 1. Exclusive tags: sorted(node_a.tags - node_b.tags) (excluding internal tags)
        excl_tags = sorted(target.tags - other.tags - internal_tags)
        conds.extend(excl_tags)

        # 2. Metadata conditions: node.metadata.get("conditions") or ("preconditions")
        meta = target.metadata or {}
        for key in ("conditions", "preconditions"):
            val = meta.get(key)
            if isinstance(val, (list, tuple, set)):
                for c in val:
                    s = str(c).strip()
                    if s:
                        conds.append(s)
            elif isinstance(val, str) and val.strip():
                conds.append(val.strip())

        # 3. Exclusive files: files in node_a.metadata.get("files") / changed_files not in node_b
        other_files: set[str] = set()
        other_meta = other.metadata or {}
        for key in ("files", "changed_files"):
            val = other_meta.get(key)
            if isinstance(val, (list, tuple, set)):
                other_files.update(str(f).strip() for f in val if str(f).strip())
            elif isinstance(val, str) and val.strip():
                other_files.add(val.strip())

        for key in ("files", "changed_files"):
            val = meta.get(key)
            if isinstance(val, (list, tuple, set)):
                for f in val:
                    s = str(f).strip()
                    if s and s not in other_files:
                        conds.append(s)
            elif isinstance(val, str) and val.strip():
                s = val.strip()
                if s not in other_files:
                    conds.append(s)

        # 4. Conditional clauses in node.content.splitlines() starting with ("when ", "if ", "under ", "requires ")
        prefixes = ("when ", "if ", "under ", "requires ")
        for line in target.content.splitlines():
            s = line.strip()
            if any(s.lower().startswith(p) for p in prefixes):
                conds.append(s)

        # Deduplicate while preserving order
        deduped: list[str] = []
        for c in conds:
            if c not in deduped:
                deduped.append(c)

        # 5. Fallback if empty: [f"context:{node.title}"]
        if not deduped:
            deduped = [f"context:{target.title}"]
        return deduped

    conditions_a = _extract_conditions(node_a, node_b)
    conditions_b = _extract_conditions(node_b, node_a)

    return {
        "node_id_a": node_a.id,
        "node_id_b": node_b.id,
        "title_a": node_a.title,
        "title_b": node_b.title,
        "conditions_a": conditions_a,
        "conditions_b": conditions_b,
        "suggested_title": f"Synthesized: {node_a.title} vs {node_b.title}",
        "suggested_content": (
            f"When {', '.join(conditions_a)}: {node_a.title}. When {', '.join(conditions_b)}: {node_b.title}."
        ),
    }


def _duplicate_of(memory: FibMind, candidate: MemoryCandidate) -> MemoryNode | None:
    from fibmind.ranking import tokenize
    from fibmind.visibility import is_visible

    needle = fingerprint(candidate.title, candidate.content)
    cand_tokens = set(tokenize(f"{candidate.title} {candidate.content}"))
    cand_negations = cand_tokens & _NEGATION_TOKENS
    workspace_id = optional_id(candidate.workspace_id)
    project_id = optional_id(candidate.project_id)
    session_id = optional_id(candidate.session_id)
    task_id = optional_id(candidate.task_id)
    live = {MemoryStatus.ACTIVE, MemoryStatus.PENDING}
    for node in memory.nodes.values():
        if node.status not in live or node.folded_into is not None:
            continue
        if node.scope != candidate.scope:
            continue
        if node.owner != candidate.owner:
            continue
        exact_context = node.workspace_id == workspace_id and node.project_id == project_id
        broader_visible = candidate.scope == MemoryScope.PERSONAL and is_visible(
            node,
            scopes={candidate.scope},
            owner=candidate.owner,
            statuses=live,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
        )
        if not (exact_context or broader_visible):
            continue
        if candidate.scope == MemoryScope.SESSION and node.session_id != session_id:
            continue
        if fingerprint(node.title, node.content) == needle:
            return node
        if node.category == candidate.category and len(cand_tokens) >= 4:
            node_tokens = set(tokenize(f"{node.title} {node.content}"))
            if len(node_tokens) >= 4 and (node_tokens & _NEGATION_TOKENS) == cand_negations:
                union = len(cand_tokens | node_tokens)
                if union > 0 and (len(cand_tokens & node_tokens) / union) >= NEAR_DUPLICATE_JACCARD:
                    return node
    return None
