"""Deterministic write gate for memories.

This is the Brain ``admit`` policy: not everything that happened is worth
keeping. The checks are conservative and do not call a model. A later
summarizer can still fold what does get written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from fibmind.graph import FibMind
from fibmind.models import MemoryNode, MemoryScope, MemoryStatus, optional_id

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

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "duplicate_node_id": self.duplicate_node_id,
        }


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
    if candidate.scope == MemoryScope.SESSION and optional_id(candidate.session_id) is None:
        return AdmitDecision(AdmitVerdict.SKIP, "session-scoped memories require session_id")

    dump_reason = _dump_reason(candidate.content)
    if dump_reason is not None:
        return AdmitDecision(AdmitVerdict.SKIP, dump_reason)

    if _is_empty_speculation(candidate):
        return AdmitDecision(AdmitVerdict.SKIP, "short speculation without a conclusion")

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


def _duplicate_of(memory: FibMind, candidate: MemoryCandidate) -> MemoryNode | None:
    needle = fingerprint(candidate.title, candidate.content)
    workspace_id = optional_id(candidate.workspace_id)
    project_id = optional_id(candidate.project_id)
    session_id = optional_id(candidate.session_id)
    live = {MemoryStatus.ACTIVE, MemoryStatus.PENDING}
    for node in memory.nodes.values():
        if node.status not in live or node.folded_into is not None:
            continue
        if node.scope != candidate.scope:
            continue
        if node.owner != candidate.owner:
            continue
        if node.workspace_id != workspace_id or node.project_id != project_id:
            continue
        if candidate.scope == MemoryScope.SESSION and node.session_id != session_id:
            continue
        if fingerprint(node.title, node.content) == needle:
            return node
    return None
