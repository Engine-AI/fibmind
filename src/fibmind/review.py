"""Session review: turn a session's episode into long-term memory candidates.

The episode is what ``FibBrain.observe`` recorded while a task ran: tool
results, tests, errors, corrections, decisions. Most of it is not worth
keeping. This module extracts the parts that are — deterministically, without a
model — and hands each one to the admission gate.

Extraction is a pure function of the episode nodes and the session's goals, so
running it twice yields byte-identical candidates. Together with the
fingerprint check in ``admission`` that makes the review idempotent: a second
run skips every candidate as a duplicate and writes nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fibmind.models import MemoryKind, MemoryNode, MemoryScope
from fibmind.procedure import PROCEDURE_CATEGORY, PROCEDURE_TAG, procedure_from_session

EPISODE_CATEGORY = "episode"
EPISODE_TAG = "fibbrain-episode"
REVIEW_TAG = "fibbrain-review"
REVIEW_KIND = "session_review"

# Observation kinds the extractor understands. Anything else is only counted.
DECISION_KINDS = {"decision", "design", "choice"}
ERROR_KINDS = {"error", "failure", "exception", "bug"}
TEST_KINDS = {"test", "verify", "verification", "check"}
CORRECTION_KINDS = {"correction", "user_correction", "feedback"}
RISK_KINDS = {"risk", "unresolved", "todo", "open_question"}
STEP_KINDS = {"step", "action", "tool_result", "tool_call"}
FILE_KEYS = ("files", "changed_files", "paths", "path", "file")
COMMAND_KEYS = ("command", "cmd")
TOOL_KEYS = ("tool", "tool_name")
SYMBOL_KEYS = ("symbols", "symbol", "functions", "classes")

MAX_LIST_ITEMS = 12


class ReviewMode(StrEnum):
    CANDIDATES = "candidates"  # extract and admit-preview only; write nothing
    APPROVE = "approve"  # write admitted candidates as ``pending``
    AUTO = "auto"  # write admitted candidates as ``active``


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    category: str
    title: str
    content: str
    tags: tuple[str, ...]
    source_node_ids: tuple[str, ...]
    scope: MemoryScope = MemoryScope.PERSONAL
    memory_kind: MemoryKind | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "category": self.category,
            "title": self.title,
            "content": self.content,
            "tags": list(self.tags),
            "scope": self.scope.value,
            "memory_kind": self.memory_kind.value if self.memory_kind else None,
            "source_node_ids": list(self.source_node_ids),
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(slots=True)
class EpisodeDigest:
    """What the episode contained, grouped by what it tells us."""

    session_id: str
    objective: str | None = None
    goal_ids: tuple[str, ...] = ()
    decisions: list[MemoryNode] = field(default_factory=list)
    errors: list[MemoryNode] = field(default_factory=list)
    tests: list[MemoryNode] = field(default_factory=list)
    corrections: list[MemoryNode] = field(default_factory=list)
    risks: list[MemoryNode] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    file_sources: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    step_sources: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    assessments: list[MemoryNode] = field(default_factory=list)
    other_kinds: dict[str, int] = field(default_factory=dict)
    episode_ids: tuple[str, ...] = ()


def episode_kind(node: MemoryNode) -> str:
    return str(node.metadata.get("kind") or "").strip().casefold()


def digest_episode(
    session_id: str,
    episodes: Iterable[MemoryNode],
    goals: Iterable[MemoryNode] = (),
) -> EpisodeDigest:
    """Group episode nodes by kind and collect the files they touched."""
    ordered = sorted(episodes, key=lambda node: (node.created_at, node.id))
    digest = EpisodeDigest(session_id=session_id, episode_ids=tuple(node.id for node in ordered))

    goal_nodes = sorted(goals, key=lambda node: (node.created_at, node.id))
    if goal_nodes:
        digest.objective = goal_nodes[-1].title
        digest.goal_ids = tuple(node.id for node in goal_nodes)

    seen_files: set[str] = set()
    seen_symbols: set[str] = set()
    for node in ordered:
        kind = episode_kind(node)
        payload = node.metadata.get("payload") or {}
        for key in FILE_KEYS:
            value = payload.get(key) if isinstance(payload, dict) else None
            for path in _as_list(value):
                if path not in seen_files:
                    seen_files.add(path)
                    digest.files.append(path)
                    digest.file_sources.append(node.id)
        for key in SYMBOL_KEYS:
            value = payload.get(key) if isinstance(payload, dict) else None
            for sym in _as_list(value):
                if sym not in seen_symbols:
                    seen_symbols.add(sym)
                    digest.symbols.append(sym)
        for key in TOOL_KEYS:
            value = payload.get(key) if isinstance(payload, dict) else None
            for tool in _as_list(value):
                if tool not in digest.tools:
                    digest.tools.append(tool)
        if kind in DECISION_KINDS:
            digest.decisions.append(node)
        elif kind in ERROR_KINDS:
            digest.errors.append(node)
        elif kind in TEST_KINDS:
            digest.tests.append(node)
            _add_step(digest, node, prefix="Verify: ")
        elif kind in CORRECTION_KINDS:
            digest.corrections.append(node)
        elif kind in RISK_KINDS:
            digest.risks.append(node)
        elif kind in STEP_KINDS:
            _add_step(digest, node)
        elif kind == "assess":
            digest.assessments.append(node)
        else:
            digest.other_kinds[kind or "unknown"] = digest.other_kinds.get(kind or "unknown", 0) + 1
    return digest


def _add_step(digest: EpisodeDigest, node: MemoryNode, prefix: str = "") -> None:
    """A step is the command if one was recorded, else the observation text."""
    command = _command_of(node)
    text = f"{prefix}{command}" if command else f"{prefix}{_first_line(node.content)}"
    if text.strip() and text not in digest.steps:
        digest.steps.append(text)
        digest.step_sources.append(node.id)


def _node_anchors_meta(
    nodes: Iterable[MemoryNode],
    fallback_files: list[str] | None = None,
    fallback_symbols: list[str] | None = None,
) -> dict[str, Any]:
    files: list[str] = []
    symbols: list[str] = []
    commands: list[str] = []
    for node in nodes:
        payload = node.metadata.get("payload") or {}
        if isinstance(payload, dict):
            for key in FILE_KEYS:
                for path in _as_list(payload.get(key)):
                    if path not in files:
                        files.append(path)
            for key in SYMBOL_KEYS:
                for sym in _as_list(payload.get(key)):
                    if sym not in symbols:
                        symbols.append(sym)
        cmd = _command_of(node)
        if cmd and cmd not in commands:
            commands.append(cmd)
    if not files and fallback_files:
        files = list(fallback_files[:MAX_LIST_ITEMS])
    if not symbols and fallback_symbols:
        symbols = list(fallback_symbols[:MAX_LIST_ITEMS])
    meta: dict[str, Any] = {}
    if files:
        meta["files"] = files[:MAX_LIST_ITEMS]
    if symbols:
        meta["symbols"] = symbols[:MAX_LIST_ITEMS]
    if commands:
        meta["commands"] = commands[:MAX_LIST_ITEMS]
        meta["command"] = commands[0]
    return meta


def extract_candidates(digest: EpisodeDigest) -> list[ReviewCandidate]:
    """Deterministic candidates: one per decision / error / correction / risk,
    one verification record, one changed-files record, one session summary."""
    candidates: list[ReviewCandidate] = []
    label = digest.objective or f"session {digest.session_id}"
    base_tags = (REVIEW_TAG, f"session:{digest.session_id}")

    for node in digest.decisions:
        candidates.append(
            ReviewCandidate(
                category="decision",
                title=node.content.strip().splitlines()[0][:120] if node.content.strip() else node.title,
                content=_with_context(node.content, label),
                tags=base_tags,
                source_node_ids=(node.id,),
                metadata=_node_anchors_meta((node,), digest.files, digest.symbols),
            )
        )
    for node in digest.errors:
        candidates.append(
            ReviewCandidate(
                category="error",
                title=node.title,
                content=_with_context(node.content, label),
                tags=base_tags,
                source_node_ids=(node.id,),
                metadata=_node_anchors_meta((node,), digest.files, digest.symbols),
            )
        )
    for node in digest.corrections:
        candidates.append(
            ReviewCandidate(
                category="correction",
                title=node.title,
                content=_with_context(node.content, label),
                tags=base_tags,
                source_node_ids=(node.id,),
                metadata=_node_anchors_meta((node,), digest.files, digest.symbols),
            )
        )
    for node in digest.risks:
        candidates.append(
            ReviewCandidate(
                category="risk",
                title=node.title,
                content=_with_context(node.content, label),
                tags=base_tags,
                source_node_ids=(node.id,),
                metadata=_node_anchors_meta((node,), digest.files, digest.symbols),
            )
        )

    if digest.tests:
        lines = []
        for node in digest.tests[:MAX_LIST_ITEMS]:
            command = _command_of(node)
            lines.append(f"- {node.content.strip()}" + (f" (`{command}`)" if command else ""))
        candidates.append(
            ReviewCandidate(
                category="verification",
                title=f"Verification for {label}",
                content="Checks recorded while working on " + label + ":\n" + "\n".join(lines),
                tags=base_tags,
                source_node_ids=tuple(node.id for node in digest.tests),
                metadata=_node_anchors_meta(digest.tests, digest.files, digest.symbols),
            )
        )

    # Collect open questions from assess events that remain unresolved
    session_text_parts = [
        *(n.title + " " + n.content for n in digest.decisions),
        *(n.title + " " + n.content for n in digest.errors),
        *(n.title + " " + n.content for n in digest.tests),
        *(n.title + " " + n.content for n in digest.corrections),
        *(n.title + " " + n.content for n in digest.risks),
        *digest.files,
    ]
    session_covered_text = " ".join(session_text_parts).casefold()

    seen_questions: set[str] = set()
    for assess_node in digest.assessments:
        payload = assess_node.metadata.get("payload") or {}
        unknown_aspects = payload.get("unknown_aspects") or []
        for aspect in unknown_aspects:
            aspect_str = str(aspect).strip()
            if not aspect_str or aspect_str in seen_questions:
                continue
            aspect_lower = aspect_str.casefold()
            is_covered = (aspect_lower in session_covered_text) or any(
                len(w) > 2 and w in session_covered_text for w in aspect_lower.split()
            )
            if not is_covered:
                seen_questions.add(aspect_str)
                candidates.append(
                    ReviewCandidate(
                        category="question",
                        title=f"Unresolved question: {aspect_str}",
                        content=f"Aspect left unanswered during session for {label}: {aspect_str}",
                        tags=base_tags + ("question",),
                        source_node_ids=(assess_node.id,),
                        metadata={"aspect": aspect_str, "question": aspect_str},
                    )
                )

    if digest.files:
        shown = digest.files[:MAX_LIST_ITEMS]
        more = len(digest.files) - len(shown)
        content = f"Files touched while working on {label}:\n" + "\n".join(f"- {path}" for path in shown)
        if more > 0:
            content += f"\n- … and {more} more"
        code_meta: dict[str, Any] = {"files": list(shown)}
        if digest.symbols:
            code_meta["symbols"] = list(digest.symbols[:MAX_LIST_ITEMS])
        candidates.append(
            ReviewCandidate(
                category="code",
                title=f"Files changed for {label}",
                content=content,
                tags=base_tags,
                source_node_ids=tuple(dict.fromkeys(digest.file_sources)),
                metadata=code_meta,
            )
        )

    procedure = _procedure_candidate(digest, label, base_tags)
    if procedure is not None:
        candidates.append(procedure)

    if digest.episode_ids:
        candidates.append(_summary(digest, label, base_tags))
    return candidates


def _procedure_candidate(digest: EpisodeDigest, label: str, tags: tuple[str, ...]) -> ReviewCandidate | None:
    """A session that stated a goal, took steps, and verified them is a
    procedure candidate. Without verification it is just history."""
    if not digest.objective or not digest.steps or not digest.tests:
        return None
    verify = "; ".join((_command_of(node) or _first_line(node.content)) for node in digest.tests[:MAX_LIST_ITEMS])
    procedure = procedure_from_session(
        objective=digest.objective,
        steps=digest.steps[:MAX_LIST_ITEMS],
        tools=digest.tools,
        verify=verify,
    )
    if procedure is None:
        return None
    proc_meta: dict[str, Any] = {"trigger": procedure.trigger}
    if digest.files:
        proc_meta["files"] = list(digest.files[:MAX_LIST_ITEMS])
    if digest.symbols:
        proc_meta["symbols"] = list(digest.symbols[:MAX_LIST_ITEMS])
    if verify:
        proc_meta["verification"] = verify
    return ReviewCandidate(
        category=PROCEDURE_CATEGORY,
        title=f"How to: {label}",
        content=procedure.encode(),
        tags=tags + (PROCEDURE_TAG,),
        source_node_ids=tuple(dict.fromkeys(digest.step_sources + [node.id for node in digest.tests])),
        memory_kind=MemoryKind.PROCEDURE,
        metadata=proc_meta,
    )


def _summary(digest: EpisodeDigest, label: str, tags: tuple[str, ...]) -> ReviewCandidate:
    """One paragraph, facts first: the excerpt a recall shows is the head of it."""
    parts = [f"Objective: {label}."]
    if digest.decisions:
        parts.append(
            "Decisions: " + "; ".join(_first_line(node.content) for node in digest.decisions[:MAX_LIST_ITEMS]) + "."
        )
    if digest.errors:
        parts.append("Errors: " + "; ".join(_first_line(node.content) for node in digest.errors[:MAX_LIST_ITEMS]) + ".")
    if digest.risks:
        parts.append(
            "Unresolved: " + "; ".join(_first_line(node.content) for node in digest.risks[:MAX_LIST_ITEMS]) + "."
        )
    counts = []
    if digest.decisions:
        counts.append(f"{len(digest.decisions)} decision(s)")
    if digest.tests:
        counts.append(f"{len(digest.tests)} verification(s)")
    if digest.errors:
        counts.append(f"{len(digest.errors)} error(s)")
    if digest.corrections:
        counts.append(f"{len(digest.corrections)} correction(s)")
    if digest.risks:
        counts.append(f"{len(digest.risks)} open risk(s)")
    if digest.files:
        counts.append(f"{len(digest.files)} file(s) touched")
    if counts:
        parts.append("Recorded " + ", ".join(counts) + ".")
    if digest.other_kinds:
        noted = ", ".join(f"{kind}×{count}" for kind, count in sorted(digest.other_kinds.items()))
        parts.append(f"Also observed: {noted}.")
    summary_meta = _node_anchors_meta(digest.tests, digest.files, digest.symbols)
    return ReviewCandidate(
        category="summary",
        title=f"Session review: {label}",
        content=" ".join(parts),
        tags=tags,
        source_node_ids=digest.episode_ids,
        metadata=summary_meta,
    )


def _with_context(text: str, label: str) -> str:
    body = text.strip()
    if label.casefold() in body.casefold():
        return body
    return f"{body}\n\nContext: while working on {label}."


def _first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0][:160] if stripped else ""


def _command_of(node: MemoryNode) -> str | None:
    payload = node.metadata.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    for key in COMMAND_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []
