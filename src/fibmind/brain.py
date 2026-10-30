"""FibBrain: the cognitive runtime that sits on top of the FibMind store.

Harnesses run the body (loop, tools, session). FibBrain decides what to
remember, what to load into a step, whether an action looks safe given past
evidence, which capabilities the body should involve, and how to record
whether a memory held up.

This is Brain v1 — Memory + Planning + Decision + Coordination + Reflection.
It does not own an agent loop or a plugin manager.
"""

from __future__ import annotations

import inspect
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from fibmind.admission import AdmitVerdict, MemoryCandidate, decide_admission
from fibmind.distill import DEFAULT_PLANNER, GapAssessor, Planner
from fibmind.graph import FibMind
from fibmind.models import (
    MemoryKind,
    MemoryNode,
    MemoryScope,
    MemoryStatus,
    RelationType,
    Verdict,
    optional_id,
    utc_now,
)
from fibmind.planning import (
    CAPABILITIES,
    CAPABILITY_ACTIONS,
    GOAL_CATEGORY,
    GOAL_TAG,
    BrainGoal,
    GoalStatus,
    PlanStep,
    encode_goal,
    infer_capabilities,
    is_goal_node,
    parse_goal_document,
)
from fibmind.procedure import (
    PROCEDURE_CATEGORY,
    PROCEDURE_TAG,
    Procedure,
    is_procedure_node,
    promotion_ready,
    render_skill,
    render_tool,
    retirement_due,
)
from fibmind.ranking import tokenize
from fibmind.review import (
    EPISODE_CATEGORY,
    EPISODE_TAG,
    REVIEW_KIND,
    REVIEW_TAG,
    ReviewCandidate,
    ReviewMode,
    digest_episode,
    extract_candidates,
)
from fibmind.service import MemoryService, _node_summary, _require_text
from fibmind.visibility import ContextLevel


class AdviseVerdict(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"
    WARN = "warn"


EPISODE_LIMIT = 48
GOAL_TITLE_LIMIT = 200
EPISODE_KIND = "fibbrain_episode"

# L0 hot memory: the standing preamble for a working context. Stable
# preferences, project conventions, and well-evidenced decisions — chosen by
# evidence and kind, never by how often they were read. The snapshot is frozen
# per session: it is computed on the first recall of a session and refreshed
# only by review_session, so a session sees one consistent preamble.
HOT_CATEGORY = "hot"
HOT_TAG = "fibbrain-hot"
HOT_KIND = "fibbrain_hot_snapshot"
HOT_MAX_ITEMS = 8
HOT_BUDGET_TOKENS = 300
DEFAULT_BUDGET_TOKENS = 500
HOT_KINDS = ("preference", "profile", "requirement", "decision", "knowledge")


@dataclass(frozen=True, slots=True)
class BrainState:
    """The working context of one recall / write / advise call."""

    owner: str | None = None
    workspace_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    model_id: str | None = None

    def identity(self) -> dict[str, str | None]:
        return {
            "owner": self.owner,
            "workspace_id": optional_id(self.workspace_id),
            "project_id": optional_id(self.project_id),
            "session_id": optional_id(self.session_id),
            "task_id": optional_id(self.task_id),
        }


@dataclass(frozen=True, slots=True)
class ObservedEvent:
    kind: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    node_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "node_id": self.node_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_node(cls, node: MemoryNode) -> ObservedEvent:
        payload = node.metadata.get("payload")
        return cls(
            kind=str(node.metadata.get("kind") or "unknown"),
            summary=node.content,
            payload=dict(payload) if isinstance(payload, dict) else {},
            created_at=node.created_at,
            node_id=node.id,
            session_id=node.session_id,
        )


@dataclass(frozen=True, slots=True)
class AdviseDecision:
    verdict: AdviseVerdict
    reason: str
    memories: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "memories": list(self.memories),
        }


def _auto_ground(metadata: dict[str, Any] | None, cwd: str | Path | None) -> dict[str, Any] | None:
    """Enrich metadata with file and symbol hashes if targets exist relative to cwd."""
    if not metadata or cwd is None:
        return metadata
    res = dict(metadata)
    try:
        from fibmind.reflex import auto_ground_metadata

        return auto_ground_metadata(res, cwd=cwd)
    except ImportError:
        pass

    try:
        from fibmind.reflex import compute_file_hashes, compute_symbol_hashes

        if "files" in res and isinstance(res["files"], (list, tuple)) and not res.get("file_hashes"):
            fhashes = compute_file_hashes(res["files"], cwd=cwd)
            if fhashes:
                res["file_hashes"] = fhashes
        if "symbols" in res and isinstance(res["symbols"], (list, tuple)) and not res.get("symbol_hashes"):
            shashes = compute_symbol_hashes(res["symbols"], cwd=cwd)
            if shashes:
                res["symbol_hashes"] = shashes
    except Exception:
        pass
    return res


def _targets_conflict(t1: str, t2: str) -> bool:
    """Return True if two resource or file targets conflict under advisory locking."""
    s1 = str(t1).strip().replace("\\", "/").rstrip("/")
    s2 = str(t2).strip().replace("\\", "/").rstrip("/")
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    if s1.endswith("/" + s2) or s2.endswith("/" + s1):
        return True
    p1 = Path(s1)
    p2 = Path(s2)
    if (len(p1.parts) == 1 or len(p2.parts) == 1) and p1.name and p1.name == p2.name:
        return True
    return False


class FibBrain:
    """Cognitive facade over a persisted FibMind store."""

    def __init__(
        self,
        service: MemoryService,
        planner: Planner = DEFAULT_PLANNER,
        *,
        max_tlb_pages: int = 32,
        tlb_ttl_steps: int = 12,
    ) -> None:
        self.memory = service
        self.planner = planner
        self.max_tlb_pages = max_tlb_pages
        self.tlb_ttl_steps = tlb_ttl_steps
        self._episode: deque[ObservedEvent] = deque(maxlen=EPISODE_LIMIT)
        self._session_pages: dict[str, dict[str, int]] = {}
        self._session_step: dict[str, int] = {}
        self._session_last_step: dict[str, dict[str, int]] = {}
        self._session_pinned: dict[str, set[str]] = {}
        self._session_parent: dict[str, str] = {}
        self._kernel_sessions: set[str] = set()
        self._ipc_mailboxes: dict[str, list[dict[str, Any]]] = {}

    def _load_session_step(self, session_id: str) -> int:
        """Load MMU clock step for session from memory cache or persisted store setting."""
        if session_id in self._session_step:
            return self._session_step[session_id]
        step = 0
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting"):
            raw = store.read_setting(f"mmu_step:{session_id}")
            if raw is not None:
                try:
                    step = int(raw)
                except (ValueError, TypeError):
                    step = 0
        self._session_step[session_id] = step
        return step

    def _save_session_step(self, session_id: str, step: int) -> None:
        """Save MMU clock step to in-memory cache and persisted store setting."""
        self._session_step[session_id] = step
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "write_setting"):
            try:
                store.write_setting(f"mmu_step:{session_id}", step)
            except Exception:
                pass

    def _load_session_last_step(self, session_id: str) -> dict[str, int]:
        """Load node last-accessed step map for session from memory cache or persisted store setting."""
        if session_id in self._session_last_step:
            return self._session_last_step[session_id]
        last_steps: dict[str, int] = {}
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting"):
            raw = store.read_setting(f"mmu_last_step:{session_id}")
            if isinstance(raw, dict):
                last_steps = {str(k): int(v) for k, v in raw.items()}
        self._session_last_step[session_id] = last_steps
        return last_steps

    def _save_session_last_step(self, session_id: str, last_steps: dict[str, int]) -> None:
        """Save node last-accessed step map to in-memory cache and persisted store setting."""
        self._session_last_step[session_id] = last_steps
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "write_setting"):
            try:
                store.write_setting(f"mmu_last_step:{session_id}", last_steps)
            except Exception:
                pass

    def _load_session_pages(self, session_id: str) -> dict[str, int]:
        """Load MMU page table for session from memory cache or persisted store setting."""
        if session_id in self._session_pages:
            return self._session_pages[session_id]
        pages: dict[str, int] = {}
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting"):
            raw = store.read_setting(f"mmu_pages:{session_id}")
            if isinstance(raw, dict):
                pages = {str(k): int(v) for k, v in raw.items()}
        self._session_pages[session_id] = pages
        return pages

    def _save_session_pages(self, session_id: str, pages: dict[str, int]) -> None:
        """Save MMU page table to in-memory cache and persisted store setting."""
        self._session_pages[session_id] = pages
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "write_setting"):
            try:
                store.write_setting(f"mmu_pages:{session_id}", pages)
            except Exception:
                pass

    def _load_session_pinned(self, session_id: str) -> set[str]:
        """Load pinned pages for session from memory cache or persisted store setting."""
        if session_id in self._session_pinned:
            return self._session_pinned[session_id]
        pinned: set[str] = set()
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting"):
            raw = store.read_setting(f"mmu_pinned:{session_id}")
            if isinstance(raw, list):
                pinned = {str(x) for x in raw}
        self._session_pinned[session_id] = pinned
        return pinned

    def _save_session_pinned(self, session_id: str, pinned: set[str]) -> None:
        """Save pinned pages to in-memory cache and persisted store setting."""
        self._session_pinned[session_id] = set(pinned)
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "write_setting"):
            try:
                store.write_setting(f"mmu_pinned:{session_id}", sorted(pinned))
            except Exception:
                pass

    def pin_page(self, session_id: str, node_id: str) -> dict[str, Any]:
        """Pin a page to cognitive MMU working set to prevent TTL and capacity eviction."""
        session_id = _require_text(session_id, "session_id")
        node_id = _require_text(node_id, "node_id")
        pinned = set(self._load_session_pinned(session_id))
        pinned.add(node_id)
        self._save_session_pinned(session_id, pinned)
        return {"session_id": session_id, "pinned_pages": sorted(pinned)}

    def unpin_page(self, session_id: str, node_id: str) -> dict[str, Any]:
        """Unpin a page from cognitive MMU working set."""
        session_id = _require_text(session_id, "session_id")
        node_id = _require_text(node_id, "node_id")
        pinned = set(self._load_session_pinned(session_id))
        pinned.discard(node_id)
        self._save_session_pinned(session_id, pinned)
        return {"session_id": session_id, "pinned_pages": sorted(pinned)}

    def page_in(
        self,
        session_id: str,
        node_ids: Iterable[str],
        *,
        pin: bool = False,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Demand page one or more memory nodes into a session's cognitive MMU page table."""
        session_id = _require_text(session_id, "session_id")

        def _run_page_in(memory: FibMind) -> dict[str, Any]:
            if state is None:
                identity = BrainState(session_id=session_id).identity()
            else:
                identity = state.identity()
                if identity.get("session_id") is None:
                    identity["session_id"] = session_id

            requested = [str(nid).strip() for nid in node_ids if str(nid).strip()]
            valid_ids: list[str] = []
            for nid in requested:
                if nid in memory.nodes:
                    node = memory.nodes[nid]
                    check_identity = dict(identity)
                    if check_identity.get("owner") is None and node.owner:
                        check_identity["owner"] = node.owner
                    if memory.is_visible(
                        node,
                        statuses=set(MemoryStatus),
                        include_folded=True,
                        **_identity_kwargs(check_identity),
                    ):
                        valid_ids.append(nid)

            mmu = self._mmu_track(session_id, valid_ids, memory, update=True)
            if pin and valid_ids:
                pinned = set(self._load_session_pinned(session_id))
                pinned.update(valid_ids)
                self._save_session_pinned(session_id, pinned)
                mmu["pinned_pages"] = sorted(pinned)

            paged_in = [{**_node_summary(memory.nodes[nid]), "content": memory.nodes[nid].content} for nid in valid_ids]
            return {
                "session_id": session_id,
                "paged_in": paged_in,
                "mmu": mmu,
            }

        return self.memory.read(_run_page_in)

    def _load_ipc_mailbox(self, session_id: str) -> list[dict[str, Any]]:
        """Load IPC mailbox for session from memory cache or persisted store setting."""
        if session_id in self._ipc_mailboxes:
            return [dict(m) for m in self._ipc_mailboxes[session_id]]
        mailbox: list[dict[str, Any]] = []
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting"):
            raw = store.read_setting(f"ipc_mailbox:{session_id}")
            if isinstance(raw, list):
                mailbox = [dict(m) for m in raw if isinstance(m, dict)]
        self._ipc_mailboxes[session_id] = [dict(m) for m in mailbox]
        return mailbox

    def _save_ipc_mailbox(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Save IPC mailbox to in-memory cache and persisted store setting (max 64 messages)."""
        trimmed = [dict(m) for m in messages[-64:]]
        self._ipc_mailboxes[session_id] = trimmed
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "write_setting"):
            try:
                store.write_setting(f"ipc_mailbox:{session_id}", trimmed)
            except Exception:
                pass

    def send_ipc(
        self,
        sender_session_id: str | None = None,
        recipient_session_id: str | None = None,
        signal: str = "",
        payload: dict[str, Any] | None = None,
        *,
        state: BrainState | None = None,
        sender: str | None = None,
        recipient: str | None = None,
    ) -> dict[str, Any]:
        """Send an asynchronous IPC signal to another agent session's mailbox."""
        effective_sender = sender or sender_session_id or ""
        effective_recipient = recipient or recipient_session_id or ""
        sender_session_id = _require_text(effective_sender, "sender_session_id")
        recipient_session_id = _require_text(effective_recipient, "recipient_session_id")
        signal = _require_text(signal, "signal")

        msg = {
            "id": f"ipc_{uuid4().hex[:12]}",
            "sender": sender_session_id,
            "recipient": recipient_session_id,
            "signal": signal.strip(),
            "payload": dict(payload or {}),
            "read": False,
            "created_at": utc_now().isoformat(),
        }
        mailbox = self._load_ipc_mailbox(recipient_session_id)
        mailbox.append(msg)
        self._save_ipc_mailbox(recipient_session_id, mailbox)
        return msg

    def poll_ipc(
        self,
        session_id: str,
        *,
        unread_only: bool = True,
        mark_read: bool = True,
    ) -> list[dict[str, Any]]:
        """Poll incoming IPC signals for a session, optionally marking unread signals as read."""
        session_id = _require_text(session_id, "session_id")
        mailbox = self._load_ipc_mailbox(session_id)
        selected = [dict(m) for m in mailbox if (not unread_only or not m.get("read"))]
        if mark_read and any(not m.get("read") for m in mailbox):
            for m in mailbox:
                m["read"] = True
            self._save_ipc_mailbox(session_id, mailbox)
        return selected

    def _record_kernel_session(self, session_id: str | None) -> None:
        """Record session_id in the kernel sessions tracking table."""
        if not session_id or not str(session_id).strip():
            return
        sess = str(session_id).strip()
        self._kernel_sessions.add(sess)
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting") and hasattr(store, "write_setting"):
            raw = store.read_setting("kernel_sessions")
            sessions: set[str] = set(self._kernel_sessions)
            if isinstance(raw, list):
                sessions.update(str(s).strip() for s in raw if str(s).strip())
            sessions.add(sess)
            try:
                store.write_setting("kernel_sessions", sorted(sessions))
            except Exception:
                pass

    def _load_kernel_sessions(self) -> set[str]:
        """Load all tracked kernel session IDs."""
        sessions = set(self._kernel_sessions)
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting"):
            raw = store.read_setting("kernel_sessions")
            if isinstance(raw, list):
                sessions.update(str(s).strip() for s in raw if str(s).strip())
        sessions.update(self._session_pages.keys())
        sessions.update(self._session_pinned.keys())
        sessions.update(self._session_parent.keys())
        sessions.update(self._session_parent.values())
        sessions.update(self._ipc_mailboxes.keys())
        return sessions

    def _load_session_parent(self, session_id: str) -> str | None:
        """Load parent session ID for a child session."""
        session_id = str(session_id).strip()
        if session_id in self._session_parent:
            return self._session_parent[session_id]
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting"):
            raw = store.read_setting(f"session_parent:{session_id}")
            if raw and isinstance(raw, str) and raw.strip():
                p = raw.strip()
                self._session_parent[session_id] = p
                return p
        return None

    def _save_session_parent(self, child_session_id: str, parent_session_id: str) -> None:
        """Persist parent session mapping for a child session."""
        child = str(child_session_id).strip()
        parent = str(parent_session_id).strip()
        self._session_parent[child] = parent
        self._record_kernel_session(child)
        self._record_kernel_session(parent)
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "write_setting"):
            try:
                store.write_setting(f"session_parent:{child}", parent)
            except Exception:
                pass

    def _load_kernel_leases(self) -> list[dict[str, Any]]:
        """Load active kernel explicit leases, automatically pruning expired leases."""
        leases: list[dict[str, Any]] = []
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "read_setting"):
            raw = store.read_setting("kernel_leases")
            if isinstance(raw, list):
                leases = [dict(item) for item in raw if isinstance(item, dict)]
        now_iso = utc_now().isoformat()
        active_leases = [lease for lease in leases if str(lease.get("expires_at", "")) > now_iso]
        if len(active_leases) != len(leases) and store is not None and hasattr(store, "write_setting"):
            try:
                store.write_setting("kernel_leases", active_leases)
            except Exception:
                pass
        return active_leases

    def _save_kernel_leases(self, leases: list[dict[str, Any]]) -> None:
        """Save active kernel explicit leases."""
        store = getattr(self.memory, "_store", None)
        if store is not None and hasattr(store, "write_setting"):
            try:
                store.write_setting("kernel_leases", leases)
            except Exception:
                pass

    def acquire_lease(
        self,
        session_id: str,
        targets: list[str] | str,
        ttl_minutes: int = 30,
        reason: str = "",
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Acquire kernel advisory leases on files or resources. Detects conflicting locks held by other sessions."""
        session_id = _require_text(session_id, "session_id")
        if isinstance(targets, str):
            norm_targets = [targets.strip()]
        else:
            norm_targets = [str(t).strip() for t in targets if str(t).strip()]
        if not norm_targets:
            raise ValueError("targets cannot be empty")

        self._record_kernel_session(session_id)
        effective_state = state or BrainState()
        identity = effective_state.identity()
        ws = identity.get("workspace_id")
        proj = identity.get("project_id")

        active_leases = self._load_kernel_leases()
        conflicts: list[dict[str, Any]] = []

        for req_target in norm_targets:
            for lease in active_leases:
                if lease.get("session_id") != session_id:
                    if _targets_conflict(req_target, str(lease.get("target", ""))):
                        conflicts.append(
                            {
                                "requested_target": req_target,
                                "conflicting_lease": lease,
                            }
                        )

        if conflicts:
            return {
                "acquired": False,
                "session_id": session_id,
                "conflicts": conflicts,
                "leases": [lease for lease in active_leases if lease.get("session_id") == session_id],
            }

        now = utc_now()
        expires = now + timedelta(minutes=max(1, ttl_minutes))
        new_leases: list[dict[str, Any]] = []
        for req_target in norm_targets:
            existing = next(
                (
                    lease
                    for lease in active_leases
                    if lease.get("session_id") == session_id and lease.get("target") == req_target
                ),
                None,
            )
            if existing is not None:
                existing["expires_at"] = expires.isoformat()
                existing["reason"] = reason or existing.get("reason", "")
                new_leases.append(existing)
            else:
                lease_record = {
                    "session_id": session_id,
                    "target": req_target,
                    "reason": reason,
                    "acquired_at": now.isoformat(),
                    "expires_at": expires.isoformat(),
                    "workspace_id": ws,
                    "project_id": proj,
                }
                active_leases.append(lease_record)
                new_leases.append(lease_record)

        self._save_kernel_leases(active_leases)
        return {
            "acquired": True,
            "session_id": session_id,
            "leases": new_leases,
            "conflicts": [],
        }

    def release_lease(
        self,
        session_id: str,
        targets: list[str] | str | None = None,
    ) -> dict[str, Any]:
        """Release kernel advisory leases held by a session."""
        session_id = _require_text(session_id, "session_id")
        active_leases = self._load_kernel_leases()

        if targets is None:
            remaining = [lease for lease in active_leases if lease.get("session_id") != session_id]
            released_count = len(active_leases) - len(remaining)
        else:
            if isinstance(targets, str):
                norm_targets = [targets.strip()]
            else:
                norm_targets = [str(t).strip() for t in targets if str(t).strip()]
            remaining = []
            released_count = 0
            for lease in active_leases:
                if lease.get("session_id") == session_id and any(
                    _targets_conflict(t, str(lease.get("target", ""))) for t in norm_targets
                ):
                    released_count += 1
                else:
                    remaining.append(lease)

        self._save_kernel_leases(remaining)
        return {
            "released": released_count,
            "session_id": session_id,
            "remaining_leases_count": len(remaining),
        }

    def process_table(
        self,
        workspace_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Query kernel process table returning active sessions, relationships, MMU pages, and locks."""
        all_sessions = sorted(self._load_kernel_sessions())
        active_leases = self._load_kernel_leases()

        def _scan_goals(mem: FibMind) -> dict[str, BrainGoal]:
            session_goals: dict[str, BrainGoal] = {}
            for node in mem.nodes.values():
                if is_goal_node(node) and node.session_id:
                    try:
                        g = parse_goal_document(node.id, node.content, node.title)
                        session_goals[node.session_id] = g
                    except Exception:
                        pass
            return session_goals

        try:
            session_goals = self.memory.read(_scan_goals)
        except Exception:
            session_goals = {}

        processes: list[dict[str, Any]] = []
        for sess in all_sessions:
            parent_sess = self._load_session_parent(sess)
            pages = self._load_session_pages(sess)
            pinned = self._load_session_pinned(sess)
            goal = session_goals.get(sess)

            status = "active"
            goal_dict = None
            if goal is not None:
                goal_dict = {
                    "goal_id": goal.goal_id,
                    "objective": goal.objective,
                    "status": goal.status.value,
                }
                if goal.status == GoalStatus.BLOCKED:
                    status = "blocked"
                elif goal.status == GoalStatus.COMPLETE:
                    status = "complete"

            sess_leases = [lease_entry for lease_entry in active_leases if lease_entry.get("session_id") == sess]

            if workspace_id:
                if any(
                    lease_entry.get("workspace_id") and lease_entry.get("workspace_id") != workspace_id
                    for lease_entry in sess_leases
                ):
                    continue
            if project_id:
                if any(
                    lease_entry.get("project_id") and lease_entry.get("project_id") != project_id
                    for lease_entry in sess_leases
                ):
                    continue

            pending_messages = self.poll_ipc(sess, unread_only=True, mark_read=False)

            processes.append(
                {
                    "session_id": sess,
                    "parent_session_id": parent_sess,
                    "status": status,
                    "goal": goal_dict,
                    "pages_count": len(pages),
                    "pinned_count": len(pinned),
                    "leases_count": len(sess_leases),
                    "active_leases": sess_leases,
                    "unread_ipc_count": len(pending_messages),
                }
            )

        filtered_leases = active_leases
        if workspace_id:
            filtered_leases = [
                lease_entry
                for lease_entry in filtered_leases
                if not lease_entry.get("workspace_id") or lease_entry.get("workspace_id") == workspace_id
            ]
        if project_id:
            filtered_leases = [
                lease_entry
                for lease_entry in filtered_leases
                if not lease_entry.get("project_id") or lease_entry.get("project_id") == project_id
            ]

        return {
            "processes": processes,
            "leases": filtered_leases,
            "total_processes": len(processes),
            "total_active_leases": len(filtered_leases),
            "workspace_id": workspace_id,
            "project_id": project_id,
        }

    def _mmu_track(
        self,
        session_id: str | None,
        candidate_ids: Iterable[str],
        memory: FibMind,
        *,
        update: bool = True,
    ) -> dict[str, Any]:
        """Track cognitive page faults, hits, dirty pages, and attention TTL/capacity evictions."""
        candidate_list = list(candidate_ids)
        if not session_id:
            return {
                "page_hits": [],
                "page_faults": candidate_list,
                "dirty_pages": [],
                "evicted_pages": [],
                "pinned_pages": [],
                "step": 0,
                "session_id": None,
            }

        self._record_kernel_session(session_id)

        current_step = self._load_session_step(session_id) + 1
        page_table = self._load_session_pages(session_id)
        last_steps = self._load_session_last_step(session_id)
        pinned = set(self._load_session_pinned(session_id))

        # Auto-pin high-confidence core constraints
        for nid in candidate_list:
            if nid in memory.nodes:
                node = memory.nodes[nid]
                if node.category in ("requirement", "correction") and node.confidence >= 0.5:
                    pinned.add(nid)

        dirty_pages: list[dict[str, Any]] = []
        evicted_pages: list[str] = []

        for prev_id, prev_rev in list(page_table.items()):
            node = memory.nodes.get(prev_id)
            reason: str | None = None
            if node is None:
                reason = "forgotten"
            elif node.status != MemoryStatus.ACTIVE:
                reason = f"status_{node.status.value}"
            elif node.revision != prev_rev:
                reason = f"revision_changed:{prev_rev}->{node.revision}"
            elif bool(node.metadata.get("conflicted")):
                reason = "conflicted"
            elif bool(node.metadata.get("drift_detected")):
                reason = f"drift:{node.metadata.get('drift_detected')}"

            if reason is not None:
                dirty_pages.append(
                    {
                        "node_id": prev_id,
                        "title": node.title if node else prev_id,
                        "reason": reason,
                    }
                )

        dirty_ids = {d["node_id"] for d in dirty_pages}
        page_hits: list[str] = []
        page_faults: list[str] = []

        for cid in candidate_list:
            if cid in page_table and cid not in dirty_ids:
                last_step_val = last_steps.get(cid, current_step)
                if (current_step - last_step_val) > self.tlb_ttl_steps and cid not in pinned:
                    if cid not in evicted_pages:
                        evicted_pages.append(cid)
                    page_faults.append(cid)
                    last_steps[cid] = current_step
                else:
                    page_hits.append(cid)
                    last_steps[cid] = current_step
            else:
                page_faults.append(cid)
                last_steps[cid] = current_step

        for did in dirty_ids:
            last_steps[did] = current_step

        if update:
            for cid in candidate_list:
                if cid in memory.nodes:
                    page_table[cid] = memory.nodes[cid].revision
            for d in dirty_pages:
                did = d["node_id"]
                node = memory.nodes.get(did)
                if node is None or node.status != MemoryStatus.ACTIVE:
                    page_table.pop(did, None)
                    last_steps.pop(did, None)
                elif did in candidate_list:
                    page_table[did] = node.revision

            # Capacity Eviction: if len(page_table) > max_tlb_pages, evict oldest accessed (excluding pinned)
            if len(page_table) > self.max_tlb_pages:
                ordered_by_lru = [
                    k for k in sorted(page_table.keys(), key=lambda k: (last_steps.get(k, 0), k)) if k not in pinned
                ]
                num_to_evict = len(page_table) - self.max_tlb_pages
                for k in ordered_by_lru[:num_to_evict]:
                    if k not in evicted_pages:
                        evicted_pages.append(k)
                    page_table.pop(k, None)
                    last_steps.pop(k, None)

            self._save_session_step(session_id, current_step)
            self._save_session_pages(session_id, page_table)
            self._save_session_last_step(session_id, last_steps)
            self._save_session_pinned(session_id, pinned)

        return {
            "page_hits": page_hits,
            "page_faults": page_faults,
            "dirty_pages": dirty_pages,
            "evicted_pages": evicted_pages,
            "pinned_pages": sorted(pinned),
            "step": current_step,
            "session_id": session_id,
        }

    def observe(
        self,
        kind: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Record an episode event.

        With a ``session_id`` or ``task_id`` in ``state`` the event is written
        as a ``scope=session`` node in the ``episode`` category, so it survives
        a process restart and only that session (or another session sharing
        the task, e.g. a different AI tool) can recall it. Without either it
        lives in the process-local buffer only. Either way it is not long-term
        memory: ``review_session`` decides what, if anything, gets promoted.
        """
        kind = _require_text(kind, "kind")
        summary = _require_text(summary, "summary")
        payload = dict(payload or {})
        identity = (state or BrainState()).identity()
        node_id: str | None = None
        if identity["session_id"] is not None or identity["task_id"] is not None:
            node_id = self.memory.run(
                lambda memory: memory.append(
                    category=EPISODE_CATEGORY,
                    title=f"{kind}: {summary[:80]}",
                    content=summary,
                    tags=[EPISODE_TAG],
                    metadata={"kind": kind, "payload": payload, "episode": EPISODE_KIND},
                    scope=MemoryScope.SESSION,
                    owner=identity["owner"],
                    workspace_id=identity["workspace_id"],
                    project_id=identity["project_id"],
                    session_id=identity["session_id"],
                    task_id=identity["task_id"],
                    auto_link=False,
                    compress=False,
                )
            )
        event = ObservedEvent(
            kind=kind,
            summary=summary,
            payload=payload,
            node_id=node_id,
            session_id=identity["session_id"],
        )
        self._episode.append(event)
        return event.to_dict()

    def recall(
        self,
        goal: str,
        state: BrainState | None = None,
        top_k: int = 5,
        depth: int = 1,
        max_chars: int = 2000,
        budget_tokens: int | None = None,
        include_hot: bool = True,
        include_gaps: bool = False,
        delta_only: bool = False,
    ) -> dict[str, Any]:
        """Load a context pack for ``goal`` inside ``state`` and a token budget.

        The pack renders the session's frozen hot snapshot first, then direct
        hits, then graph expansion, and reports ``budget`` usage per section.
        ``budget_tokens`` defaults to 500 (roughly ``max_chars / 4``).
        """
        identity = (state or BrainState()).identity()
        budget = budget_tokens if budget_tokens is not None else DEFAULT_BUDGET_TOKENS
        hot_ids = self.hot_snapshot(state)["node_ids"] if include_hot else []
        gaps_summary: str | None = None
        if include_gaps:
            assessment = self.assess(goal, state=state, top_k=top_k)
            verdict = assessment.get("verdict", "unknown")
            unknown_items = assessment.get("unknown", [])
            if unknown_items:
                top_unknown = ", ".join(f"{uk['aspect']} ({uk['route']})" for uk in unknown_items[:3])
                gaps_summary = f"Gaps: verdict={verdict}; unknown: {top_unknown}"
            else:
                gaps_summary = f"Gaps: verdict={verdict}"
        pack = self.memory.context(
            goal,
            top_k=top_k,
            depth=depth,
            max_chars=max_chars,
            exclude_categories=[EPISODE_CATEGORY, HOT_CATEGORY],
            budget_tokens=budget,
            hot_node_ids=hot_ids,
            hot_budget_tokens=min(HOT_BUDGET_TOKENS, budget // 3) if hot_ids else 0,
            gaps=gaps_summary,
            **_identity_kwargs(identity),
        )
        pack["state"] = identity
        pack["episode"] = [event.to_dict() for event in self._episode_for(identity)]

        hit_ids = [h["node_id"] for h in pack.get("hits", [])]

        def _track_and_swap(mem: FibMind) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            mmu_res = self._mmu_track(identity["session_id"], hit_ids, mem, update=True)
            raw_budget = pack.get("budget")
            raw_truncated: list[Any] = []
            if isinstance(raw_budget, dict):
                raw_truncated = raw_budget.get("truncated", [])
            elif hasattr(raw_budget, "truncated"):
                raw_truncated = getattr(raw_budget, "truncated", [])
            swap: list[dict[str, Any]] = []
            seen_swap: set[str] = set()
            for item in raw_truncated:
                nid = item.get("node_id") if isinstance(item, dict) else (item if isinstance(item, str) else None)
                if nid and nid in mem.nodes and nid not in seen_swap:
                    seen_swap.add(nid)
                    node = mem.nodes[nid]
                    swap.append(
                        {
                            "node_id": nid,
                            "title": node.title,
                            "category": node.category,
                            "confidence": round(node.confidence, 4),
                            "uri": f"fibmind://node/{nid}",
                        }
                    )
            return mmu_res, swap

        mmu, swap_index = self.memory.read(_track_and_swap)
        if delta_only and mmu.get("page_hits"):
            hit_set = set(mmu["page_hits"])
            pack["hits"] = [h for h in pack.get("hits", []) if h["node_id"] not in hit_set]
        pack["mmu"] = {**mmu, "delta_only": delta_only, "swap_index": swap_index}
        return pack

    def assess(
        self,
        goal: str,
        state: BrainState | None = None,
        top_k: int = 8,
        workspace_root: str | Path | None = None,
        assessor: GapAssessor | None = None,
        observe_event: bool = True,
    ) -> dict[str, Any]:
        """Assess what is known, uncertain, contradicted, or unknown about a goal."""
        from fibmind.assessment import classify

        goal = _require_text(goal, "goal")
        effective_state = state or BrainState()
        identity = effective_state.identity()
        cwd = workspace_root or getattr(effective_state, "workspace_path", None)

        def run(memory: FibMind) -> dict[str, Any]:
            excluded = {EPISODE_CATEGORY, HOT_CATEGORY, GOAL_CATEGORY}
            active_hits = memory.search(
                goal,
                top_k=top_k,
                exclude_categories=excluded,
                statuses={MemoryStatus.ACTIVE},
                **_identity_kwargs(identity),
            )
            candidate_nodes = [hit.node for hit in active_hits]

            refuted_hits = memory.search(
                goal,
                top_k=top_k,
                exclude_categories=excluded,
                statuses={MemoryStatus.REFUTED},
                **_identity_kwargs(identity),
            )
            refuted_nodes = [hit.node for hit in refuted_hits]

            open_questions = [
                node
                for node in memory.nodes.values()
                if node.category == "question"
                and node.status in (MemoryStatus.ACTIVE, MemoryStatus.PENDING)
                and memory.is_visible(node, **_identity_kwargs(identity))
            ]

            embeddings = getattr(self.memory, "_embeddings", None)
            provider = embeddings.provider if embeddings and embeddings.enabled else None

            kwargs: dict[str, Any] = {
                "refuted_candidates": refuted_nodes,
                "lexical": memory.lexical,
                "outcome_lookup": memory.outcome_counts,
                "last_confirmation_lookup": memory.last_confirmation,
                "cwd": cwd,
                "model_id": effective_state.model_id,
                "open_questions": open_questions,
                "vector_provider": provider,
                "assessor": assessor,
                "state": identity,
            }
            classify_sig = inspect.signature(classify)
            if "p_refuted_given_known" in classify_sig.parameters:
                p_refuted = 0.0
                try:
                    p_refuted = float(self.memory.status().get("metacognition", {}).get("p_refuted_given_known", 0.0))
                except Exception:
                    p_refuted = 0.0
                kwargs["p_refuted_given_known"] = p_refuted

            return classify(
                goal,
                candidate_nodes,
                **kwargs,
            )

        result = self.memory.read(run)

        if observe_event and identity["session_id"] is not None:
            self.observe(
                "assess",
                f"Assessed goal: {goal[:60]}",
                payload={
                    "verdict": result["verdict"],
                    "known_ids": [k["node_id"] for k in result["known"]],
                    "uncertain_ids": [u["node_id"] for u in result["uncertain"]],
                    "contradicted_ids": [c["node_id"] for c in result["contradicted"]],
                    "unknown_aspects": [uk["aspect"] for uk in result["unknown"]],
                },
                state=effective_state,
            )

        return result

    def entity_profile(
        self,
        target: str,
        state: BrainState | None = None,
        workspace_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Aggregate an entity world model view for a file path or symbol."""
        from fibmind.entity import build_entity_profile

        target = _require_text(target, "target")
        effective_state = state or BrainState()
        cwd = workspace_root or getattr(effective_state, "workspace_path", None)
        identity = effective_state.identity()

        return self.memory.read(
            lambda mem: build_entity_profile(
                mem,
                target,
                identity=identity,
                cwd=cwd,
            )
        )

    def consolidate(
        self,
        state: BrainState | None = None,
        workspace_root: str | Path | None = None,
        auto_mark_drift: bool = True,
        auto_promote_clusters: bool = False,
    ) -> dict[str, Any]:
        """Active environmental drift audit and memory consolidation."""
        from fibmind.admission import propose_conflict_synthesis
        from fibmind.models import EventOp, MemoryStatus, RelationType, utc_now
        from fibmind.reflex import check_node_preconditions

        effective_state = state or BrainState()
        identity = effective_state.identity()
        cwd = workspace_root or getattr(effective_state, "workspace_path", None)

        def _run_consolidate(mem: Any) -> dict[str, Any]:
            scanned = 0
            drifted: list[dict[str, Any]] = []
            conflicts: list[dict[str, Any]] = []
            contradicted_pairs: set[tuple[str, str]] = set()

            for node in mem.nodes.values():
                if node.status != MemoryStatus.ACTIVE or node.folded_into is not None:
                    continue
                if not mem.is_visible(
                    node,
                    owner=identity.get("owner"),
                    statuses={MemoryStatus.ACTIVE},
                    workspace_id=identity.get("workspace_id"),
                    project_id=identity.get("project_id"),
                    session_id=identity.get("session_id"),
                    task_id=identity.get("task_id"),
                ):
                    continue

                scanned += 1
                meta = node.metadata or {}

                # 1. Conflict collection
                if meta.get("conflicted") or bool(meta.get("potential_conflicts")):
                    conflicts.append(
                        {
                            "node_id": node.id,
                            "title": node.title,
                            "conflicted": bool(meta.get("conflicted")),
                            "conflict_reason": meta.get("conflict_reason"),
                            "potential_conflicts": list(meta.get("potential_conflicts") or ()),
                        }
                    )
                    for other_id in meta.get("potential_conflicts") or ():
                        if other_id in mem.nodes:
                            other_node = mem.nodes[other_id]
                            if other_node.status == MemoryStatus.ACTIVE and other_node.folded_into is None:
                                if mem.is_visible(
                                    other_node,
                                    owner=identity.get("owner"),
                                    statuses={MemoryStatus.ACTIVE},
                                    workspace_id=identity.get("workspace_id"),
                                    project_id=identity.get("project_id"),
                                    session_id=identity.get("session_id"),
                                    task_id=identity.get("task_id"),
                                ):
                                    pair = (min(node.id, other_id), max(node.id, other_id))
                                    contradicted_pairs.add(pair)

                # Check CONTRADICTS graph edges
                for edge in mem.edges.values():
                    if edge.relation_type == RelationType.CONTRADICTS:
                        if edge.from_node_id in mem.nodes and edge.to_node_id in mem.nodes:
                            n_from = mem.nodes[edge.from_node_id]
                            n_to = mem.nodes[edge.to_node_id]
                            if (
                                n_from.status == MemoryStatus.ACTIVE
                                and n_from.folded_into is None
                                and n_to.status == MemoryStatus.ACTIVE
                                and n_to.folded_into is None
                            ):
                                if mem.is_visible(
                                    n_from,
                                    owner=identity.get("owner"),
                                    statuses={MemoryStatus.ACTIVE},
                                    workspace_id=identity.get("workspace_id"),
                                    project_id=identity.get("project_id"),
                                    session_id=identity.get("session_id"),
                                    task_id=identity.get("task_id"),
                                ) and mem.is_visible(
                                    n_to,
                                    owner=identity.get("owner"),
                                    statuses={MemoryStatus.ACTIVE},
                                    workspace_id=identity.get("workspace_id"),
                                    project_id=identity.get("project_id"),
                                    session_id=identity.get("session_id"),
                                    task_id=identity.get("task_id"),
                                ):
                                    pair = (min(n_from.id, n_to.id), max(n_from.id, n_to.id))
                                    contradicted_pairs.add(pair)

                # 2. File / Symbol drift check
                has_file_precond = (
                    bool(meta.get("file_hashes"))
                    or bool(meta.get("symbol_hashes"))
                    or bool(meta.get("require_files_exist"))
                )
                if has_file_precond:
                    ok, reason = check_node_preconditions(node, cwd=cwd)
                    if not ok and reason.startswith(
                        (
                            "missing_hashed_file",
                            "file_hash_mismatch",
                            "missing_target_file",
                            "missing_symbol",
                            "symbol_hash_mismatch",
                        )
                    ):
                        drifted.append(
                            {
                                "node_id": node.id,
                                "title": node.title,
                                "reason": reason,
                                "previous_confidence": round(node.confidence, 4),
                            }
                        )
                        if auto_mark_drift:
                            meta["drift_detected"] = reason
                            node.metadata = meta
                            if node.confidence >= 0.5:
                                node.confidence = 0.25
                                node.confidence_source = f"drift_demoted: {reason}"
                            node.updated_at = utc_now()
                            mem._record(
                                EventOp.OBSERVE,
                                {
                                    "node_id": node.id,
                                    "verdict": "inconclusive",
                                    "source": f"consolidate_drift: {reason}",
                                    "note": reason,
                                    "confidence": node.confidence,
                                    "status": node.status.value,
                                    "status_reason": node.status_reason,
                                    "updated_at": node.updated_at.isoformat(),
                                },
                            )

            synthesis_proposals: list[dict[str, Any]] = []
            for id_a, id_b in sorted(contradicted_pairs):
                node_a = mem.nodes.get(id_a)
                node_b = mem.nodes.get(id_b)
                if node_a and node_b:
                    try:
                        proposal = propose_conflict_synthesis(mem, node_a, node_b)
                    except Exception:
                        proposal = {
                            "node_id_a": node_a.id,
                            "node_id_b": node_b.id,
                            "title_a": node_a.title,
                            "title_b": node_b.title,
                            "conditions_a": [f"context:{node_a.title}"],
                            "conditions_b": [f"context:{node_b.title}"],
                            "suggested_title": f"Synthesized: {node_a.title} vs {node_b.title}",
                            "suggested_content": (
                                f"When context:{node_a.title}: {node_a.title}. "
                                f"When context:{node_b.title}: {node_b.title}."
                            ),
                        }
                    synthesis_proposals.append(proposal)

            return {
                "scanned": scanned,
                "drifted": drifted,
                "conflicts": conflicts,
                "synthesis_proposals": synthesis_proposals,
            }

        if auto_mark_drift:
            result = self.memory.run(_run_consolidate)
        else:
            result = self.memory.read(_run_consolidate)

        # 3. Revalidation due count
        try:
            reval = self.revalidation_candidates(state=effective_state)
            reval_count = len(reval)
        except Exception:
            reval_count = 0

        # 4. L0 hot snapshot refresh
        hot_node_ids: list[str] = []
        if effective_state.session_id:
            try:
                hot_pack = self.hot_snapshot(state=effective_state, refresh=True)
                hot_node_ids = [str(nid) for nid in hot_pack.get("node_ids", [])]
            except Exception:
                pass

        clusters: list[dict[str, Any]] = []
        promoted_knowledge: list[dict[str, Any]] = []
        if hasattr(self.memory, "discover_knowledge_clusters"):
            try:
                clusters_res = self.memory.discover_knowledge_clusters(
                    identity=identity,
                    auto_promote=auto_promote_clusters,
                )
                clusters = clusters_res.get("clusters", [])
                promoted_knowledge = clusters_res.get("promoted", [])
            except Exception:
                clusters = []
                promoted_knowledge = []

        return {
            "scanned": result["scanned"],
            "drifted": result["drifted"],
            "conflicts": result["conflicts"],
            "synthesis_proposals": result.get("synthesis_proposals", []),
            "revalidation_due": reval_count,
            "hot_node_ids": hot_node_ids,
            "knowledge_clusters": clusters,
            "promoted_knowledge": promoted_knowledge,
            "state": identity,
        }

    def sync_skills(
        self,
        target_dir: str | Path,
        state: BrainState | None = None,
        min_maturity: str = "verified",
        workspace_root: str | Path | None = None,
        host: str = "generic",
        write_manifest: bool = True,
        ingest_modified: bool = False,
    ) -> dict[str, Any]:
        """Materialize mature procedures as SKILL.md and auto-revoke refuted/drifted ones."""
        from fibmind.procedure import sync_skills

        effective_state = state or BrainState()
        identity = effective_state.identity()
        cwd = workspace_root or getattr(effective_state, "workspace_path", None)

        def _run_sync(mem: Any) -> dict[str, Any]:
            sig = inspect.signature(sync_skills)
            kwargs: dict[str, Any] = {
                "identity": identity,
                "min_maturity": min_maturity,
                "cwd": cwd,
            }
            if "host" in sig.parameters:
                kwargs["host"] = host
            if "write_manifest" in sig.parameters:
                kwargs["write_manifest"] = write_manifest
            if "ingest_modified" in sig.parameters:
                kwargs["ingest_modified"] = ingest_modified
            return sync_skills(mem, target_dir, **kwargs)

        if ingest_modified:
            return self.memory.run(_run_sync)
        return self.memory.read(_run_sync)

    def context(
        self,
        goal: str = "",
        state: BrainState | None = None,
        *,
        top_k: int = 5,
        profile_top_k: int = 3,
        budget_tokens: int = 500,
        min_relevance: float = 0.12,
        model_id: str | None = None,
        delta_only: bool = False,
        cwd: str | Path | None = None,
    ) -> dict[str, Any]:
        """Read user background for any harness; no plan, episode, or snapshot writes."""
        from fibmind.background import build_background

        effective = state or BrainState()
        session_id = effective.session_id
        effective_cwd = cwd or getattr(effective, "workspace_path", None)

        def _run_context(memory: FibMind) -> dict[str, Any]:
            already_paged: set[str] = set()
            if delta_only and session_id:
                next_step = self._load_session_step(session_id) + 1
                last_steps = self._load_session_last_step(session_id)
                pinned = self._load_session_pinned(session_id)
                already_paged = {
                    nid
                    for nid, rev in self._load_session_pages(session_id).items()
                    if nid in memory.nodes
                    and memory.nodes[nid].revision == rev
                    and memory.nodes[nid].status == MemoryStatus.ACTIVE
                    and not memory.nodes[nid].metadata.get("conflicted")
                    and not memory.nodes[nid].metadata.get("drift_detected")
                    and (nid in pinned or (next_step - last_steps.get(nid, next_step)) <= self.tlb_ttl_steps)
                }

            sig = inspect.signature(build_background)
            bg_kwargs: dict[str, Any] = {
                "top_k": top_k,
                "profile_top_k": profile_top_k,
                "budget_tokens": budget_tokens,
                "min_relevance": min_relevance,
                "model_id": model_id or effective.model_id,
            }
            if "cwd" in sig.parameters:
                bg_kwargs["cwd"] = effective_cwd
            passed_exclude = False
            if "exclude_node_ids" in sig.parameters:
                bg_kwargs["exclude_node_ids"] = already_paged if (delta_only and session_id) else None
                passed_exclude = True

            result = build_background(
                memory,
                goal,
                **effective.identity(),
                **bg_kwargs,
            )

            candidate_ids: list[str]
            if delta_only and session_id and already_paged:
                if passed_exclude:
                    rendered_ids = [m["node_id"] for m in result.get("memories", [])]
                    candidate_ids = list(already_paged) + rendered_ids
                else:
                    original_memories = result.get("memories", [])
                    candidate_ids = [m["node_id"] for m in original_memories]
                    kept_memories = [m for m in original_memories if m["node_id"] not in already_paged]
                    excluded_titles = {memory.nodes[nid].title for nid in already_paged if nid in memory.nodes}
                    lines = result.get("text", "").splitlines()
                    filtered_lines = [
                        line
                        for line in lines
                        if not any(
                            (
                                f"] {title}:" in line
                                or f"- [{title}]" in line
                                or (line.startswith("- [") and f"] {title}:" in line)
                            )
                            for title in excluded_titles
                        )
                    ]
                    result["text"] = "\n".join(filtered_lines)
                    result["memories"] = kept_memories
                    result["has_task_context"] = any(m.get("section") == "task" for m in kept_memories)
            else:
                candidate_ids = [m["node_id"] for m in result.get("memories", [])]

            mmu = self._mmu_track(session_id, candidate_ids, memory, update=True)

            if mmu.get("dirty_pages"):
                warnings = result.setdefault("warnings", [])
                for dp in mmu["dirty_pages"]:
                    warnings.append(
                        {
                            "node_id": dp["node_id"],
                            "title": dp.get("title", dp["node_id"]),
                            "kind": "dirty_page",
                            "reason": dp.get("reason", "dirty_page"),
                        }
                    )

            raw_truncated = result.get("budget", {}).get("truncated", [])
            swap_index: list[dict[str, Any]] = []
            seen_swap: set[str] = set()
            for item in raw_truncated:
                nid = item.get("node_id") if isinstance(item, dict) else (item if isinstance(item, str) else None)
                if nid and nid in memory.nodes and nid not in seen_swap:
                    seen_swap.add(nid)
                    node = memory.nodes[nid]
                    swap_index.append(
                        {
                            "node_id": nid,
                            "title": node.title,
                            "category": node.category,
                            "confidence": round(node.confidence, 4),
                            "uri": f"fibmind://node/{nid}",
                        }
                    )

            result["mmu"] = {**mmu, "delta_only": delta_only, "swap_index": swap_index}
            return result

        return self.memory.read(_run_context)

    def reflex(
        self,
        goal: str,
        *,
        state: BrainState | None = None,
        execute: bool = False,
        inputs: dict[str, str] | None = None,
        min_confidence: float = 0.75,
        min_maturity: str = "verified",
        min_relevance: float = 0.25,
        cwd: str | Path | None = None,
        model_id: str | None = None,
        require_files_exist: bool = False,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """System 1 fast reflex: answer from high-confidence verified memory or run a verified procedure without LLM."""
        from fibmind.reflex import run_reflex

        return run_reflex(
            self,
            goal,
            state=state,
            execute=execute,
            inputs=inputs,
            min_confidence=min_confidence,
            min_maturity=min_maturity,
            min_relevance=min_relevance,
            cwd=cwd,
            model_id=model_id,
            require_files_exist=require_files_exist,
            timeout=timeout,
        )

    def kernel_step(
        self,
        goal: str,
        state: BrainState | None = None,
        *,
        workspace_root: str | Path | None = None,
        execute_reflex: bool = False,
        delta_only: bool = False,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    ) -> dict[str, Any]:
        """Execute a unified cognitive cycle (assess -> reflex -> interrupt/route -> mmu context)."""
        goal = _require_text(goal, "goal")
        effective_state = state or BrainState()
        cwd = workspace_root or getattr(effective_state, "workspace_path", None)

        # Step 1: Run assessment
        ipc_messages = (
            self.poll_ipc(effective_state.session_id, unread_only=True, mark_read=True)
            if effective_state.session_id
            else []
        )
        assessment = self.assess(goal, state=effective_state, workspace_root=cwd, observe_event=False)

        # Step 2: Check System 1 Fast Reflex if known
        reflex_res: dict[str, Any] | None = None
        if assessment.get("verdict") == "known" and not ipc_messages:
            reflex_res = self.reflex(goal, state=effective_state, execute=execute_reflex, cwd=cwd)
            if reflex_res.get("bypass_llm") is True:
                return {
                    "goal": goal,
                    "dispatch_mode": "system1_reflex",
                    "bypass_llm": True,
                    "assessment": assessment,
                    "reflex": reflex_res,
                    "context": None,
                    "mmu": None,
                    "entities": [],
                    "interrupts": [],
                    "ipc_messages": [],
                    "next_actions": [{"route": "reflex", "target": reflex_res.get("node_id") or "procedure"}],
                    "state": effective_state.identity(),
                }

        # Step 3: Determine Kernel Interrupt / Dispatch Mode
        has_conflict = bool(assessment.get("contradicted")) or any(
            "conflicted" in u.get("reasons", []) for u in assessment.get("uncertain", [])
        )
        if has_conflict:
            dispatch_mode = "interrupt_conflict"
        elif any(u.get("route") == "verify" for u in assessment.get("uncertain", [])):
            dispatch_mode = "interrupt_verify"
        elif ipc_messages:
            dispatch_mode = "interrupt_ipc"
        else:
            dispatch_mode = "system2_reason"

        # Step 4: Load MMU-paged Context & World Model
        ctx = self.context(
            goal,
            state=effective_state,
            budget_tokens=budget_tokens,
            delta_only=delta_only,
            cwd=cwd,
        )

        next_actions = assessment.get("next", [])
        interrupts: list[dict[str, Any]] = []

        # 1. Contradicted memories
        for c in assessment.get("contradicted", []):
            interrupts.append(
                {
                    "kind": "conflict",
                    "target": c.get("node_id"),
                    "reason": c.get("reason", "refuted"),
                }
            )

        # 2. Conflicted or upstream refuted uncertain memories
        for u in assessment.get("uncertain", []):
            reasons = u.get("reasons", [])
            if "conflicted" in reasons or "upstream_refuted" in reasons:
                interrupts.append(
                    {
                        "kind": "conflict",
                        "target": u.get("node_id"),
                        "reason": ",".join(reasons),
                    }
                )

        # 3. Warnings from context (drift / active_lease)
        for w in ctx.get("warnings", []):
            if w.get("kind") == "drift":
                interrupts.append(
                    {
                        "kind": "drift",
                        "target": w.get("entity") or w.get("node_id"),
                        "reason": w.get("reason", "drift"),
                    }
                )
            elif w.get("kind") == "active_lease":
                interrupts.append(
                    {
                        "kind": "lease_conflict",
                        "target": w.get("entity"),
                        "reason": w.get("reason", "active_lease"),
                    }
                )

        # 4. Verify actions
        for act in next_actions:
            if act.get("route") == "verify":
                interrupts.append(
                    {
                        "kind": "verify",
                        "target": act.get("target"),
                        "reason": "unverified memory with executable check",
                    }
                )

        # 5. IPC signals
        for msg in ipc_messages:
            interrupts.append(
                {
                    "kind": "ipc",
                    "target": msg["sender"],
                    "reason": f"{msg['signal']}",
                }
            )

        return {
            "goal": goal,
            "dispatch_mode": dispatch_mode,
            "bypass_llm": False,
            "assessment": assessment,
            "reflex": reflex_res if assessment.get("verdict") == "known" else None,
            "context": ctx,
            "mmu": ctx.get("mmu"),
            "entities": ctx.get("entities", []),
            "interrupts": interrupts,
            "ipc_messages": ipc_messages,
            "next_actions": next_actions,
            "state": ctx.get("state"),
        }

    # ------------------------------------------------------------------
    # L0 hot memory

    def hot_snapshot(self, state: BrainState | None = None, refresh: bool = False) -> dict[str, Any]:
        """The frozen hot-memory snapshot for this session.

        Selection is deterministic: active memories of a hot kind visible to
        this identity, ordered by confidence then recency, capped by count and
        by ``HOT_BUDGET_TOKENS``. The result is persisted as a session-scoped
        node so every recall in the session sees the same preamble; ``refresh``
        (used by ``review_session``) recomputes it for the *next* session and
        leaves the current one frozen.
        """
        identity = (state or BrainState()).identity()
        model_id = state.model_id if state else None
        session_id = identity["session_id"]

        def find_existing(memory: FibMind) -> MemoryNode | None:
            if session_id is None:
                return None
            return next(
                (
                    node
                    for node in memory.nodes.values()
                    if node.category == HOT_CATEGORY
                    and HOT_TAG in node.tags
                    and node.session_id == session_id
                    and node.status == MemoryStatus.ACTIVE
                    and memory.is_visible(node, **_identity_kwargs(identity))
                ),
                None,
            )

        def read_existing(memory: FibMind) -> dict[str, Any] | None:
            existing = find_existing(memory)
            if existing is None:
                return None
            data = json.loads(existing.content)
            live = [nid for nid in data.get("node_ids", []) if nid in memory.nodes]
            return {"node_ids": live, "frozen": True, "snapshot_id": existing.id, "session_id": session_id}

        def run(memory: FibMind) -> dict[str, Any]:
            existing = find_existing(memory)
            if existing is not None and not refresh:
                cached = read_existing(memory)
                assert cached is not None
                return cached

            selected = _select_hot(memory, identity, model_id=model_id)
            node_ids = [node.id for node in selected]
            snapshot_id = None
            if session_id is not None:
                if existing is not None:
                    memory.set_status(existing.id, MemoryStatus.STALE, "hot snapshot refreshed")
                snapshot_id = memory.append(
                    category=HOT_CATEGORY,
                    title=f"hot snapshot for {session_id}",
                    content=json.dumps({"kind": HOT_KIND, "node_ids": node_ids}, ensure_ascii=False),
                    tags=[HOT_TAG],
                    metadata={"kind": HOT_KIND, "count": len(node_ids)},
                    scope=MemoryScope.SESSION,
                    owner=identity["owner"],
                    workspace_id=identity["workspace_id"],
                    project_id=identity["project_id"],
                    session_id=session_id,
                    task_id=identity["task_id"],
                    auto_link=False,
                    compress=False,
                )
            return {
                "node_ids": node_ids,
                "frozen": session_id is not None,
                "snapshot_id": snapshot_id,
                "session_id": session_id,
            }

        if session_id is None:
            return self.memory.read(run)
        if not refresh:
            cached = self.memory.read(read_existing)
            if cached is not None:
                return cached
        return self.memory.run(run)

    def _episode_for(self, identity: dict[str, str | None]) -> list[ObservedEvent]:
        """The persisted episode of this session or task, else the in-process buffer."""
        session_id = identity["session_id"]
        task_id = identity["task_id"]
        if session_id is None and task_id is None:
            return list(self._episode)
        nodes = self.memory.read(lambda memory: _episode_nodes(memory, identity))
        return [ObservedEvent.from_node(node) for node in nodes][-EPISODE_LIMIT:]

    def admit(
        self,
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        scope: str = MemoryScope.PERSONAL.value,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Decide whether a candidate should become a long-term memory."""
        candidate = _candidate(category, title, content, tags, scope, state)
        return self.memory.admit(candidate)

    def remember(
        self,
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        scope: str = MemoryScope.PERSONAL.value,
        state: BrainState | None = None,
        context_level: str | None = None,
        cwd: str | Path | None = None,
    ) -> dict[str, Any]:
        """Admit, then write. Skipped candidates are not persisted."""
        effective_cwd = cwd or getattr(state, "workspace_path", None)
        if effective_cwd is not None and metadata:
            metadata = _auto_ground(metadata, effective_cwd)
        if context_level is not None:
            level = ContextLevel(context_level)
            identity = (state or BrainState()).identity()
            if scope != MemoryScope.PERSONAL.value or not identity["owner"]:
                raise ValueError("context_level requires personal scope and a human owner")
            origin = {key: identity[key] for key in ("workspace_id", "project_id")}
            if level == ContextLevel.USER:
                identity.update(workspace_id=None, project_id=None)
            elif level == ContextLevel.WORKSPACE:
                if not identity["workspace_id"]:
                    raise ValueError("workspace context requires workspace_id")
                identity["project_id"] = None
            elif not identity["project_id"]:
                raise ValueError("project context requires project_id")
            state = BrainState(**identity)
            metadata = {**(metadata or {}), "context_level": level.value, "source_context": origin}
        candidate = _candidate(category, title, content, tags, scope, state)
        res = self.memory.remember(candidate, metadata=metadata)
        if res.get("verdict") == "write" or res.get("node_id"):
            self._try_advance_active_goal("remember", state, target=title)
        return res

    def advise(
        self,
        action: str,
        kind: str = "tool",
        state: BrainState | None = None,
        top_k: int = 5,
        arguments: dict[str, Any] | None = None,
        planned_files: list[str] | tuple[str, ...] | None = None,
        enforce_leases: bool = False,
    ) -> dict[str, Any]:
        """Should this action run, given what we already know?

        A refuted memory blocks the action when it matches the action name,
        shares argument terms, or matches refuted attempts targeting planned_files.
        """
        _require_text(action, "action")
        identity = (state or BrainState()).identity()
        action_terms = set(tokenize(action))
        argument_terms = set(tokenize(_flatten_arguments(arguments))) - action_terms if arguments else set()
        file_terms: set[str] = set()
        if planned_files:
            for pf in planned_files:
                for part in Path(pf).parts:
                    file_terms.update(tokenize(part))
        query = " ".join([kind, action, *sorted(argument_terms), *sorted(file_terms)])
        search_kwargs = _identity_kwargs(identity)
        active = self.memory.search(query, top_k=top_k, **search_kwargs)
        refuted = self.memory.search(
            query,
            top_k=top_k * 2,
            statuses=[MemoryStatus.REFUTED.value],
            **search_kwargs,
        )
        notes = list(active["results"])
        blockers = []
        for hit in refuted["results"]:
            matched = set(hit.get("matched_terms") or [])
            hit_files: set[str] = set()
            meta = hit.get("metadata") or {}
            for f in meta.get("files", []) or []:
                for part in Path(f).parts:
                    hit_files.update(tokenize(part))

            file_match = bool(file_terms and (file_terms & matched or file_terms & hit_files))
            action_match = bool(matched & action_terms)

            if not (action_match or file_match):
                continue
            if argument_terms and not (matched & argument_terms or file_match):
                continue
            blockers.append(hit)
        if blockers:
            decision = AdviseDecision(
                AdviseVerdict.REJECT,
                f"refuted evidence advises against {action}"
                + (f" targeting {', '.join(sorted(planned_files))}" if planned_files else "")
                + (f" with {', '.join(sorted(argument_terms))}" if argument_terms else ""),
                memories=tuple(blockers + notes),
            )
        else:
            reason = (
                f"no refuted evidence against {action}"
                if not notes
                else f"recalled {len(notes)} related memories; no blocking refutation"
            )
            decision = AdviseDecision(AdviseVerdict.ALLOW, reason, memories=tuple(notes))

        decision_payload = decision.to_dict()

        active_leases: list[dict[str, Any]] = []
        if planned_files:
            from fibmind.entity import build_entity_profile

            cwd = getattr(state, "workspace_path", None)
            seen_lease_ids: set[str] = set()
            for pf in planned_files:
                try:
                    target_file = str(pf)
                    prof = self.memory.read(
                        lambda mem, target=target_file: build_entity_profile(mem, target, identity=identity, cwd=cwd)
                    )
                    for lease in prof.get("active_leases", []):
                        lid = str(lease.get("node_id") or lease.get("id") or lease)
                        if lid not in seen_lease_ids:
                            seen_lease_ids.add(lid)
                            active_leases.append(lease)
                except Exception:
                    pass

            explicit_leases = self._load_kernel_leases()
            sess_id = identity.get("session_id")
            for ex_lease in explicit_leases:
                if sess_id and ex_lease.get("session_id") == sess_id:
                    continue
                ex_target = str(ex_lease.get("target", "")).strip()
                for pf in planned_files:
                    if _targets_conflict(str(pf), ex_target):
                        lid = f"kernel_lease:{ex_lease.get('session_id')}:{ex_target}"
                        if lid not in seen_lease_ids:
                            seen_lease_ids.add(lid)
                            active_leases.append(ex_lease)

        decision_payload["active_leases"] = active_leases

        if enforce_leases and active_leases and decision_payload["verdict"] != AdviseVerdict.REJECT.value:
            holding_sessions = sorted(
                {str(lease.get("session_id")) for lease in active_leases if lease.get("session_id")}
            )
            if not holding_sessions:
                holding_sessions = ["unknown"]
            conflict_files = ", ".join(sorted(str(pf) for pf in planned_files or []))
            sessions_str = ", ".join(holding_sessions)
            decision_payload["verdict"] = AdviseVerdict.REJECT.value
            decision_payload["reason"] = f"active lease conflict on {conflict_files} held by session(s) {sessions_str}"

        return decision_payload

    def plan(
        self,
        objective: str | None = None,
        goal_id: str | None = None,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Create or refresh a persisted goal and its deterministic plan."""
        identity = (state or BrainState()).identity()
        existing = self._resolve_goal(objective, goal_id, state, allow_missing=True)
        resolved_objective = _require_text(
            (existing.objective if existing else None) or objective or "",
            "objective",
        )
        pack = self.recall(resolved_objective, state=state)
        assessment = self.assess(resolved_objective, state=state, observe_event=False)
        evidence = list(pack.get("hits") or []) + self._refuted_hits(resolved_objective, identity)
        if existing is None:
            written = self.memory.append(
                GOAL_CATEGORY,
                _goal_title(resolved_objective),
                encode_goal(self.planner.plan(resolved_objective, memories=evidence, assessment=assessment)),
                tags=[GOAL_TAG],
                metadata={"kind": "fibbrain_goal"},
                scope=MemoryScope.PERSONAL.value,
                **identity,
            )
            goal_id = written["node_id"]
        else:
            goal_id = existing.goal_id
        goal = self.planner.plan(
            resolved_objective,
            memories=evidence,
            status=GoalStatus.ACTIVE,
            goal_id=goal_id,
            assessment=assessment,
        )
        from fibmind.planning import advance_goal_steps

        goal = advance_goal_steps(goal, "plan")
        self.memory.revise(goal.goal_id, content=encode_goal(goal), identity=identity)
        self.observe(
            "plan",
            f"planned {goal.objective}",
            {"goal_id": goal.goal_id, "steps": len(goal.steps)},
            state=state,
        )
        payload = goal.to_dict()
        payload["text"] = pack.get("text", "")
        payload["hits"] = pack.get("hits", [])
        payload["assessment"] = assessment
        payload["state"] = identity
        payload["capabilities"] = infer_capabilities(resolved_objective)
        payload["planner"] = self.planner.name
        return payload

    def coordinate(
        self,
        objective: str | None = None,
        goal_id: str | None = None,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Name the next capabilities to involve, and advise each suggested action.

        Coordination is a recommendation, not a plugin manager: the harness
        still executes. A rejected advise marks that capability as blocked.
        """
        planned = self.plan(objective=objective, goal_id=goal_id, state=state)
        identity = (state or BrainState()).identity()
        procedures = self._matching_procedures(planned["objective"], identity)
        if procedures:
            work = _capabilities_from_procedures(procedures) or list(planned["capabilities"])
            source = "procedure"
        else:
            work = list(planned["capabilities"])
            source = "heuristic"

        # Integrate assessment routes into capabilities (verify->test, code->code, search->search, ask->ask)
        assessment = planned.get("assessment") or {}
        routes_map = {
            "verify": "test",
            "code": "code",
            "search": "search",
            "ask": "ask",
        }
        assessment_capabilities: list[str] = []
        for step in assessment.get("next", []):
            route = step.get("route")
            if route in routes_map:
                cap = routes_map[route]
                if cap not in work and cap not in assessment_capabilities:
                    assessment_capabilities.append(cap)
        work = work + assessment_capabilities

        sequence: list[tuple[str, str]] = [("memory", CAPABILITIES["memory"])]
        for capability in work:
            sequence.append((capability, CAPABILITIES[capability]))
        sequence.append(("memory", "Reflect on evidence, then remember what is worth keeping"))

        seen: set[str] = set()
        capabilities: list[dict[str, Any]] = []
        blocked = False
        for capability, reason in sequence:
            key = capability if capability != "memory" else f"memory:{reason}"
            if key in seen:
                continue
            seen.add(key)
            kind, action = CAPABILITY_ACTIONS[capability]
            if capability == "memory" and "Reflect" in reason:
                kind, action = "brain", "reflect"
            decision = self.advise(action, kind=kind, state=state)
            item = {
                "capability": capability,
                "reason": reason,
                "action": action,
                "kind": kind,
                "advise": {
                    "verdict": decision["verdict"],
                    "reason": decision["reason"],
                },
            }
            if decision["verdict"] == AdviseVerdict.REJECT.value:
                blocked = True
            capabilities.append(item)

        if blocked:
            blocked_goal = BrainGoal(
                goal_id=planned["goal_id"],
                objective=planned["objective"],
                status=GoalStatus.BLOCKED,
                steps=tuple(PlanStep.from_dict(step) for step in planned["steps"]),
            )
            self.memory.revise(
                planned["goal_id"], content=encode_goal(blocked_goal), identity=(state or BrainState()).identity()
            )
            planned["status"] = GoalStatus.BLOCKED.value

            sess_id = identity.get("session_id")
            if sess_id:
                parent_sess = self._load_session_parent(sess_id)
                if parent_sess:
                    self.send_ipc(
                        sender=sess_id,
                        recipient=parent_sess,
                        signal="child_blocked",
                        payload={
                            "child_session_id": sess_id,
                            "goal_id": planned.get("goal_id"),
                            "objective": planned.get("objective"),
                        },
                        state=state,
                    )

        self.observe(
            "coordinate",
            f"coordinated {planned['objective']}",
            {"goal_id": planned["goal_id"], "blocked": blocked},
            state=state,
        )
        return {
            "goal_id": planned["goal_id"],
            "objective": planned["objective"],
            "status": planned["status"],
            "steps": planned["steps"],
            "capabilities": capabilities,
            "capability_source": source,
            "procedures": procedures,
            "blocked": blocked,
            "state": planned["state"],
        }

    def _matching_procedures(self, objective: str, identity: dict[str, str | None]) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for row in self._procedures(identity, query=objective, top_k=3):
            procedure = Procedure.parse(row["content"])
            if procedure is None or row.get("score", 0.0) <= 0.0:
                continue
            outcomes = row["outcomes"]
            matches.append(
                {
                    "node_id": row["node_id"],
                    "title": row["title"],
                    "trigger": procedure.trigger,
                    "steps": list(procedure.steps),
                    "tools": list(procedure.tools),
                    "verify": procedure.verify,
                    "score": row.get("score", 0.0),
                    "maturity": _maturity(outcomes),
                    "outcomes": outcomes,
                }
            )
        return matches

    def complete_goal(
        self,
        goal_id: str | None = None,
        objective: str | None = None,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Mark a persisted goal complete. The node stays recallable as history."""
        goal = self._resolve_goal(objective, goal_id, state, allow_missing=False)
        from fibmind.planning import advance_goal_steps

        completed = advance_goal_steps(goal, "complete_goal")
        self.memory.revise(goal.goal_id, content=encode_goal(completed), identity=(state or BrainState()).identity())
        self.observe("complete_goal", f"completed {goal.objective}", {"goal_id": goal.goal_id}, state=state)
        return completed.to_dict()

    def reflect(
        self,
        verdict: str,
        source: str,
        node_id: str | None = None,
        note: str | None = None,
        goal: str | None = None,
        state: BrainState | None = None,
        trust_ring: str | None = None,
    ) -> dict[str, Any]:
        """Record whether a memory held up. Resolves ``node_id`` from ``goal`` if needed."""
        parsed = Verdict(verdict)
        target = node_id
        if target is None:
            if not goal or not goal.strip():
                raise ValueError("reflect needs node_id or a goal that recalls a memory")
            identity = (state or BrainState()).identity()
            found = self.memory.search(
                goal,
                top_k=1,
                **_identity_kwargs(identity),
            )
            results = found["results"]
            if not results:
                raise ValueError("reflect found no memory matching the goal")
            target = results[0]["node_id"]
        identity = (state or BrainState()).identity()
        outcome_kwargs: dict[str, Any] = {
            "note": note,
            "session_id": identity["session_id"],
            "identity": identity,
        }
        if "trust_ring" in inspect.signature(self.memory.record_outcome).parameters:
            outcome_kwargs["trust_ring"] = trust_ring
        updated = self.memory.record_outcome(target, parsed.value, source, **outcome_kwargs)
        if updated.get("memory_kind") == MemoryKind.PROCEDURE.value:
            updated["evolution"] = self._evolve_procedure(target, identity)
        self.observe(
            "reflect",
            f"{parsed.value} {updated['title']} via {source}",
            {"node_id": target, "verdict": parsed.value, "source": source},
            state=state,
        )
        if parsed == Verdict.CONFIRMED:
            self._try_advance_active_goal("reflect_confirmed", state, target=updated.get("title", ""))
        return updated

    def resolve_conflict(
        self,
        winning_node_id: str,
        losing_node_id: str,
        reason: str,
        source: str,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Arbitrate a conflict: refute losing node with evidence and clear conflict on winner."""
        identity = (state or BrainState()).identity()
        result = self.memory.resolve_conflict(
            winning_node_id,
            losing_node_id,
            reason=reason,
            source=source,
            identity=identity,
        )
        self.observe(
            "resolve_conflict",
            f"arbitrated conflict: {winning_node_id} over {losing_node_id} via {source}",
            {"winner_id": winning_node_id, "loser_id": losing_node_id, "reason": reason, "source": source},
            state=state,
        )
        self._try_advance_active_goal("resolve_conflict", state, target=winning_node_id)
        return result

    def synthesize_conflict(
        self,
        node_id_a: str,
        node_id_b: str,
        *,
        title: str,
        content: str,
        conditions_a: list[str] | tuple[str, ...] = (),
        conditions_b: list[str] | tuple[str, ...] = (),
        source: str = "dialectical_synthesis",
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Synthesize conflicting memories into a higher-order principle (Dao dialectics)."""
        identity = (state or BrainState()).identity()
        result = self.memory.synthesize_conflict(
            node_id_a,
            node_id_b,
            title=title,
            content=content,
            conditions_a=conditions_a,
            conditions_b=conditions_b,
            source=source,
            identity=identity,
        )
        self.observe(
            "synthesize_conflict",
            f"synthesized {node_id_a} and {node_id_b} into {result['node_id']} ({title})",
            {
                "synthesized_node_id": result["node_id"],
                "node_id_a": node_id_a,
                "node_id_b": node_id_b,
                "source": source,
            },
            state=state,
        )
        self._try_advance_active_goal("synthesize_conflict", state, target=title)
        return result

    def _evolve_procedure(self, node_id: str, identity: dict[str, str | None]) -> dict[str, Any]:
        """Apply the deterministic evolution rules to one procedure.

        Retirement is automatic: refuted twice with no confirmation means the
        procedure does not work here. Promotion into shared knowledge is only
        *reported*: it still goes through ``promote_knowledge`` and its
        three-supporter bar, which one node cannot satisfy by itself.
        """
        outcomes = self.memory.read(lambda memory: memory.outcome_counts(node_id))
        retired = False
        if retirement_due(outcomes):
            current = self.memory.inspect(node_id, identity=identity)
            if current["status"] == MemoryStatus.ACTIVE.value:
                self.memory.mark_stale(
                    node_id,
                    f"procedure refuted {outcomes['refuted']}× with no confirmation",
                    identity=identity,
                )
            retired = True
        return {
            **outcomes,
            "maturity": _maturity(outcomes),
            "promotion_ready": promotion_ready(outcomes),
            "retired": retired,
        }

    # ------------------------------------------------------------------
    # Procedural memory

    def remember_procedure(
        self,
        title: str,
        trigger: str,
        steps: list[str],
        when: list[str] | None = None,
        tools: list[str] | None = None,
        verify: str = "",
        inputs: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        state: BrainState | None = None,
        cwd: str | Path | None = None,
    ) -> dict[str, Any]:
        """Store a repeatable way of doing a class of task.

        Goes through the same admission gate as any memory. If an active
        procedure with the same trigger already exists in this working context,
        the new one is linked to it as a ``version_of`` so both stay traceable;
        outcomes decide which one ``render`` prefers.
        """
        procedure = Procedure(
            trigger=_require_text(trigger, "trigger"),
            steps=tuple(step.strip() for step in steps if step and step.strip()),
            when=tuple(item.strip() for item in (when or []) if item and item.strip()),
            tools=tuple(item.strip() for item in (tools or []) if item and item.strip()),
            verify=(verify or "").strip(),
            inputs={str(k): str(v) for k, v in (inputs or {}).items()},
        )
        if not procedure.steps:
            raise ValueError("a procedure needs at least one step")
        identity = (state or BrainState()).identity()
        previous = [
            node["node_id"]
            for node in self._procedures(identity, query=None)
            if Procedure.parse(node["content"]) is not None
            and Procedure.parse(node["content"]).trigger.casefold() == procedure.trigger.casefold()
        ]
        proc_meta = {"kind": "fibbrain_procedure_node", "trigger": procedure.trigger}
        if metadata:
            proc_meta.update(metadata)
        result = self.remember(
            PROCEDURE_CATEGORY,
            _require_text(title, "title"),
            procedure.encode(),
            tags=[PROCEDURE_TAG],
            metadata=proc_meta,
            state=state,
            cwd=cwd,
        )
        if result.get("verdict") == AdmitVerdict.WRITE.value:
            for old_id in previous:
                self.memory.link(
                    result["node_id"], old_id, RelationType.VERSION_OF.value, weight=1.0, identity=identity
                )
            result["supersedes"] = previous
        return result

    def render(
        self,
        goal: str,
        state: BrainState | None = None,
        format: str = "skill",
        top_k: int = 3,
        min_maturity: str = "candidate",
    ) -> dict[str, Any]:
        """Turn the procedures that fit ``goal`` into skills or tool definitions.

        ``format`` is ``skill`` (a SKILL.md document per procedure) or ``tool``
        (a JSON-Schema tool definition per procedure). ``min_maturity`` filters
        by evidence: ``candidate`` (anything active), ``verified`` (confirmed at
        least once), ``established`` (meets the promotion bar). A ``goal`` of
        ``"*"`` lists every visible procedure instead of searching — what a
        host's skill catalog needs, since it has no goal at lookup time.
        Rendering never executes anything; the harness decides whether to
        register the result.
        """
        _require_text(goal, "goal")
        list_all = goal.strip() == "*"
        if format not in {"skill", "tool"}:
            raise ValueError("format must be 'skill' or 'tool'")
        if min_maturity not in _MATURITY_ORDER:
            raise ValueError(f"min_maturity must be one of {sorted(_MATURITY_ORDER)}")
        identity = (state or BrainState()).identity()
        rendered: list[dict[str, Any]] = []
        rows = self._procedures(identity, query=None if list_all else goal, top_k=top_k * 2)
        for row in rows:
            procedure = Procedure.parse(row["content"])
            if procedure is None:
                continue
            outcomes = row["outcomes"]
            maturity = _maturity(outcomes)
            if _MATURITY_ORDER[maturity] < _MATURITY_ORDER[min_maturity]:
                continue
            # ``render_skill`` / ``render_tool`` read only id, title and
            # content off the node, all of which the row already carries.
            node = _node_from_summary(row)
            item = (
                render_skill(node, procedure, outcomes) if format == "skill" else render_tool(node, procedure, outcomes)
            )
            item.update(
                {
                    "node_id": row["node_id"],
                    "title": row["title"],
                    "score": row.get("score", 0.0),
                    "confidence": row.get("confidence", 0.0),
                    "maturity": maturity,
                    "outcomes": outcomes,
                }
            )
            rendered.append(item)
            if len(rendered) >= top_k:
                break
        return {"goal": goal, "format": format, "state": identity, "items": rendered}

    def _procedures(
        self,
        identity: dict[str, str | None],
        query: str | None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Active procedure nodes visible to this identity, best match first.

        One store read for the whole list. It used to be a search followed by
        an ``inspect`` per row, and every ``inspect`` reloaded the entire
        forest; the rows now carry ``content`` and ``outcomes`` so callers do
        not need a second lookup either.
        """
        kwargs = _identity_kwargs(identity)

        def run(memory: FibMind) -> list[dict[str, Any]]:
            if query:
                candidates = [
                    (hit.node, hit.score)
                    for hit in memory.search(query, top_k=top_k, categories={PROCEDURE_CATEGORY}, **kwargs)
                ]
            else:
                candidates = [
                    (node, 0.0)
                    for node in memory.nodes.values()
                    if is_procedure_node(node) and memory.is_visible(node, **kwargs)
                ]
            rows = [
                {
                    **_node_summary(node),
                    "score": score,
                    "content": node.content,
                    "outcomes": memory.outcome_counts(node.id),
                }
                for node, score in candidates
                if is_procedure_node(node)
            ]
            rows.sort(key=lambda row: (row["score"], row["confidence"]), reverse=True)
            return rows

        return self.memory.read(run)

    # ------------------------------------------------------------------
    # Self-tuning

    def tune(
        self,
        apply: bool = True,
        evaluate: Any = None,
        dataset_dir: str | None = None,
    ) -> dict[str, Any]:
        """Propose a bounded ranking-weight change from evidence; adopt it only
        if the evaluation gate passes.

        Deterministic and auditable: the evidence profile, the proposal, the
        gate scores, and the decision are all returned and appended to the log
        as one ``tuning`` observation. ``apply=False`` runs everything but the
        adoption. ``evaluate`` may replace the P0 suite in tests.
        """
        from fibmind.tuning import (
            admission_pressure,
            evaluate_with_weights,
            gate,
            profile_evidence,
            propose,
            record_tuning,
        )

        current = self.memory.weights
        profile = self.memory.read(profile_evidence)
        pressure = self.memory.read(admission_pressure)
        proposal = propose(current, profile)
        outcome: dict[str, Any] = {
            "current": current.to_dict(),
            "evidence": profile.to_dict(),
            "admission_pressure": pressure,
            "proposal": proposal.to_dict() if proposal else None,
            "gate": None,
            "adopted": False,
        }
        if proposal is None:
            outcome["decision"] = "no_change: insufficient or non-directional evidence"
        else:
            runner = evaluate or (lambda weights: evaluate_with_weights(weights, dataset_dir))
            result = gate(current, proposal.weights, evaluate=runner)
            outcome["gate"] = result.to_dict()
            if result.passed and apply:
                self.memory.set_weights(proposal.weights)
                outcome["adopted"] = True
                outcome["decision"] = "adopted: gate passed"
            elif result.passed:
                outcome["decision"] = "gate passed; not applied (apply=False)"
            else:
                outcome["decision"] = "rejected: " + "; ".join(result.reasons)
        self.memory.run(lambda memory: record_tuning(memory, outcome))
        return outcome

    def revalidation_candidates(
        self, state: BrainState | None = None, older_than_days: int = 90
    ) -> list[dict[str, Any]]:
        """Knowledge and procedures with no confirmation for a long time.

        Output only: nothing is changed. The list is what a maintenance pass or
        a human should re-check; auto-retiring on age alone would throw away
        true-but-quiet knowledge.
        """
        identity = (state or BrainState()).identity()
        cutoff = utc_now() - timedelta(days=older_than_days)

        def run(memory: FibMind) -> list[dict[str, Any]]:
            rows = []
            for node in memory.nodes.values():
                if node.status != MemoryStatus.ACTIVE:
                    continue
                if node.scope != MemoryScope.KNOWLEDGE and node.memory_kind != MemoryKind.PROCEDURE:
                    continue
                if not memory.is_visible(node, **_identity_kwargs(identity)):
                    continue
                last = _last_confirmation(memory, node.id) or node.created_at
                if last >= cutoff:
                    continue
                rows.append(
                    {
                        **_node_summary(node),
                        "last_confirmed_at": last.isoformat(),
                        "days_since": (utc_now() - last).days,
                        "outcomes": memory.outcome_counts(node.id),
                    }
                )
            rows.sort(key=lambda row: row["days_since"], reverse=True)
            return rows

        return self.memory.read(run)

    def revalidate_memories(
        self,
        state: BrainState | None = None,
        *,
        cwd: str | Path | None = None,
        execute_commands: bool = False,
        older_than_days: int = 90,
    ) -> dict[str, Any]:
        """Revalidate knowledge and procedure memories that have gone unconfirmed."""
        import fibmind.verify as verify_module

        if hasattr(verify_module, "revalidate_memories"):
            return verify_module.revalidate_memories(
                self,
                state=state,
                cwd=cwd,
                execute_commands=execute_commands,
                older_than_days=older_than_days,
            )

        candidates = self.revalidation_candidates(state=state, older_than_days=older_than_days)
        revalidated: list[dict[str, Any]] = []
        for cand in candidates:
            nid = cand["node_id"]
            if execute_commands:
                meta = self.memory.inspect(nid).get("metadata", {})
                verify_cmd = meta.get("verify")
                if verify_cmd:
                    try:
                        res = verify_module.run_command_and_reflect(
                            verify_cmd,
                            node_id=nid,
                            brain=self,
                            cwd=cwd,
                        )
                        revalidated.append({"node_id": nid, "status": "executed", "result": res})
                        continue
                    except Exception as exc:
                        revalidated.append({"node_id": nid, "status": "error", "error": str(exc)})
                        continue
            revalidated.append({"node_id": nid, "status": "candidate", "days_since": cand.get("days_since", 0)})

        return {
            "candidate_count": len(candidates),
            "revalidated": revalidated,
            "older_than_days": older_than_days,
        }

    # ------------------------------------------------------------------
    # Session process tree (fork & merge)

    def fork_session(
        self,
        parent_session_id: str,
        child_session_id: str,
        objective: str | None = None,
        *,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Fork a child subagent session from a parent session (CoW MMU page table)."""
        parent_session_id = _require_text(parent_session_id, "parent_session_id")
        child_session_id = _require_text(child_session_id, "child_session_id")
        if parent_session_id == child_session_id:
            raise ValueError("child_session_id must be different from parent_session_id")

        self._save_session_parent(child_session_id, parent_session_id)

        base_state = state or BrainState()
        child_state = BrainState(
            owner=base_state.owner,
            workspace_id=base_state.workspace_id,
            project_id=base_state.project_id,
            session_id=child_session_id,
            task_id=base_state.task_id,
            model_id=base_state.model_id,
        )
        parent_state = BrainState(
            owner=base_state.owner,
            workspace_id=base_state.workspace_id,
            project_id=base_state.project_id,
            session_id=parent_session_id,
            task_id=base_state.task_id,
            model_id=base_state.model_id,
        )

        parent_pages = dict(self._load_session_pages(parent_session_id))
        self._save_session_pages(child_session_id, parent_pages)
        parent_step = self._load_session_step(parent_session_id)
        self._save_session_step(child_session_id, parent_step)
        parent_last_steps = dict(self._load_session_last_step(parent_session_id))
        self._save_session_last_step(child_session_id, parent_last_steps)

        parent_pinned = set(self._load_session_pinned(parent_session_id))
        self._save_session_pinned(child_session_id, parent_pinned)

        child_goal_id: str | None = None
        parent_goal_id: str | None = None
        if objective and objective.strip():
            parent_goal = self._resolve_goal(None, None, parent_state, allow_missing=True)
            if parent_goal is not None:
                parent_goal_id = parent_goal.goal_id
            child_plan = self.plan(objective=objective.strip(), state=child_state)
            child_goal_id = child_plan.get("goal_id")
            if parent_goal_id and child_goal_id:
                self.memory.link(
                    child_goal_id,
                    parent_goal_id,
                    RelationType.DERIVED_FROM.value,
                    identity=child_state.identity(),
                )

        # Copy/inherit parent's hot snapshot for child_session_id
        parent_hot = self.hot_snapshot(parent_state)
        node_ids = parent_hot.get("node_ids", [])
        if node_ids and child_session_id is not None:
            identity = child_state.identity()
            self.memory.run(
                lambda memory: memory.append(
                    category=HOT_CATEGORY,
                    title=f"hot snapshot for {child_session_id}",
                    content=json.dumps({"kind": HOT_KIND, "node_ids": node_ids}, ensure_ascii=False),
                    tags=[HOT_TAG],
                    metadata={"kind": HOT_KIND, "count": len(node_ids), "forked_from": parent_session_id},
                    scope=MemoryScope.SESSION,
                    owner=identity["owner"],
                    workspace_id=identity["workspace_id"],
                    project_id=identity["project_id"],
                    session_id=child_session_id,
                    task_id=identity["task_id"],
                    auto_link=False,
                    compress=False,
                )
            )

        self.observe(
            "fork_session",
            f"forked child session {child_session_id}" + (f" for {objective}" if objective else ""),
            {
                "parent_session_id": parent_session_id,
                "child_session_id": child_session_id,
                "objective": objective,
                "inherited_pages": len(parent_pages),
                "child_goal_id": child_goal_id,
                "parent_goal_id": parent_goal_id,
            },
            state=child_state,
        )

        return {
            "parent_session_id": parent_session_id,
            "child_session_id": child_session_id,
            "inherited_pages": len(parent_pages),
            "page_ids": sorted(parent_pages),
            "objective": objective,
            "child_goal_id": child_goal_id,
            "parent_goal_id": parent_goal_id,
            "state": child_state.identity(),
        }

    def merge_session(
        self,
        parent_session_id: str,
        child_session_id: str,
        *,
        mode: str = ReviewMode.AUTO.value,
        close_child: bool = True,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Merge a completed child subagent session into its parent session."""
        parent_session_id = _require_text(parent_session_id, "parent_session_id")
        child_session_id = _require_text(child_session_id, "child_session_id")

        base_state = state or BrainState()
        child_state = BrainState(
            owner=base_state.owner,
            workspace_id=base_state.workspace_id,
            project_id=base_state.project_id,
            session_id=child_session_id,
            task_id=base_state.task_id,
            model_id=base_state.model_id,
        )
        parent_state = BrainState(
            owner=base_state.owner,
            workspace_id=base_state.workspace_id,
            project_id=base_state.project_id,
            session_id=parent_session_id,
            task_id=base_state.task_id,
            model_id=base_state.model_id,
        )

        child_pinned = self._load_session_pinned(child_session_id)
        parent_pinned = set(self._load_session_pinned(parent_session_id))
        self._save_session_pinned(parent_session_id, parent_pinned | child_pinned)

        completed_child_goal_id: str | None = None
        child_goal = self._resolve_goal(None, None, child_state, allow_missing=True)

        review = self.review_session(child_state, mode=mode, close=close_child)

        if close_child and child_goal is not None and child_goal.status != GoalStatus.COMPLETE:
            self.complete_goal(goal_id=child_goal.goal_id, state=child_state)
            completed_child_goal_id = child_goal.goal_id
            self._try_advance_active_goal("remember", parent_state, target=child_goal.objective)

        child_pages = self._load_session_pages(child_session_id)
        parent_pages = dict(self._load_session_pages(parent_session_id))
        for nid, rev in child_pages.items():
            parent_pages[nid] = max(parent_pages.get(nid, 0), rev)
        for w in review.get("written", []):
            nid = w.get("node_id")
            if nid:
                rev = w.get("revision", 1)
                parent_pages[nid] = max(parent_pages.get(nid, 0), rev)
        self._save_session_pages(parent_session_id, parent_pages)

        child_last_steps = self._load_session_last_step(child_session_id)
        parent_last_steps = dict(self._load_session_last_step(parent_session_id))
        for nid, step_val in child_last_steps.items():
            parent_last_steps[nid] = max(parent_last_steps.get(nid, 0), step_val)
        self._save_session_last_step(parent_session_id, parent_last_steps)

        self.observe(
            "merge_session",
            f"merged child session {child_session_id} ({len(review.get('written', []))} memories)",
            {
                "parent_session_id": parent_session_id,
                "child_session_id": child_session_id,
                "written_count": len(review.get("written", [])),
                "completed_child_goal_id": completed_child_goal_id,
            },
            state=parent_state,
        )

        if close_child:
            self.release_lease(child_session_id)

        self.send_ipc(
            sender=child_session_id,
            recipient=parent_session_id,
            signal="child_merged",
            payload={
                "child_session_id": child_session_id,
                "written_count": len(review.get("written", [])),
                "completed_child_goal_id": completed_child_goal_id,
            },
            state=parent_state,
        )

        return {
            "parent_session_id": parent_session_id,
            "child_session_id": child_session_id,
            "review": review,
            "merged_pages": len(parent_pages),
            "completed_child_goal_id": completed_child_goal_id,
            "closed_child": close_child,
            "state": parent_state.identity(),
        }

    # ------------------------------------------------------------------
    # Session review

    def review_session(
        self,
        state: BrainState,
        mode: str = ReviewMode.APPROVE.value,
        close: bool = False,
        *,
        cwd: str | Path | None = None,
    ) -> dict[str, Any]:
        """Extract long-term candidates from one session's episode.

        ``mode``:

        - ``candidates`` — extract and run each through ``admit``; write nothing.
        - ``approve`` — write admitted candidates with status ``pending``; they
          stay out of recall until ``approve_memory`` moves them to ``active``.
        - ``auto`` — write admitted candidates as ``active`` immediately.

        Extraction is deterministic and admission rejects exact duplicates, so
        running the review twice writes nothing the second time. ``close``
        marks the reviewed episode nodes ``stale`` so a later review of the same
        session starts from a clean slate while the raw material stays
        inspectable.
        """
        parsed_mode = ReviewMode(mode)
        identity = state.identity()
        session_id = identity["session_id"]
        if session_id is None:
            raise ValueError("review_session needs a session_id in state")

        effective_cwd = cwd or getattr(state, "workspace_path", None)

        def run(memory: FibMind) -> dict[str, Any]:
            episodes = _episode_nodes(memory, identity)
            goals = _goal_nodes(memory, identity)
            digest = digest_episode(session_id, episodes, goals)
            candidates = extract_candidates(digest)
            written: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []
            for candidate in candidates:
                cand_meta = dict(candidate.metadata)
                for src_id in candidate.source_node_ids:
                    src_node = memory.nodes.get(src_id)
                    if src_node:
                        payload = src_node.metadata.get("payload")
                        if isinstance(payload, dict):
                            if "symbols" in payload and "symbols" not in cand_meta:
                                cand_meta["symbols"] = list(payload["symbols"])
                            if "files" in payload and "files" not in cand_meta:
                                cand_meta["files"] = list(payload["files"])

                if effective_cwd is not None and cand_meta:
                    cand_meta = _auto_ground(cand_meta, effective_cwd) or cand_meta

                mem_candidate = _review_candidate(candidate, identity)
                decision = decide_admission(memory, mem_candidate)
                entry = {**candidate.to_dict(), "admit": decision.to_dict()}
                if cand_meta:
                    entry["metadata"] = dict(cand_meta)
                if decision.verdict != AdmitVerdict.WRITE:
                    skipped.append(entry)
                    continue
                if parsed_mode == ReviewMode.CANDIDATES:
                    written.append(entry)
                    continue
                status = MemoryStatus.PENDING if parsed_mode == ReviewMode.APPROVE else MemoryStatus.ACTIVE
                node_id = memory.append(
                    category=candidate.category,
                    title=candidate.title,
                    content=candidate.content,
                    tags=list(candidate.tags),
                    metadata={
                        **cand_meta,
                        "kind": REVIEW_KIND,
                        "session_id": session_id,
                        "source_node_ids": list(candidate.source_node_ids),
                        "mode": parsed_mode.value,
                    },
                    scope=candidate.scope,
                    owner=identity["owner"],
                    workspace_id=identity["workspace_id"],
                    project_id=identity["project_id"],
                    session_id=session_id,
                    task_id=identity["task_id"],
                    status=status,
                    memory_kind=candidate.memory_kind,
                )
                entry.update(_node_summary(memory.nodes[node_id]))
                written.append(entry)
            closed = 0
            if close and parsed_mode != ReviewMode.CANDIDATES:
                for node in episodes:
                    memory.set_status(node.id, MemoryStatus.STALE, f"reviewed in {parsed_mode.value} mode")
                    closed += 1
            return {
                "next_hot": [node.id for node in _select_hot(memory, identity)]
                if parsed_mode != ReviewMode.CANDIDATES
                else None,
                "session_id": session_id,
                "mode": parsed_mode.value,
                "objective": digest.objective,
                "episode_count": len(episodes),
                "candidates": len(candidates),
                "written": written,
                "skipped": skipped,
                "closed_episodes": closed,
                "state": identity,
            }

        if parsed_mode == ReviewMode.CANDIDATES:
            res = self.memory.read(run)
        else:
            res = self.memory.run(run)
        if close:
            self.release_lease(session_id)
        return res

    def approve_memory(
        self, node_id: str, reason: str = "approved by reviewer", state: BrainState | None = None
    ) -> dict[str, Any]:
        """Move a ``pending`` review memory into normal recall."""
        return self._resolve_pending(node_id, MemoryStatus.ACTIVE, reason, state)

    def reject_memory(
        self, node_id: str, reason: str = "rejected by reviewer", state: BrainState | None = None
    ) -> dict[str, Any]:
        """Retire a ``pending`` review memory. It keeps its provenance as ``stale``."""
        return self._resolve_pending(node_id, MemoryStatus.STALE, reason, state)

    def pending_reviews(self, state: BrainState) -> list[dict[str, Any]]:
        """Review memories awaiting approval in this working context."""
        identity = state.identity()

        def run(memory: FibMind) -> list[dict[str, Any]]:
            rows = []
            for node in memory.nodes.values():
                if node.status != MemoryStatus.PENDING or REVIEW_TAG not in node.tags:
                    continue
                if not memory.is_visible(
                    node,
                    statuses={MemoryStatus.PENDING},
                    **_identity_kwargs(identity),
                ):
                    continue
                rows.append({**_node_summary(node), "content": node.content})
            rows.sort(key=lambda row: row["title"])
            return rows

        return self.memory.read(run)

    def _resolve_pending(
        self, node_id: str, status: MemoryStatus, reason: str, state: BrainState | None
    ) -> dict[str, Any]:
        _require_text(node_id, "node_id")
        identity = (state or BrainState()).identity()
        current = self.memory.inspect(node_id, identity=identity)
        if current["status"] != MemoryStatus.PENDING.value:
            raise ValueError(f"node {node_id} is {current['status']}, not pending")
        return self.memory.set_status(node_id, status.value, reason, identity=identity)

    def _resolve_goal(
        self,
        objective: str | None,
        goal_id: str | None,
        state: BrainState | None,
        *,
        allow_missing: bool,
    ):
        if goal_id and goal_id.strip():
            inspected = self.memory.inspect(goal_id, identity=(state or BrainState()).identity())
            if not is_goal_node(inspected):
                raise ValueError(f"node {goal_id} is not a FibBrain goal")
            return parse_goal_document(inspected["node_id"], inspected["content"], inspected["title"])

        if objective and objective.strip():
            identity = (state or BrainState()).identity()
            found = self.memory.search(
                objective,
                top_k=8,
                categories=[GOAL_CATEGORY],
                **_identity_kwargs(identity),
            )
            title = _goal_title(objective)
            for hit in found["results"]:
                if not is_goal_node(hit) or hit["title"] != title:
                    continue
                inspected = self.memory.inspect(hit["node_id"], identity=identity)
                parsed = parse_goal_document(inspected["node_id"], inspected["content"], inspected["title"])
                if parsed.status != GoalStatus.COMPLETE:
                    return parsed
            if allow_missing:
                return None
            raise ValueError("no active goal matches the objective")

        identity = (state or BrainState()).identity()
        if identity.get("session_id") or identity.get("task_id"):
            goals = self.memory.read(lambda memory: _goal_nodes(memory, identity))
            for node in reversed(goals):
                if node.status == MemoryStatus.ACTIVE:
                    parsed = parse_goal_document(node.id, node.content, node.title)
                    if parsed.status != GoalStatus.COMPLETE:
                        return parsed

        if allow_missing:
            return None
        raise ValueError("plan needs an objective or a goal_id")

    def _try_advance_active_goal(
        self,
        event_kind: str,
        state: BrainState | None,
        target: str = "",
    ) -> None:
        try:
            goal = self._resolve_goal(None, None, state, allow_missing=True)
            if goal is not None and goal.status != GoalStatus.COMPLETE:
                from fibmind.planning import advance_goal_steps, encode_goal

                advanced = advance_goal_steps(goal, event_kind, target=target)
                self.memory.revise(
                    advanced.goal_id,
                    content=encode_goal(advanced),
                    identity=(state or BrainState()).identity(),
                )
        except Exception:
            pass

    def _refuted_hits(self, objective: str, identity: dict[str, str | None]) -> list[dict[str, Any]]:
        found = self.memory.search(
            objective,
            top_k=5,
            statuses=[MemoryStatus.REFUTED.value],
            **_identity_kwargs(identity),
        )
        return list(found["results"])


_MATURITY_ORDER = {"candidate": 0, "verified": 1, "established": 2}

# Which harness capability a tool name implies. Unknown tools fall back to the
# objective heuristic in ``planning.infer_capabilities``.
_TOOL_CAPABILITIES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pytest", "test", "unittest", "jest", "vitest", "go test", "cargo test"), "test"),
    (("edit", "write", "patch", "apply_patch", "sed", "refactor"), "code"),
    (("search", "grep", "web", "fetch", "browse"), "search"),
    (("doc", "readme", "markdown"), "docs"),
    (("diff", "review", "read"), "review"),
)


def _maturity(outcomes: dict[str, int]) -> str:
    if promotion_ready(outcomes):
        return "established"
    if outcomes.get("confirmed", 0) > 0:
        return "verified"
    return "candidate"


def _capabilities_from_procedures(procedures: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for procedure in procedures:
        for name in [*procedure.get("tools", []), *procedure.get("steps", [])]:
            text = str(name).casefold()
            for needles, capability in _TOOL_CAPABILITIES:
                if capability not in found and any(needle in text for needle in needles):
                    found.append(capability)
    return found


def _node_from_summary(inspected: dict[str, Any]) -> MemoryNode:
    """A light node view over a summary row.

    Used where only identity and text matter: ``is_procedure_node`` (kind and
    tags) and the renderers (id, title, content). ``node_id`` is carried over
    because ``render_skill`` prints it as the reflect target.
    """
    node = MemoryNode(title=inspected["title"], content=inspected.get("content", ""), category=inspected["category"])
    if inspected.get("node_id"):
        node.id = inspected["node_id"]
    node.tags = set(inspected.get("tags") or [])
    kind = inspected.get("memory_kind")
    node.memory_kind = MemoryKind(kind) if kind else None
    return node


def _last_confirmation(memory: FibMind, node_id: str) -> datetime | None:
    return memory.last_confirmation(node_id)


def _select_hot(
    memory: FibMind,
    identity: dict[str, str | None],
    model_id: str | None = None,
) -> list[MemoryNode]:
    """Deterministic L0 selection: hot kinds, visible, active, non-conflicted, evidence first."""
    from fibmind.tokens import DEFAULT_COUNTER

    kwargs = _identity_kwargs(identity)
    candidates = [
        node
        for node in memory.nodes.values()
        if node.memory_kind is not None
        and node.memory_kind.value in HOT_KINDS
        and node.node_type.value == "raw"
        and node.category not in {EPISODE_CATEGORY, HOT_CATEGORY, GOAL_CATEGORY}
        and not node.metadata.get("conflicted")
        and not (
            model_id
            and node.metadata.get("grounding") == "model_heuristic"
            and node.metadata.get("model_id")
            and str(node.metadata.get("model_id")) != str(model_id)
        )
        and memory.is_visible(node, **kwargs)
    ]
    candidates.sort(key=lambda node: (node.confidence, node.updated_at, node.id), reverse=True)
    selected: list[MemoryNode] = []
    spent = 0
    for node in candidates:
        cost = DEFAULT_COUNTER.count(f"- [{node.category}] {node.title}: {node.content[:160]}")
        if len(selected) >= HOT_MAX_ITEMS or spent + cost > HOT_BUDGET_TOKENS:
            break
        selected.append(node)
        spent += cost
    return selected


def _episode_nodes(memory: FibMind, identity: dict[str, str | None]) -> list[MemoryNode]:
    nodes = [
        node
        for node in memory.nodes.values()
        if node.category == EPISODE_CATEGORY
        and EPISODE_TAG in node.tags
        and memory.is_visible(node, **_identity_kwargs(identity))
    ]
    nodes.sort(key=lambda node: (node.created_at, node.id))
    return nodes


def _goal_nodes(memory: FibMind, identity: dict[str, str | None]) -> list[MemoryNode]:
    session_id = identity["session_id"]
    task_id = identity["task_id"]
    nodes = [
        node
        for node in memory.nodes.values()
        if is_goal_node({"category": node.category, "tags": node.tags})
        and (
            (session_id is not None and node.session_id == session_id)
            or (task_id is not None and node.task_id == task_id)
        )
        and memory.is_visible(node, **_identity_kwargs(identity))
    ]
    nodes.sort(key=lambda node: (node.created_at, node.id))
    return nodes


def _review_candidate(candidate: ReviewCandidate, identity: dict[str, str | None]) -> MemoryCandidate:
    return MemoryCandidate(
        category=candidate.category,
        title=candidate.title,
        content=candidate.content,
        tags=candidate.tags,
        scope=candidate.scope,
        owner=identity["owner"],
        workspace_id=identity["workspace_id"],
        project_id=identity["project_id"],
        session_id=identity["session_id"],
        task_id=identity["task_id"],
    )


def _candidate(
    category: str,
    title: str,
    content: str,
    tags: list[str] | None,
    scope: str,
    state: BrainState | None,
) -> MemoryCandidate:
    identity = (state or BrainState()).identity()
    return MemoryCandidate(
        category=_require_text(category, "category"),
        title=_require_text(title, "title"),
        content=_require_text(content, "content"),
        tags=tuple(tags or ()),
        scope=MemoryScope(scope),
        owner=identity["owner"],
        workspace_id=identity["workspace_id"],
        project_id=identity["project_id"],
        session_id=identity["session_id"],
        task_id=identity["task_id"],
    )


def _flatten_arguments(arguments: dict[str, Any] | None) -> str:
    """Argument values as searchable text; keys are structure, not content."""
    if not arguments:
        return ""
    parts: list[str] = []
    for value in arguments.values():
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)
        elif isinstance(value, dict):
            parts.append(_flatten_arguments(value))
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _identity_kwargs(identity: dict[str, str | None]) -> dict[str, str | None]:
    return dict(identity)


def _goal_title(objective: str) -> str:
    text = objective.strip()
    if len(text) <= GOAL_TITLE_LIMIT:
        return text
    return text[: GOAL_TITLE_LIMIT - 1].rstrip() + "…"


def brain_state(
    owner: str | None = None,
    workspace_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
) -> BrainState:
    return BrainState(
        owner=owner,
        workspace_id=workspace_id,
        project_id=project_id,
        session_id=session_id,
        task_id=task_id,
    )
