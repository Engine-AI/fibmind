"""FibBrain: the cognitive runtime that sits on top of the FibMind store.

Harnesses run the body (loop, tools, session). FibBrain decides what to
remember, what to load into a step, whether an action looks safe given past
evidence, which capabilities the body should involve, and how to record
whether a memory held up.

This is Brain v1 — Memory + Planning + Decision + Coordination + Reflection.
It does not own an agent loop or a plugin manager.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from fibmind.admission import AdmitVerdict, MemoryCandidate, decide_admission
from fibmind.graph import FibMind
from fibmind.models import MemoryNode, MemoryScope, MemoryStatus, Verdict, optional_id, utc_now
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
    synthesize_plan,
)
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


class AdviseVerdict(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"


EPISODE_LIMIT = 48
GOAL_TITLE_LIMIT = 200
EPISODE_KIND = "fibbrain_episode"


@dataclass(frozen=True, slots=True)
class BrainState:
    """The working context of one recall / write / advise call."""

    owner: str | None = None
    workspace_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None

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
    def from_node(cls, node: MemoryNode) -> "ObservedEvent":
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


class FibBrain:
    """Cognitive facade over a persisted FibMind store."""

    def __init__(self, service: MemoryService) -> None:
        self.memory = service
        self._episode: deque[ObservedEvent] = deque(maxlen=EPISODE_LIMIT)

    def observe(
        self,
        kind: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Record an episode event.

        With a ``session_id`` in ``state`` the event is written as a
        ``scope=session`` node in the ``episode`` category, so it survives a
        process restart and only that session can recall it. Without one it
        lives in the process-local buffer only. Either way it is not long-term
        memory: ``review_session`` decides what, if anything, gets promoted.
        """
        kind = _require_text(kind, "kind")
        summary = _require_text(summary, "summary")
        payload = dict(payload or {})
        identity = (state or BrainState()).identity()
        node_id: str | None = None
        if identity["session_id"] is not None:
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
    ) -> dict[str, Any]:
        """Load a context pack for ``goal`` inside ``state``."""
        identity = (state or BrainState()).identity()
        pack = self.memory.context(
            goal,
            top_k=top_k,
            depth=depth,
            max_chars=max_chars,
            exclude_categories=[EPISODE_CATEGORY],
            **_identity_kwargs(identity),
        )
        pack["state"] = identity
        pack["episode"] = [event.to_dict() for event in self._episode_for(identity)]
        return pack

    def _episode_for(self, identity: dict[str, str | None]) -> list[ObservedEvent]:
        """The persisted episode of this session, else the in-process buffer."""
        session_id = identity["session_id"]
        if session_id is None:
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
    ) -> dict[str, Any]:
        """Admit, then write. Skipped candidates are not persisted."""
        candidate = _candidate(category, title, content, tags, scope, state)
        return self.memory.remember(candidate, metadata=metadata)

    def advise(
        self,
        action: str,
        kind: str = "tool",
        state: BrainState | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Should this action run, given what we already know?"""
        _require_text(action, "action")
        identity = (state or BrainState()).identity()
        query = f"{kind} {action}"
        search_kwargs = _identity_kwargs(identity)
        active = self.memory.search(query, top_k=top_k, **search_kwargs)
        refuted = self.memory.search(
            query,
            top_k=top_k,
            statuses=[MemoryStatus.REFUTED.value],
            **search_kwargs,
        )
        notes = list(active["results"])
        blockers = list(refuted["results"])
        if blockers:
            decision = AdviseDecision(
                AdviseVerdict.REJECT,
                f"refuted evidence advises against {action}",
                memories=tuple(blockers + notes),
            )
        else:
            reason = (
                f"no refuted evidence against {action}"
                if not notes
                else f"recalled {len(notes)} related memories; no blocking refutation"
            )
            decision = AdviseDecision(AdviseVerdict.ALLOW, reason, memories=tuple(notes))
        return decision.to_dict()

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
        evidence = list(pack.get("hits") or []) + self._refuted_hits(resolved_objective, identity)
        if existing is None:
            written = self.memory.append(
                GOAL_CATEGORY,
                _goal_title(resolved_objective),
                encode_goal(
                    synthesize_plan(resolved_objective, memories=evidence)
                ),
                tags=[GOAL_TAG],
                metadata={"kind": "fibbrain_goal"},
                scope=MemoryScope.PERSONAL.value,
                **identity,
            )
            goal_id = written["node_id"]
        else:
            goal_id = existing.goal_id
        goal = synthesize_plan(
            resolved_objective,
            memories=evidence,
            status=GoalStatus.ACTIVE,
            goal_id=goal_id,
        )
        self.memory.revise(goal.goal_id, content=encode_goal(goal))
        self.observe(
            "plan",
            f"planned {goal.objective}",
            {"goal_id": goal.goal_id, "steps": len(goal.steps)},
            state=state,
        )
        payload = goal.to_dict()
        payload["text"] = pack.get("text", "")
        payload["hits"] = pack.get("hits", [])
        payload["state"] = identity
        payload["capabilities"] = infer_capabilities(resolved_objective)
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
        work = list(planned["capabilities"])
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
            self.memory.revise(planned["goal_id"], content=encode_goal(blocked_goal))
            planned["status"] = GoalStatus.BLOCKED.value

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
            "blocked": blocked,
            "state": planned["state"],
        }

    def complete_goal(
        self,
        goal_id: str | None = None,
        objective: str | None = None,
        state: BrainState | None = None,
    ) -> dict[str, Any]:
        """Mark a persisted goal complete. The node stays recallable as history."""
        goal = self._resolve_goal(objective, goal_id, state, allow_missing=False)
        completed = BrainGoal(
            goal_id=goal.goal_id,
            objective=goal.objective,
            status=GoalStatus.COMPLETE,
            steps=goal.steps,
        )
        self.memory.revise(goal.goal_id, content=encode_goal(completed))
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
        updated = self.memory.record_outcome(target, parsed.value, source, note=note)
        self.observe(
            "reflect",
            f"{parsed.value} {updated['title']} via {source}",
            {"node_id": target, "verdict": parsed.value, "source": source},
            state=state,
        )
        return updated

    # ------------------------------------------------------------------
    # Session review

    def review_session(
        self,
        state: BrainState,
        mode: str = ReviewMode.APPROVE.value,
        close: bool = False,
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

        def run(memory: FibMind) -> dict[str, Any]:
            episodes = _episode_nodes(memory, identity)
            goals = _goal_nodes(memory, identity)
            digest = digest_episode(session_id, episodes, goals)
            candidates = extract_candidates(digest)
            written: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []
            for candidate in candidates:
                mem_candidate = _review_candidate(candidate, identity)
                decision = decide_admission(memory, mem_candidate)
                entry = {**candidate.to_dict(), "admit": decision.to_dict()}
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
                )
                entry.update(_node_summary(memory.nodes[node_id]))
                written.append(entry)
            closed = 0
            if close and parsed_mode != ReviewMode.CANDIDATES:
                for node in episodes:
                    memory.set_status(node.id, MemoryStatus.STALE, f"reviewed in {parsed_mode.value} mode")
                    closed += 1
            return {
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
            return self.memory.read(run)
        return self.memory.run(run)

    def approve_memory(self, node_id: str, reason: str = "approved by reviewer") -> dict[str, Any]:
        """Move a ``pending`` review memory into normal recall."""
        return self._resolve_pending(node_id, MemoryStatus.ACTIVE, reason)

    def reject_memory(self, node_id: str, reason: str = "rejected by reviewer") -> dict[str, Any]:
        """Retire a ``pending`` review memory. It keeps its provenance as ``stale``."""
        return self._resolve_pending(node_id, MemoryStatus.STALE, reason)

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

    def _resolve_pending(self, node_id: str, status: MemoryStatus, reason: str) -> dict[str, Any]:
        _require_text(node_id, "node_id")
        current = self.memory.inspect(node_id)
        if current["status"] != MemoryStatus.PENDING.value:
            raise ValueError(f"node {node_id} is {current['status']}, not pending")
        return self.memory.set_status(node_id, status.value, reason)

    def _resolve_goal(
        self,
        objective: str | None,
        goal_id: str | None,
        state: BrainState | None,
        *,
        allow_missing: bool,
    ):
        if goal_id and goal_id.strip():
            inspected = self.memory.inspect(goal_id)
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
                inspected = self.memory.inspect(hit["node_id"])
                parsed = parse_goal_document(inspected["node_id"], inspected["content"], inspected["title"])
                if parsed.status != GoalStatus.COMPLETE:
                    return parsed
            if allow_missing:
                return None
            raise ValueError("no active goal matches the objective")

        if allow_missing:
            return None
        raise ValueError("plan needs an objective or a goal_id")

    def _refuted_hits(self, objective: str, identity: dict[str, str | None]) -> list[dict[str, Any]]:
        found = self.memory.search(
            objective,
            top_k=5,
            statuses=[MemoryStatus.REFUTED.value],
            **_identity_kwargs(identity),
        )
        return list(found["results"])


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
    nodes = [
        node
        for node in memory.nodes.values()
        if is_goal_node({"category": node.category, "tags": node.tags})
        and node.session_id == identity["session_id"]
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


def _identity_kwargs(identity: dict[str, str | None]) -> dict[str, str | None]:
    return {key: value for key, value in identity.items() if key != "task_id"}


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
