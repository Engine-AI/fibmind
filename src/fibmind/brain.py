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

from fibmind.admission import MemoryCandidate
from fibmind.models import MemoryScope, MemoryStatus, Verdict, optional_id, utc_now
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
from fibmind.service import MemoryService, _require_text


class AdviseVerdict(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"


EPISODE_LIMIT = 48
GOAL_TITLE_LIMIT = 200


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
        }


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

    def observe(self, kind: str, summary: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a short-lived episode event. Not persisted as long-term memory."""
        event = ObservedEvent(
            kind=_require_text(kind, "kind"),
            summary=_require_text(summary, "summary"),
            payload=dict(payload or {}),
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
            **_identity_kwargs(identity),
        )
        pack["state"] = identity
        pack["episode"] = [event.to_dict() for event in self._episode]
        return pack

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
        self.observe("complete_goal", f"completed {goal.objective}", {"goal_id": goal.goal_id})
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
        )
        return updated

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
