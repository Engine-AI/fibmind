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
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from fibmind.admission import AdmitVerdict, MemoryCandidate, decide_admission
from fibmind.distill import DEFAULT_PLANNER, Planner
from fibmind.graph import FibMind
from fibmind.models import MemoryKind, MemoryNode, MemoryScope, MemoryStatus, RelationType, Verdict, optional_id, utc_now
import json
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


class AdviseVerdict(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"


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
HOT_KINDS = ("preference", "requirement", "decision", "knowledge")


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

    def __init__(self, service: MemoryService, planner: Planner = DEFAULT_PLANNER) -> None:
        self.memory = service
        self.planner = planner
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
        budget_tokens: int | None = None,
        include_hot: bool = True,
    ) -> dict[str, Any]:
        """Load a context pack for ``goal`` inside ``state`` and a token budget.

        The pack renders the session's frozen hot snapshot first, then direct
        hits, then graph expansion, and reports ``budget`` usage per section.
        ``budget_tokens`` defaults to 500 (roughly ``max_chars / 4``).
        """
        identity = (state or BrainState()).identity()
        budget = budget_tokens if budget_tokens is not None else DEFAULT_BUDGET_TOKENS
        hot_ids = self.hot_snapshot(state)["node_ids"] if include_hot else []
        pack = self.memory.context(
            goal,
            top_k=top_k,
            depth=depth,
            max_chars=max_chars,
            exclude_categories=[EPISODE_CATEGORY, HOT_CATEGORY],
            budget_tokens=budget,
            hot_node_ids=hot_ids,
            hot_budget_tokens=min(HOT_BUDGET_TOKENS, budget // 3) if hot_ids else 0,
            **_identity_kwargs(identity),
        )
        pack["state"] = identity
        pack["episode"] = [event.to_dict() for event in self._episode_for(identity)]
        return pack

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
        session_id = identity["session_id"]

        def run(memory: FibMind) -> dict[str, Any]:
            existing = None
            if session_id is not None:
                for node in memory.nodes.values():
                    if (
                        node.category == HOT_CATEGORY
                        and HOT_TAG in node.tags
                        and node.session_id == session_id
                        and node.status == MemoryStatus.ACTIVE
                        and memory.is_visible(node, **_identity_kwargs(identity))
                    ):
                        existing = node
                        break
            if existing is not None and not refresh:
                data = json.loads(existing.content)
                live = [nid for nid in data.get("node_ids", []) if nid in memory.nodes]
                return {"node_ids": live, "frozen": True, "snapshot_id": existing.id, "session_id": session_id}

            selected = _select_hot(memory, identity)
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
            return {"node_ids": node_ids, "frozen": session_id is not None, "snapshot_id": snapshot_id, "session_id": session_id}

        if session_id is None:
            return self.memory.read(run)
        return self.memory.run(run)

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
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Should this action run, given what we already know?

        A refuted memory blocks the action only when it matches the action
        name itself — not merely a word from the arguments. When ``arguments``
        are given, the blocker must also share at least one argument term, so
        evidence that ``rm build/`` failed does not block ``rm tmp/``; without
        arguments, any refuted memory about the action blocks it.
        """
        _require_text(action, "action")
        identity = (state or BrainState()).identity()
        action_terms = set(tokenize(action))
        argument_terms = set(tokenize(_flatten_arguments(arguments))) - action_terms if arguments else set()
        query = " ".join([kind, action, *sorted(argument_terms)])
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
            if not matched & action_terms:
                continue
            if argument_terms and not matched & argument_terms:
                continue
            blockers.append(hit)
        if blockers:
            decision = AdviseDecision(
                AdviseVerdict.REJECT,
                f"refuted evidence advises against {action}"
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
                    self.planner.plan(resolved_objective, memories=evidence)
                ),
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
            outcomes = self.memory.read(lambda memory, nid=row["node_id"]: memory.outcome_counts(nid))
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
        identity = (state or BrainState()).identity()
        updated = self.memory.record_outcome(
            target, parsed.value, source, note=note, session_id=identity["session_id"]
        )
        if updated.get("memory_kind") == MemoryKind.PROCEDURE.value:
            updated["evolution"] = self._evolve_procedure(target)
        self.observe(
            "reflect",
            f"{parsed.value} {updated['title']} via {source}",
            {"node_id": target, "verdict": parsed.value, "source": source},
            state=state,
        )
        return updated

    def _evolve_procedure(self, node_id: str) -> dict[str, Any]:
        """Apply the deterministic evolution rules to one procedure.

        Retirement is automatic: refuted twice with no confirmation means the
        procedure does not work here. Promotion into shared knowledge is only
        *reported*: it still goes through ``promote_knowledge`` and its
        three-supporter bar, which one node cannot satisfy by itself.
        """
        outcomes = self.memory.read(lambda memory: memory.outcome_counts(node_id))
        retired = False
        if retirement_due(outcomes):
            current = self.memory.inspect(node_id)
            if current["status"] == MemoryStatus.ACTIVE.value:
                self.memory.mark_stale(
                    node_id,
                    f"procedure refuted {outcomes['refuted']}× with no confirmation",
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
        state: BrainState | None = None,
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
        result = self.remember(
            PROCEDURE_CATEGORY,
            _require_text(title, "title"),
            procedure.encode(),
            tags=[PROCEDURE_TAG],
            metadata={"kind": "fibbrain_procedure_node", "trigger": procedure.trigger},
            state=state,
        )
        if result.get("verdict") == AdmitVerdict.WRITE.value:
            for old_id in previous:
                self.memory.link(result["node_id"], old_id, RelationType.VERSION_OF.value, weight=1.0)
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
            outcomes = self.memory.read(lambda memory, nid=row["node_id"]: memory.outcome_counts(nid))
            maturity = _maturity(outcomes)
            if _MATURITY_ORDER[maturity] < _MATURITY_ORDER[min_maturity]:
                continue
            node = self.memory.read(lambda memory, nid=row["node_id"]: memory.nodes[nid])
            item = render_skill(node, procedure, outcomes) if format == "skill" else render_tool(node, procedure, outcomes)
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
        """Active procedure nodes visible to this identity, best match first."""
        kwargs = _identity_kwargs(identity)
        if query:
            found = self.memory.search(query, top_k=top_k, categories=[PROCEDURE_CATEGORY], **kwargs)
            rows = list(found["results"])
        else:
            rows = self.memory.read(
                lambda memory: [
                    {**_node_summary(node), "score": 0.0}
                    for node in memory.nodes.values()
                    if is_procedure_node(node) and memory.is_visible(node, **kwargs)
                ]
            )
        out: list[dict[str, Any]] = []
        for row in rows:
            inspected = self.memory.inspect(row["node_id"])
            if not is_procedure_node(_node_from_summary(inspected)):
                continue
            out.append({**row, "content": inspected["content"]})
        out.sort(key=lambda row: (row.get("score", 0.0), row.get("confidence", 0.0)), reverse=True)
        return out

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

    def revalidation_candidates(self, state: BrainState | None = None, older_than_days: int = 90) -> list[dict[str, Any]]:
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
    """A light node view for ``is_procedure_node``; only kind and tags matter."""
    node = MemoryNode(title=inspected["title"], content=inspected.get("content", ""), category=inspected["category"])
    node.tags = set(inspected.get("tags") or [])
    kind = inspected.get("memory_kind")
    node.memory_kind = MemoryKind(kind) if kind else None
    return node


def _last_confirmation(memory: FibMind, node_id: str) -> datetime | None:
    last = None
    for event in memory.events:
        if event.op.value != "observe" or event.payload.get("node_id") != node_id:
            continue
        if event.payload.get("verdict") == Verdict.CONFIRMED.value:
            last = event.created_at if last is None or event.created_at > last else last
    return last


def _select_hot(memory: FibMind, identity: dict[str, str | None]) -> list[MemoryNode]:
    """Deterministic L0 selection: hot kinds, visible, active, evidence first."""
    from fibmind.tokens import DEFAULT_COUNTER

    kwargs = _identity_kwargs(identity)
    candidates = [
        node
        for node in memory.nodes.values()
        if node.memory_kind is not None
        and node.memory_kind.value in HOT_KINDS
        and node.node_type.value == "raw"
        and node.category not in {EPISODE_CATEGORY, HOT_CATEGORY, GOAL_CATEGORY}
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
