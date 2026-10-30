"""Deterministic Goal / Plan / Capability helpers for FibBrain.

Planning here is not an LLM planner and not a harness Goal service. It
persists an explicit objective, turns recalled evidence into avoid/reuse
steps, and names which capabilities the body should involve next.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fibmind.models import new_id

GOAL_CATEGORY = "goal"
GOAL_TAG = "fibbrain-goal"
GOAL_KIND = "fibbrain_goal"

# Capabilities the brain can name. The harness owns execution.
CAPABILITIES: dict[str, str] = {
    "memory": "Recall, remember, and reflect against the store",
    "code": "Inspect or change source",
    "test": "Run verification commands",
    "search": "Look up missing information",
    "docs": "Update documentation",
    "review": "Inspect evidence or diffs",
    "ask": "Ask the user for intent or missing requirements",
}

# Suggested actions used when asking ``advise`` about a capability.
CAPABILITY_ACTIONS: dict[str, tuple[str, str]] = {
    "memory": ("brain", "recall"),
    "code": ("tool", "edit_file"),
    "test": ("tool", "pytest"),
    "search": ("tool", "web_search"),
    "docs": ("tool", "edit_file"),
    "review": ("tool", "read_diff"),
    "ask": ("tool", "ask_user"),
}

_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("test", "pytest", "verify", "验收", "单测"), "test"),
    (("fix", "bug", "implement", "refactor", "修", "实现", "改代码"), "code"),
    (("doc", "readme", "文档"), "docs"),
    (("search", "look up", "查"), "search"),
)


class GoalStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETE = "complete"


class StepStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class PlanStep:
    title: str
    capabilities: tuple[str, ...]
    id: str = ""
    status: StepStatus = StepStatus.PENDING
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", new_id("step"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "capabilities": list(self.capabilities),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        return cls(
            id=str(data.get("id") or new_id("step")),
            title=str(data["title"]),
            status=StepStatus(data.get("status", StepStatus.PENDING.value)),
            capabilities=tuple(data.get("capabilities") or ()),
            reason=str(data.get("reason") or ""),
        )


@dataclass(frozen=True, slots=True)
class BrainGoal:
    goal_id: str
    objective: str
    status: GoalStatus
    steps: tuple[PlanStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "objective": self.objective,
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
        }

    def document(self) -> dict[str, Any]:
        return {
            "kind": GOAL_KIND,
            "objective": self.objective,
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
        }


def infer_capabilities(objective: str) -> list[str]:
    """Name the work capabilities implied by ``objective``. Memory is added later."""
    text = objective.casefold()
    found: list[str] = []
    for needles, capability in _HINTS:
        if capability not in found and any(needle in text for needle in needles):
            found.append(capability)
    if not found:
        found.append("review")
    return found


def synthesize_plan(
    objective: str,
    memories: Iterable[dict[str, Any]] | None = None,
    status: GoalStatus = GoalStatus.ACTIVE,
    goal_id: str = "",
    assessment: dict[str, Any] | None = None,
) -> BrainGoal:
    """Build a deterministic plan from the objective plus recalled evidence and optional assessment."""
    steps: list[PlanStep] = [
        PlanStep(
            title="Recall related history",
            capabilities=("memory",),
            reason="Load what this working context already knows before acting",
        )
    ]
    work = infer_capabilities(objective)

    if assessment is not None:
        avoid_count = 0
        reuse_count = 0
        # 1. Avoid: <title>
        for item in assessment.get("contradicted", []):
            title = str(item.get("title") or item.get("node_id") or "").strip()
            if title and avoid_count < 3:
                step_title = f"Avoid: {title}"
                if not any(s.title == step_title for s in steps):
                    steps.append(
                        PlanStep(
                            title=step_title,
                            capabilities=("memory", "review"),
                            reason="A stored memory for this action was later refuted",
                        )
                    )
                    avoid_count += 1
        if avoid_count < 3 and memories:
            for memory in memories:
                if str(memory.get("status") or "") == "refuted":
                    title = str(memory.get("title") or "").strip()
                    step_title = f"Avoid: {title}"
                    if title and not any(s.title == step_title for s in steps):
                        steps.append(
                            PlanStep(
                                title=step_title,
                                capabilities=("memory", "review"),
                                reason="A stored memory for this action was later refuted",
                            )
                        )
                        avoid_count += 1
                        if avoid_count >= 3:
                            break

        # 2. Reuse: <title>
        for item in assessment.get("known", []):
            title = str(item.get("title") or "").strip()
            if title and reuse_count < 3:
                step_title = f"Reuse: {title}"
                if not any(s.title == step_title for s in steps):
                    steps.append(
                        PlanStep(
                            title=step_title,
                            capabilities=("memory",),
                            reason="Confirmed evidence already exists for this goal",
                        )
                    )
                    reuse_count += 1

        # 3. Verify: <target>
        for item in assessment.get("uncertain", []):
            route = item.get("route")
            target = item.get("target") or item.get("title") or ""
            if route == "verify" and target:
                step_title = f"Verify: {target}"
                if not any(s.title == step_title for s in steps):
                    steps.append(
                        PlanStep(
                            title=step_title,
                            capabilities=("test",),
                            reason=f"Verify uncertain memory: {item.get('title', target)}",
                        )
                    )
            elif route == "code" and target:
                step_title = f"Verify: {target}"
                if not any(s.title == step_title for s in steps):
                    steps.append(
                        PlanStep(
                            title=step_title,
                            capabilities=("code",),
                            reason=f"Inspect code for uncertain memory: {item.get('title', target)}",
                        )
                    )

        # 4. Resolve: <title>
        for item in assessment.get("uncertain", []):
            if item.get("route") == "resolve":
                title = str(item.get("title") or item.get("node_id") or "").strip()
                if title:
                    step_title = f"Resolve: {title}"
                    if not any(s.title == step_title for s in steps):
                        steps.append(
                            PlanStep(
                                title=step_title,
                                capabilities=("memory", "review"),
                                reason=f"Resolve dialectical conflict: {title}",
                            )
                        )

        # 5. Investigate: <aspect> via <route>
        for item in assessment.get("unknown", []):
            route = item.get("route")
            aspect = str(item.get("aspect") or "").strip()
            if route in ("code", "search") and aspect:
                step_title = f"Investigate: {aspect} via {route}"
                if not any(s.title == step_title for s in steps):
                    cap = "code" if route == "code" else "search"
                    steps.append(
                        PlanStep(
                            title=step_title,
                            capabilities=(cap,),
                            reason=f"Investigate unknown aspect via {route}",
                        )
                    )

        # 6. Ask: <aspect>
        for item in assessment.get("uncertain", []):
            if item.get("route") == "ask":
                title = str(item.get("title") or item.get("target") or "user").strip()
                step_title = f"Ask: {title}"
                if not any(s.title == step_title for s in steps):
                    steps.append(
                        PlanStep(
                            title=step_title,
                            capabilities=("ask",),
                            reason="Ask the user for clarification",
                        )
                    )
        for item in assessment.get("unknown", []):
            if item.get("route") == "ask":
                aspect = str(item.get("aspect") or "").strip()
                if aspect:
                    step_title = f"Ask: {aspect}"
                    if not any(s.title == step_title for s in steps):
                        steps.append(
                            PlanStep(
                                title=step_title,
                                capabilities=("ask",),
                                reason="Ask the user for missing intent or requirements",
                            )
                        )
    else:
        avoid_count = 0
        reuse_count = 0
        for memory in memories or ():
            title = str(memory.get("title") or "").strip()
            if not title:
                continue
            memory_status = str(memory.get("status") or "")
            confidence = float(memory.get("confidence") or 0.0)
            if memory_status == "refuted" and avoid_count < 3:
                steps.append(
                    PlanStep(
                        title=f"Avoid: {title}",
                        capabilities=("memory", "review"),
                        reason="A stored memory for this action was later refuted",
                    )
                )
                avoid_count += 1
            elif confidence > 0 and memory_status != "refuted" and reuse_count < 3:
                steps.append(
                    PlanStep(
                        title=f"Reuse: {title}",
                        capabilities=("memory",),
                        reason="Confirmed evidence already exists for this goal",
                    )
                )
                reuse_count += 1

    steps.append(
        PlanStep(
            title=f"Do the work: {objective.strip()}",
            capabilities=tuple(work),
            reason="Capabilities inferred from the objective",
        )
    )
    verify = ("test",) if "test" in work else ("review",)
    steps.append(
        PlanStep(
            title="Verify the result",
            capabilities=verify,
            reason="Check the work against something outside the model",
        )
    )
    steps.append(
        PlanStep(
            title="Reflect and remember",
            capabilities=("memory",),
            reason="Record whether prior memories held, then write only what is worth keeping",
        )
    )
    return BrainGoal(
        goal_id=goal_id,
        objective=objective.strip(),
        status=status,
        steps=tuple(steps),
    )


def parse_goal_document(node_id: str, content: str, title: str) -> BrainGoal:
    """Read a persisted goal node back into a ``BrainGoal``."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict) or data.get("kind") != GOAL_KIND:
        return BrainGoal(
            goal_id=node_id,
            objective=title,
            status=GoalStatus.ACTIVE,
            steps=(),
        )
    steps = tuple(PlanStep.from_dict(item) for item in data.get("steps") or () if isinstance(item, dict))
    status_value = data.get("status", GoalStatus.ACTIVE.value)
    try:
        status = GoalStatus(status_value)
    except ValueError:
        status = GoalStatus.ACTIVE
    return BrainGoal(
        goal_id=node_id,
        objective=str(data.get("objective") or title),
        status=status,
        steps=steps,
    )


def encode_goal(goal: BrainGoal) -> str:
    return json.dumps(goal.document(), ensure_ascii=False, indent=2)


def advance_goal_steps(goal: BrainGoal, event_kind: str, target: str = "") -> BrainGoal:
    """Advance PlanStep state machine in response to cognitive events."""
    steps = list(goal.steps)

    def _set_status(step: PlanStep, new_status: StepStatus) -> PlanStep:
        return PlanStep(
            id=step.id,
            title=step.title,
            status=new_status,
            capabilities=step.capabilities,
            reason=step.reason,
        )

    if event_kind in ("plan", "coordinate"):
        for i, step in enumerate(steps):
            if step.title == "Recall related history" or step.title.startswith("Reuse:"):
                steps[i] = _set_status(step, StepStatus.DONE)
        if not any(s.status == StepStatus.ACTIVE for s in steps):
            for i, step in enumerate(steps):
                if step.status == StepStatus.PENDING:
                    steps[i] = _set_status(step, StepStatus.ACTIVE)
                    break

    elif event_kind in ("verify_confirmed", "reflect_confirmed"):
        target_lower = target.lower().strip() if target else ""
        for i, step in enumerate(steps):
            if step.title.startswith("Verify"):
                if (
                    not target_lower
                    or target_lower in step.title.lower()
                    or target_lower in step.reason.lower()
                    or step.title == "Verify the result"
                ):
                    steps[i] = _set_status(step, StepStatus.DONE)

    elif event_kind in ("resolve_conflict", "synthesize_conflict"):
        for i, step in enumerate(steps):
            if step.title.startswith("Resolve:"):
                steps[i] = _set_status(step, StepStatus.DONE)

    elif event_kind == "remember":
        for i, step in enumerate(steps):
            if step.title.startswith("Do the work:") or step.title == "Reflect and remember":
                steps[i] = _set_status(step, StepStatus.DONE)

    elif event_kind == "complete_goal":
        for i, step in enumerate(steps):
            if step.status in (StepStatus.PENDING, StepStatus.ACTIVE):
                steps[i] = _set_status(step, StepStatus.DONE)

    if event_kind != "complete_goal":
        if not any(s.status == StepStatus.ACTIVE for s in steps):
            for i, step in enumerate(steps):
                if step.status == StepStatus.PENDING:
                    steps[i] = _set_status(step, StepStatus.ACTIVE)
                    break

    return BrainGoal(
        goal_id=goal.goal_id,
        objective=goal.objective,
        status=GoalStatus.COMPLETE if event_kind == "complete_goal" else goal.status,
        steps=tuple(steps),
    )


def is_goal_node(summary: dict[str, Any]) -> bool:
    tags = summary.get("tags") or []
    return summary.get("category") == GOAL_CATEGORY and GOAL_TAG in tags
