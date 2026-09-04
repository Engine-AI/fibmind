"""Deterministic Goal / Plan / Capability helpers for FibBrain.

Planning here is not an LLM planner and not a harness Goal service. It
persists an explicit objective, turns recalled evidence into avoid/reuse
steps, and names which capabilities the body should involve next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable

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
}

# Suggested actions used when asking ``advise`` about a capability.
CAPABILITY_ACTIONS: dict[str, tuple[str, str]] = {
    "memory": ("brain", "recall"),
    "code": ("tool", "edit_file"),
    "test": ("tool", "pytest"),
    "search": ("tool", "web_search"),
    "docs": ("tool", "edit_file"),
    "review": ("tool", "read_diff"),
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
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
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
) -> BrainGoal:
    """Build a deterministic plan from the objective plus recalled evidence."""
    steps: list[PlanStep] = [
        PlanStep(
            title="Recall related history",
            capabilities=("memory",),
            reason="Load what this working context already knows before acting",
        )
    ]
    work = infer_capabilities(objective)
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


def is_goal_node(summary: dict[str, Any]) -> bool:
    tags = summary.get("tags") or []
    return summary.get("category") == GOAL_CATEGORY and GOAL_TAG in tags
