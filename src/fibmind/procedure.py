"""Procedural memory: a repeatable way of doing a class of task.

Declarative memories say what is true. A procedure says what to *do*: when it
applies, the steps, the tools it needs, and how to tell it worked. It is stored
like any other node (``memory_kind=procedure``) with a structured JSON body, so
it is governed by the same identity, status, and evidence rules — and it is
the only kind ``render`` can turn into a skill or a tool definition.

Evolution is deterministic and outcome-driven:

- confirmed by ``PROMOTE_MIN_CONFIRMED`` reflections across at least
  ``PROMOTE_MIN_SESSIONS`` distinct sessions → eligible for promotion into
  shared knowledge;
- refuted ``RETIRE_MIN_REFUTED`` times with no confirmation → retired as stale.

Nothing here executes a procedure. The harness does that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from fibmind.models import MemoryKind, MemoryNode

PROCEDURE_KIND = "fibbrain_procedure"
PROCEDURE_CATEGORY = "procedure"
PROCEDURE_TAG = "fibbrain-procedure"

PROMOTE_MIN_CONFIRMED = 3
PROMOTE_MIN_SESSIONS = 2
RETIRE_MIN_REFUTED = 2


@dataclass(frozen=True, slots=True)
class Procedure:
    """The structured body of a procedure node."""

    trigger: str
    steps: tuple[str, ...]
    when: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    verify: str = ""
    inputs: dict[str, str] = field(default_factory=dict)

    def document(self) -> dict[str, Any]:
        return {
            "kind": PROCEDURE_KIND,
            "trigger": self.trigger,
            "when": list(self.when),
            "steps": list(self.steps),
            "tools": list(self.tools),
            "verify": self.verify,
            "inputs": dict(self.inputs),
        }

    def encode(self) -> str:
        return json.dumps(self.document(), ensure_ascii=False, indent=2)

    def search_text(self) -> str:
        """What keyword search should see: trigger, cues, tools, steps."""
        return " ".join([self.trigger, *self.when, *self.tools, *self.steps, self.verify])

    @classmethod
    def from_document(cls, data: dict[str, Any]) -> "Procedure":
        if data.get("kind") != PROCEDURE_KIND:
            raise ValueError("not a FibBrain procedure document")
        trigger = str(data.get("trigger") or "").strip()
        steps = tuple(str(step).strip() for step in data.get("steps") or () if str(step).strip())
        if not trigger:
            raise ValueError("procedure needs a trigger")
        if not steps:
            raise ValueError("procedure needs at least one step")
        inputs = data.get("inputs") or {}
        return cls(
            trigger=trigger,
            steps=steps,
            when=tuple(str(item).strip() for item in data.get("when") or () if str(item).strip()),
            tools=tuple(str(item).strip() for item in data.get("tools") or () if str(item).strip()),
            verify=str(data.get("verify") or "").strip(),
            inputs={str(k): str(v) for k, v in inputs.items()} if isinstance(inputs, dict) else {},
        )

    @classmethod
    def parse(cls, content: str) -> "Procedure | None":
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        try:
            return cls.from_document(data)
        except ValueError:
            return None


def is_procedure_node(node: MemoryNode) -> bool:
    return node.memory_kind == MemoryKind.PROCEDURE or PROCEDURE_TAG in node.tags


def slug(text: str, limit: int = 48) -> str:
    """A filesystem- and tool-name-safe identifier."""
    cleaned = re.sub(r"[^a-z0-9一-鿿]+", "-", text.casefold()).strip("-")
    return (cleaned or "procedure")[:limit].rstrip("-")


# --------------------------------------------------------------------------
# Rendering


def render_skill(node: MemoryNode, procedure: Procedure, outcomes: dict[str, int]) -> dict[str, Any]:
    """A SKILL.md-shaped document: frontmatter plus the steps."""
    name = slug(node.title)
    description = procedure.trigger
    if procedure.when:
        description += " Use when: " + ", ".join(procedure.when) + "."
    lines = [
        "---",
        f"name: {name}",
        f"description: {_yaml_scalar(description)}",
        "---",
        "",
        f"# {node.title}",
        "",
        procedure.trigger,
        "",
    ]
    if procedure.inputs:
        lines.append("## Inputs")
        lines.append("")
        for key, meaning in procedure.inputs.items():
            lines.append(f"- `{key}`: {meaning}")
        lines.append("")
    lines.append("## Steps")
    lines.append("")
    for index, step in enumerate(procedure.steps, start=1):
        lines.append(f"{index}. {step}")
    lines.append("")
    if procedure.tools:
        lines.append("## Tools")
        lines.append("")
        lines.extend(f"- {tool}" for tool in procedure.tools)
        lines.append("")
    if procedure.verify:
        lines.append("## Verify")
        lines.append("")
        lines.append(procedure.verify)
        lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        f"FibBrain procedure `{node.id}` — confirmed {outcomes.get('confirmed', 0)}×, "
        f"refuted {outcomes.get('refuted', 0)}×, across {outcomes.get('sessions', 0)} session(s). "
        f"Report the outcome with `fibbrain_reflect(node_id=\"{node.id}\", ...)`."
    )
    return {
        "format": "skill",
        "name": name,
        "path": f"{name}/SKILL.md",
        "text": "\n".join(lines) + "\n",
    }


def render_tool(node: MemoryNode, procedure: Procedure, outcomes: dict[str, int]) -> dict[str, Any]:
    """A tool definition in the JSON-Schema shape MCP and dsh both accept."""
    name = slug(node.title).replace("-", "_")
    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, meaning in procedure.inputs.items():
        properties[key] = {"type": "string", "description": meaning}
        required.append(key)
    description = procedure.trigger
    if procedure.when:
        description += " Use when: " + ", ".join(procedure.when) + "."
    return {
        "format": "tool",
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "steps": list(procedure.steps),
        "tools": list(procedure.tools),
        "verify": procedure.verify,
        "provenance": {"node_id": node.id, **outcomes},
    }


def _yaml_scalar(text: str) -> str:
    """Quote a one-line YAML string safely."""
    return json.dumps(" ".join(text.split()), ensure_ascii=False)


# --------------------------------------------------------------------------
# Evolution rules


def promotion_ready(outcomes: dict[str, int]) -> bool:
    return (
        outcomes.get("confirmed", 0) >= PROMOTE_MIN_CONFIRMED
        and outcomes.get("sessions", 0) >= PROMOTE_MIN_SESSIONS
    )


def retirement_due(outcomes: dict[str, int]) -> bool:
    return outcomes.get("refuted", 0) >= RETIRE_MIN_REFUTED and outcomes.get("confirmed", 0) == 0


# --------------------------------------------------------------------------
# Extraction from a session (used by review)


def procedure_from_session(
    objective: str,
    steps: Iterable[str],
    tools: Iterable[str],
    verify: str,
    when: Iterable[str] = (),
) -> Procedure | None:
    """Build a procedure candidate from what a session did, if there is enough."""
    step_list = tuple(dict.fromkeys(step.strip() for step in steps if step and step.strip()))
    if not objective.strip() or not step_list:
        return None
    return Procedure(
        trigger=objective.strip(),
        steps=step_list,
        tools=tuple(dict.fromkeys(tool.strip() for tool in tools if tool and tool.strip())),
        verify=verify.strip(),
        when=tuple(dict.fromkeys(item.strip() for item in when if item and item.strip())),
    )
