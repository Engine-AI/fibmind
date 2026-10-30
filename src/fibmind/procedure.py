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
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fibmind.models import MemoryKind, MemoryNode, MemoryStatus, Verdict

if TYPE_CHECKING:
    from fibmind.graph import FibMind

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
    def from_document(cls, data: dict[str, Any]) -> Procedure:
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
    def parse(cls, content: str) -> Procedure | None:
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
        f'Report the outcome with `fibbrain_reflect(node_id="{node.id}", ...)`.'
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


_MATURITY_ORDER = {"candidate": 0, "verified": 1, "established": 2}
_PROVENANCE_RE = re.compile(r"FibBrain procedure `(?P<node_id>[a-zA-Z0-9_\-]+)`")


def _maturity(outcomes: dict[str, int]) -> str:
    if promotion_ready(outcomes):
        return "established"
    if outcomes.get("confirmed", 0) > 0:
        return "verified"
    return "candidate"


def promotion_ready(outcomes: dict[str, int]) -> bool:
    return outcomes.get("confirmed", 0) >= PROMOTE_MIN_CONFIRMED and outcomes.get("sessions", 0) >= PROMOTE_MIN_SESSIONS


def retirement_due(outcomes: dict[str, int]) -> bool:
    return outcomes.get("refuted", 0) >= RETIRE_MIN_REFUTED and outcomes.get("confirmed", 0) == 0


def parse_skill_markdown(text: str) -> dict[str, Any] | None:
    """Parse a SKILL.md markdown document back into title, node_id, and Procedure."""
    if not text or not str(text).strip():
        return None

    match_prov = _PROVENANCE_RE.search(text)
    node_id = match_prov.group("node_id") if match_prov else None

    when: list[str] = []
    body = text
    if text.startswith("---"):
        end_fm = text.find("\n---", 3)
        if end_fm != -1:
            fm_text = text[3:end_fm]
            body = text[end_fm + 4 :].lstrip()
            for line in fm_text.splitlines():
                line_s = line.strip()
                if line_s.startswith("description:"):
                    desc_val = line_s.split("description:", 1)[1].strip().strip("\"'")
                    if "Use when:" in desc_val:
                        cues_part = desc_val.split("Use when:", 1)[1].strip().rstrip(".")
                        for cue in cues_part.split(","):
                            c = cue.strip()
                            if c:
                                when.append(c)

    lines = body.splitlines()
    title = ""
    h1_idx = -1
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith("# ") and not s.startswith("## "):
            title = s[2:].strip()
            h1_idx = idx
            break

    if not title or h1_idx == -1:
        return None

    trigger_lines: list[str] = []
    cur_idx = h1_idx + 1
    while cur_idx < len(lines):
        line = lines[cur_idx]
        if line.strip().startswith("## "):
            break
        trigger_lines.append(line)
        cur_idx += 1

    trigger = "\n".join(trigger_lines).strip()
    if not trigger:
        return None

    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    while cur_idx < len(lines):
        line = lines[cur_idx]
        s = line.strip()
        if s.startswith("## "):
            current_section = s[3:].strip().lower()
            sections[current_section] = []
        elif current_section is not None:
            sections[current_section].append(line)
        cur_idx += 1

    inputs: dict[str, str] = {}
    for line in sections.get("inputs", []):
        m = re.match(r"^-\s*`([^`]+)`:\s*(.*)$", line.strip())
        if m:
            inputs[m.group(1).strip()] = m.group(2).strip()

    steps: list[str] = []
    for line in sections.get("steps", []):
        s = line.strip()
        m_num = re.match(r"^\d+\.\s*(.+)$", s)
        if m_num:
            steps.append(m_num.group(1).strip())
        elif s.startswith("- "):
            steps.append(s[2:].strip())
        elif s.startswith("* "):
            steps.append(s[2:].strip())

    if not steps:
        return None

    tools: list[str] = []
    for line in sections.get("tools", []):
        s = line.strip()
        if s.startswith("- ") or s.startswith("* "):
            tools.append(s[2:].strip())

    verify_lines = sections.get("verify", [])
    verify = "\n".join(verify_lines).strip()

    try:
        proc = Procedure(
            trigger=trigger,
            steps=tuple(steps),
            when=tuple(when),
            tools=tuple(tools),
            verify=verify,
            inputs=inputs,
        )
    except ValueError:
        return None

    return {
        "node_id": node_id,
        "title": title,
        "procedure": proc,
    }


def sync_skills(
    memory: FibMind,
    target_dir: str | Path,
    *,
    identity: dict[str, str | None],
    min_maturity: str = "verified",
    cwd: str | Path | None = None,
    host: str = "generic",
    write_manifest: bool = True,
    ingest_modified: bool = False,
) -> dict[str, Any]:
    """Materialize mature procedures as SKILL.md and auto-revoke refuted/drifted ones."""
    from fibmind.reflex import check_node_preconditions

    allowed_hosts = {"generic", "agy", "claude-code", "codex"}
    if host not in allowed_hosts:
        raise ValueError(f"host must be one of {sorted(allowed_hosts)}, got {host!r}")

    if min_maturity not in _MATURITY_ORDER:
        raise ValueError(f"min_maturity must be one of {sorted(_MATURITY_ORDER)}")

    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    ingested: list[dict[str, Any]] = []
    if ingest_modified:
        for skill_file in sorted(target_path.glob("*/SKILL.md")):
            try:
                content = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue

            match = _PROVENANCE_RE.search(content)
            if not match:
                continue

            file_node_id = match.group("node_id")
            node = memory.nodes.get(file_node_id)
            if node is None or node.status != MemoryStatus.ACTIVE:
                continue

            parsed = parse_skill_markdown(content)
            if parsed is None:
                continue

            existing_proc = Procedure.parse(node.content)
            new_proc = parsed["procedure"]
            if existing_proc is not None and not new_proc.when and existing_proc.when:
                new_proc = Procedure(
                    trigger=new_proc.trigger,
                    steps=new_proc.steps,
                    when=existing_proc.when,
                    tools=new_proc.tools,
                    verify=new_proc.verify,
                    inputs=new_proc.inputs,
                )

            if (
                existing_proc is None
                or new_proc.document() != existing_proc.document()
                or parsed["title"] != node.title
            ):
                prev_outcomes = memory.outcome_counts(node.id)
                prev_conf = node.confidence
                memory.revise(node.id, title=parsed["title"], content=new_proc.encode())
                if prev_outcomes.get("confirmed", 0) > 0 or prev_conf > 0:
                    memory.record_outcome(
                        node.id,
                        Verdict.CONFIRMED,
                        source="skill_ingest",
                        session_id=identity.get("session_id"),
                        trust_ring="ring0",
                    )
                ingested.append({"node_id": node.id, "title": node.title, "path": str(skill_file)})

    eligible_nodes: dict[str, tuple[MemoryNode, Procedure, dict[str, int], str]] = {}
    for node in memory.nodes.values():
        if node.status != MemoryStatus.ACTIVE or node.folded_into is not None:
            continue
        if not is_procedure_node(node) and node.category != PROCEDURE_CATEGORY:
            continue
        if not memory.is_visible(
            node,
            owner=identity.get("owner"),
            statuses={MemoryStatus.ACTIVE},
            workspace_id=identity.get("workspace_id"),
            project_id=identity.get("project_id"),
            session_id=identity.get("session_id"),
            task_id=identity.get("task_id"),
        ):
            continue

        outcomes = memory.outcome_counts(node.id)
        mat = _maturity(outcomes)
        if _MATURITY_ORDER.get(mat, 0) < _MATURITY_ORDER[min_maturity]:
            continue

        ok, _ = check_node_preconditions(node, cwd=cwd)
        if not ok:
            continue

        proc = Procedure.parse(node.content)
        if proc is None:
            continue

        eligible_nodes[node.id] = (node, proc, outcomes, mat)

    # 1. Materialize eligible procedures
    materialized: list[dict[str, Any]] = []
    manifest_skills: list[dict[str, Any]] = []
    for _node_id, (node, proc, outcomes, mat) in eligible_nodes.items():
        rendered = render_skill(node, proc, outcomes)
        skill_file = target_path / rendered["path"]
        skill_file.parent.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(rendered["text"], encoding="utf-8")
        materialized.append(
            {
                "node_id": node.id,
                "title": node.title,
                "path": str(skill_file),
                "maturity": mat,
            }
        )
        manifest_skills.append(
            {
                "node_id": node.id,
                "name": rendered["name"],
                "title": node.title,
                "trigger": proc.trigger,
                "maturity": mat,
                "verify": proc.verify,
                "path": str(skill_file),
            }
        )

    # 2. Auto-revoke scanning for deleted, refuted, or drifted FibBrain procedures
    revoked: list[dict[str, Any]] = []
    for skill_file in sorted(target_path.glob("*/SKILL.md")):
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError:
            continue

        match = _PROVENANCE_RE.search(content)
        if not match:
            # User hand-written skill: do not touch
            continue

        file_node_id = match.group("node_id")
        if file_node_id not in eligible_nodes:
            try:
                skill_file.unlink()
                parent_dir = skill_file.parent
                if parent_dir != target_path and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
                revoked.append(
                    {
                        "node_id": file_node_id,
                        "path": str(skill_file),
                        "reason": "ineligible_or_refuted",
                    }
                )
            except OSError:
                pass

    manifest_path = target_path / "fibbrain-skills.manifest.json"
    if write_manifest:
        manifest_payload = {
            "host": host,
            "min_maturity": min_maturity,
            "skills": manifest_skills,
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return {
        "materialized": materialized,
        "revoked": revoked,
        "target_dir": str(target_path),
        "host": host,
        "manifest_path": str(manifest_path) if write_manifest else None,
        "ingested": ingested,
    }


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
