"""Turn a DeepSeek-Harness (dsh) session log into a P0 evaluation candidate.

dsh keeps one append-only JSONL log per session. Every model-visible fact is
an event ``{"seq": n, "type": "...", "time": ms, "data": {...}}``. The two
event types this importer reads:

- ``tool/call``   data: ``{turn, step, callId, name, arguments}`` — ``arguments``
  is the raw JSON string the model produced;
- ``tool/result`` data: ``{turn, step, message: {callId, content: [...]}, ...}``
  — content blocks carry ``{"type": "text", "text": "..."}``.

FibBrain tools appear as ``mcp__<serverName>__fibbrain_*``. A written
``fibbrain_remember`` becomes a memory fixture; a ``fibbrain_recall`` becomes a
case whose ``relevant_ids`` are auto-labelled from what recall returned. The
output is a *candidate*: review the labels before promoting it into
``evals/datasets/``, or the evaluation only measures agreement with itself.

    python -m evals.import_dsh_session path/to/session.jsonl[.zstd] \
        --output evals/candidates/real_dsh.json

The default dsh encoding is concatenated Zstandard frames; this module shells
out to the ``zstd`` CLI for those and reads ``.jsonl`` files directly.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

TOOL_PREFIX = re.compile(r"^(?:mcp__[^_].*?__)?(fibbrain_\w+|fibmind_\w+)$")
IDENTITY_FIELDS = ("owner", "workspace_id", "project_id", "session_id", "task_id")
DEFAULT_DATASET_NAME = "real_dsh"


@dataclass(slots=True)
class ToolExchange:
    """One matched tool call and its result."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    turn: int
    step: int


@dataclass(slots=True)
class ImportReport:
    session_path: str
    memories: list[dict[str, Any]] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    tool_calls_seen: int = 0

    def dataset(self, name: str, description: str) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "source": {"kind": "dsh-session", "path": self.session_path},
            "memories": self.memories,
            "cases": self.cases,
        }


# --------------------------------------------------------------------------
# Reading the log


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read every logical event from a dsh session log, in seq order."""
    log_path = Path(path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    if log_path.suffix in {".zst", ".zstd"}:
        text = _decompress_zstd(log_path)
    else:
        text = log_path.read_text(encoding="utf-8")
    events = list(_parse_lines(text.splitlines()))
    events.sort(key=lambda item: int(item.get("seq", 0)))
    return events


def _decompress_zstd(path: Path) -> str:
    binary = shutil.which("zstd")
    if binary is None:
        raise RuntimeError(
            "reading a compressed dsh log needs the `zstd` CLI (brew install zstd), "
            "or configure dsh-session-persistence-jsonl with compression: 'none'"
        )
    completed = subprocess.run(
        [binary, "-d", "-c", "-q", str(path)],
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8", errors="replace")


def _parse_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue  # torn tail or header shape we do not need
        if not isinstance(item, dict):
            continue
        # The first physical line is the private header; it has no `type`.
        if "type" not in item or "data" not in item:
            continue
        yield item


# --------------------------------------------------------------------------
# Pairing calls with results


def pair_tool_exchanges(events: Iterable[dict[str, Any]]) -> list[ToolExchange]:
    """Match each ``tool/call`` with its ``tool/result`` by ``callId``."""
    calls: dict[str, ToolExchange] = {}
    order: list[str] = []
    for event in events:
        kind = event.get("type")
        data = event.get("data") or {}
        if kind == "tool/call":
            call_id = str(data.get("callId", ""))
            if not call_id:
                continue
            exchange = ToolExchange(
                call_id=call_id,
                name=str(data.get("name", "")),
                arguments=_parse_arguments(data.get("arguments")),
                result=None,
                turn=int(data.get("turn", 0)),
                step=int(data.get("step", 0)),
            )
            calls[call_id] = exchange
            order.append(call_id)
        elif kind == "tool/result":
            message = data.get("message") or {}
            call_id = str(message.get("callId") or data.get("callId") or "")
            exchange = calls.get(call_id)
            if exchange is None:
                continue
            exchange.result = _parse_result(message.get("content"))
    return [calls[call_id] for call_id in order]


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_result(content: Any) -> dict[str, Any] | None:
    """MCP tools return JSON as text blocks; take the first that parses to a dict."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return None
    for block in content:
        text = block.get("text") if isinstance(block, dict) else None
        if not isinstance(text, str):
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def brain_tool_name(name: str) -> str | None:
    """Strip dsh's ``mcp__<server>__`` prefix; return None for non-FibBrain tools."""
    match = TOOL_PREFIX.match(name)
    return match.group(1) if match else None


# --------------------------------------------------------------------------
# Building the dataset


def build_report(
    exchanges: Iterable[ToolExchange],
    session_path: str,
    *,
    default_owner: str | None = None,
) -> ImportReport:
    report = ImportReport(session_path=session_path)
    node_to_fixture: dict[str, str] = {}
    used_ids: set[str] = set()

    for exchange in exchanges:
        tool = brain_tool_name(exchange.name)
        if tool is None:
            continue
        report.tool_calls_seen += 1

        if tool in {"fibbrain_remember", "fibmind_append"}:
            result = exchange.result or {}
            if tool == "fibbrain_remember" and result.get("verdict") != "write":
                report.skipped.append(
                    f"{exchange.call_id}: remember skipped ({result.get('reason', 'no result')})"
                )
                continue
            node_id = str(result.get("node_id") or "")
            if not node_id:
                report.skipped.append(f"{exchange.call_id}: write returned no node_id")
                continue
            fixture_id = _fixture_id(exchange.arguments.get("title", ""), node_id, used_ids)
            node_to_fixture[node_id] = fixture_id
            memory = {
                "id": fixture_id,
                "category": exchange.arguments.get("category", "general"),
                "title": exchange.arguments.get("title", ""),
                "content": exchange.arguments.get("content", ""),
                "tags": list(exchange.arguments.get("tags") or []),
                "scope": exchange.arguments.get("scope", "personal"),
                "metadata": {"dsh_node_id": node_id, "turn": exchange.turn, "step": exchange.step},
            }
            for key in IDENTITY_FIELDS:
                value = exchange.arguments.get(key)
                if key == "owner" and value is None:
                    value = default_owner
                if value is not None:
                    memory[key] = value
            report.memories.append(memory)

        elif tool in {"fibbrain_recall", "fibmind_context", "fibmind_search"}:
            query = exchange.arguments.get("goal") or exchange.arguments.get("query") or ""
            if not str(query).strip():
                report.skipped.append(f"{exchange.call_id}: recall without a goal")
                continue
            result = exchange.result or {}
            hits = result.get("hits") if "hits" in result else result.get("results")
            relevant: list[str] = []
            for hit in hits or []:
                node_id = str((hit or {}).get("node_id") or "")
                fixture_id = node_to_fixture.get(node_id)
                if fixture_id and fixture_id not in relevant:
                    relevant.append(fixture_id)
            case: dict[str, Any] = {
                "id": f"turn{exchange.turn}_step{exchange.step}_{exchange.call_id}",
                "query": str(query),
                "relevant_ids": relevant,
                "expected_answer_terms": [],
                "review": (
                    "auto-labelled from recall output; confirm relevant_ids, add "
                    "forbidden_ids and expected_answer_terms"
                ),
            }
            for key in IDENTITY_FIELDS:
                value = exchange.arguments.get(key)
                if key == "owner" and value is None:
                    value = default_owner
                if value is not None:
                    case[key] = value
            report.cases.append(case)

    return report


def _fixture_id(title: str, node_id: str, used: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.casefold()).strip("_")[:40] or node_id[-8:]
    candidate = slug
    counter = 2
    while candidate in used:
        candidate = f"{slug}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def import_session(
    path: str | Path,
    *,
    name: str = DEFAULT_DATASET_NAME,
    default_owner: str | None = None,
) -> tuple[dict[str, Any], ImportReport]:
    events = read_events(path)
    exchanges = pair_tool_exchanges(events)
    report = build_report(exchanges, str(path), default_owner=default_owner)
    dataset = report.dataset(
        name,
        f"Imported from dsh session {Path(path).parent.name}; labels need review.",
    )
    return dataset, report


# --------------------------------------------------------------------------
# CLI


def _overlay(root: Path) -> str:
    python = root / ".venv" / "bin" / "python"
    store = root / ".fibmind" / "memory.db"
    return (
        "- insert:\n"
        "    - id: memory-fibbrain\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        "        serverName: fibbrain\n"
        "        transport: stdio\n"
        f"        command: {python}\n"
        "        args:\n"
        "          - -m\n"
        "          - fibmind.mcp_server\n"
        "          - --store\n"
        f"          - {store}\n"
        "        env: {}\n"
        f"        cwd: {root}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.import_dsh_session",
        description="Convert a dsh session log into a P0 evaluation dataset candidate.",
    )
    parser.add_argument("session", nargs="?", help="path to session.jsonl or session.jsonl.zstd")
    parser.add_argument("--output", type=Path, help="where to write the candidate dataset")
    parser.add_argument("--name", default=DEFAULT_DATASET_NAME, help="dataset name")
    parser.add_argument("--owner", default=None, help="owner to assume when calls omit it")
    parser.add_argument(
        "--print-overlay",
        action="store_true",
        help="print a dsh cordis overlay for this checkout and exit",
    )
    args = parser.parse_args(argv)

    if args.print_overlay:
        sys.stdout.write(_overlay(Path(__file__).resolve().parents[1]))
        return 0
    if not args.session:
        parser.error("session path is required unless --print-overlay is given")

    dataset, report = import_session(args.session, name=args.name, default_owner=args.owner)
    serialized = json.dumps(dataset, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        target = str(args.output)
    else:
        sys.stdout.write(serialized)
        target = "stdout"

    sys.stderr.write(
        f"fibbrain tool calls: {report.tool_calls_seen}, memories: {len(report.memories)}, "
        f"cases: {len(report.cases)}, skipped: {len(report.skipped)} -> {target}\n"
    )
    for line in report.skipped:
        sys.stderr.write(f"  skip {line}\n")
    if report.memories and not report.cases:
        sys.stderr.write("  note: no recall calls found; add cases by hand\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
