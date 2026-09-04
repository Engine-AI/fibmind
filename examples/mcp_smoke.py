"""Local stdio MCP smoke for FibBrain.

Starts ``python -m fibmind.mcp_server`` as a real MCP subprocess, then walks
the Brain protocol. Intended for a machine-local check, not a harness plugin.

    .venv/bin/python examples/mcp_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
STORE = ROOT / ".fibmind" / "mcp-smoke.db"

STATE = {
    "owner": "local-smoke",
    "workspace_id": "ws-fibmind",
    "project_id": "fibmind",
    "session_id": "smoke-session",
}


def _payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
    raise AssertionError(f"MCP tool returned no structured payload: {result!r}")


def _print(title: str, payload: dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


async def main() -> int:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    for leftover in STORE.parent.glob("mcp-smoke.db*"):
        leftover.unlink()

    params = StdioServerParameters(
        command=str(PYTHON),
        args=["-m", "fibmind.mcp_server", "--store", str(STORE)],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
            print(f"server: {getattr(info, 'name', '?')} {getattr(info, 'version', '')}".strip())

            tools = await session.list_tools()
            names = sorted(tool.name for tool in tools.tools)
            brain = [name for name in names if name.startswith("fibbrain_")]
            print("brain tools:", ", ".join(brain))
            required = {
                "fibbrain_recall",
                "fibbrain_remember",
                "fibbrain_observe",
                "fibbrain_advise",
                "fibbrain_reflect",
                "fibbrain_plan",
                "fibbrain_coordinate",
                "fibbrain_complete_goal",
            }
            missing = required - set(names)
            if missing:
                raise SystemExit(f"missing tools: {sorted(missing)}")

            planned = _payload(
                await session.call_tool(
                    "fibbrain_plan",
                    {"objective": "Verify local FibBrain MCP stdio", **STATE},
                )
            )
            _print("plan", planned)
            if not planned.get("goal_id") or not planned.get("steps"):
                raise SystemExit("plan did not persist a goal")

            coordinated = _payload(
                await session.call_tool(
                    "fibbrain_coordinate",
                    {"goal_id": planned["goal_id"], **STATE},
                )
            )
            _print("coordinate", coordinated)
            names = [item.get("capability") for item in coordinated.get("capabilities") or []]
            if names[:1] != ["memory"]:
                raise SystemExit("coordinate should start with memory")

            observed = _payload(
                await session.call_tool(
                    "fibbrain_observe",
                    {"kind": "turn_start", "summary": "local MCP smoke started"},
                )
            )
            _print("observe", observed)

            remembered = _payload(
                await session.call_tool(
                    "fibbrain_remember",
                    {
                        "category": "decision",
                        "title": "Local MCP smoke decision",
                        "content": "FibBrain MCP stdio remember/recall works on this machine.",
                        **STATE,
                    },
                )
            )
            _print("remember", remembered)
            if remembered.get("verdict") != "write" or "node_id" not in remembered:
                raise SystemExit("remember did not write")

            duplicate = _payload(
                await session.call_tool(
                    "fibbrain_remember",
                    {
                        "category": "decision",
                        "title": "Local MCP smoke decision",
                        "content": "FibBrain MCP stdio remember/recall works on this machine.",
                        **STATE,
                    },
                )
            )
            _print("remember duplicate", duplicate)
            if duplicate.get("verdict") != "skip":
                raise SystemExit("duplicate remember was not skipped")

            recalled = _payload(
                await session.call_tool(
                    "fibbrain_recall",
                    {"goal": "Does local MCP remember/recall work?", **STATE},
                )
            )
            _print("recall", {"text": recalled.get("text"), "hits": recalled.get("hits"), "episode": recalled.get("episode")})
            if "Local MCP smoke decision" not in (recalled.get("text") or ""):
                raise SystemExit("recall missed the written memory")

            advised = _payload(
                await session.call_tool(
                    "fibbrain_advise",
                    {"action": "remember", "kind": "tool", **STATE},
                )
            )
            _print("advise", advised)
            if advised.get("verdict") != "allow":
                raise SystemExit("advise should allow an unused action")

            reflected = _payload(
                await session.call_tool(
                    "fibbrain_reflect",
                    {
                        "verdict": "confirmed",
                        "source": "examples/mcp_smoke.py",
                        "node_id": remembered["node_id"],
                    },
                )
            )
            _print("reflect", reflected)
            if reflected.get("confidence", 0) <= 0:
                raise SystemExit("reflect did not raise confidence")

            completed = _payload(
                await session.call_tool(
                    "fibbrain_complete_goal",
                    {"goal_id": planned["goal_id"], **STATE},
                )
            )
            _print("complete_goal", completed)
            if completed.get("status") != "complete":
                raise SystemExit("complete_goal did not mark the goal complete")

    print(f"\nOK  store={STORE}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
