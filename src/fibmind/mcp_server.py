"""FibMind MCP server: exposes long-term memory tools over stdio.

Run it directly::

    python -m fibmind.mcp_server --store .fibmind/memory.json

Or register it with Claude Code::

    claude mcp add --scope project --transport stdio fibmind -- \\
        /path/to/.venv/bin/python -m fibmind.mcp_server --store .fibmind/memory.json

The heavy lifting lives in :mod:`fibmind.service`; this module is only the thin
FastMCP wiring so it can stay optional (the ``mcp`` dependency is not needed by
the core package or its tests).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from fibmind.service import MemoryService

DEFAULT_STORE = ".fibmind/memory.json"


def build_server(service: MemoryService) -> FastMCP:
    """Create a FastMCP app whose tools delegate to ``service``."""
    mcp = FastMCP("fibmind")

    @mcp.tool()
    def fibmind_append(
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save one memory (task result, error, requirement, plan, or code summary).

        Use this after finishing a task to record what happened. ``category``
        groups related memories into a tree (e.g. "code", "requirement",
        "decision"). Returns the created node summary including its node_id.
        """
        return service.append(category, title, content, tags=tags, metadata=metadata)

    @mcp.tool()
    def fibmind_search(
        query: str,
        top_k: int = 5,
        categories: list[str] | None = None,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Search memories by keyword relevance to ``query``.

        Read-only. Returns the top ranked memories with their node_ids so you
        can expand or link them. Optionally restrict to ``categories``.
        """
        return service.search(query, top_k=top_k, categories=categories, min_score=min_score)

    @mcp.tool()
    def fibmind_search_from(
        node_id: str,
        depth: int = 2,
        relation_types: list[str] | None = None,
        reinforce: bool = False,
    ) -> dict[str, Any]:
        """Expand the association tree rooted at ``node_id`` up to ``depth`` hops.

        Use a node_id returned by fibmind_search or fibmind_context to explore
        related memories. Set ``reinforce`` to strengthen the traversed memories.
        """
        return service.search_from(
            node_id, depth=depth, relation_types=relation_types, reinforce=reinforce
        )

    @mcp.tool()
    def fibmind_context(
        goal: str,
        top_k: int = 5,
        depth: int = 1,
        max_chars: int = 2000,
        reinforce: bool = False,
    ) -> dict[str, Any]:
        """Build a compact memory pack to load into context before a task.

        Call this at the START of a multi-step task with the user's goal. It
        searches for relevant history and expands related memories, returning a
        char-bounded ``text`` block ready to drop into your working context,
        plus the structured ``hits``.
        """
        return service.context(
            goal, top_k=top_k, depth=depth, max_chars=max_chars, reinforce=reinforce
        )

    return mcp


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fibmind.mcp_server", description="FibMind MCP server")
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help=f"Path to the JSON memory store (default: {DEFAULT_STORE})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    service = MemoryService(Path(args.store))
    server = build_server(service)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
