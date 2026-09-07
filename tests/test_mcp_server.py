"""Smoke tests for the MCP server wiring (skipped if the SDK is absent)."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from fibmind.mcp_server import _parse_args, build_server  # noqa: E402
from fibmind.service import MemoryService  # noqa: E402

EXPECTED_TOOLS = {
    "fibmind_append",
    "fibmind_link",
    "fibmind_search",
    "fibmind_search_from",
    "fibmind_context",
    "fibmind_record_outcome",
    "fibmind_revise",
    "fibmind_mark_stale",
    "fibmind_forget",
    "fibmind_promote_knowledge",
    "fibbrain_recall",
    "fibbrain_remember",
    "fibbrain_observe",
    "fibbrain_advise",
    "fibbrain_reflect",
    "fibbrain_plan",
    "fibbrain_coordinate",
    "fibbrain_complete_goal",
    "fibbrain_review_session",
    "fibbrain_pending_reviews",
    "fibbrain_approve_memory",
    "fibbrain_reject_memory",
    "fibbrain_remember_procedure",
    "fibbrain_render",
}


def _result_json(result: object) -> dict:
    """Extract the structured payload from an MCPServer call_tool result.

    MCP 2 returns a ``CallToolResult`` carrying ``structured_content``.
    """
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, dict):
        raise AssertionError("tool call did not return structured content")
    return structured


class McpServerTests(unittest.TestCase):
    def test_all_v1_tools_are_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(Path(tmp) / "memory.json")
            server = build_server(service)

            registered = {tool.name for tool in server._tool_manager.list_tools()}
            self.assertEqual(EXPECTED_TOOLS, registered)

    def test_tools_delegate_to_service_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(Path(tmp) / "memory.json")
            server = build_server(service)

            async def scenario() -> None:
                appended = await server.call_tool(
                    "fibmind_append",
                    {
                        "category": "code",
                        "title": "Login bug",
                        "content": "401 after refresh token expiry",
                    },
                )
                node_id = _result_json(appended)["node_id"]

                second = await server.call_tool(
                    "fibmind_append",
                    {
                        "category": "requirement",
                        "title": "Refresh token",
                        "content": "Rotate token before expiry",
                    },
                )
                second_id = _result_json(second)["node_id"]

                linked = await server.call_tool(
                    "fibmind_link",
                    {
                        "from_node_id": node_id,
                        "to_node_id": second_id,
                        "relation_type": "related_to",
                        "weight": 0.8,
                    },
                )
                self.assertEqual(_result_json(linked)["relation_type"], "related_to")

                found = await server.call_tool("fibmind_search", {"query": "login token"})
                self.assertTrue(
                    any(r["node_id"] == node_id for r in _result_json(found)["results"])
                )

                tree = await server.call_tool("fibmind_search_from", {"node_id": node_id})
                self.assertEqual(_result_json(tree)["root"], node_id)

                ctx = await server.call_tool("fibmind_context", {"goal": "login token expiry"})
                self.assertIn("Login bug", _result_json(ctx)["text"])

                recalled = await server.call_tool(
                    "fibbrain_recall",
                    {"goal": "login token expiry"},
                )
                self.assertIn("Login bug", _result_json(recalled)["text"])
                remembered = await server.call_tool(
                    "fibbrain_remember",
                    {
                        "category": "decision",
                        "title": "Retry policy",
                        "content": "retry three times with backoff",
                    },
                )
                self.assertEqual(_result_json(remembered)["verdict"], "write")

            asyncio.run(scenario())

    def test_parse_args_defaults_to_dot_fibmind(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.store, ".fibmind/memory.db")

    def test_parse_args_accepts_custom_store(self) -> None:
        args = _parse_args(["--store", "/tmp/custom.json"])
        self.assertEqual(args.store, "/tmp/custom.json")


if __name__ == "__main__":
    unittest.main()
