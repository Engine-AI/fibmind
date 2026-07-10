"""Smoke tests for the FastMCP wiring (skipped if the mcp SDK is absent)."""

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
    "fibmind_search",
    "fibmind_search_from",
    "fibmind_context",
}


def _result_json(result: object) -> dict:
    """Extract the structured payload from a FastMCP call_tool result.

    call_tool returns ``(content_blocks, structured_result)``; the structured
    result is the tool's return value as a dict.
    """
    _content, structured = result
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

                found = await server.call_tool("fibmind_search", {"query": "login token"})
                self.assertTrue(
                    any(r["node_id"] == node_id for r in _result_json(found)["results"])
                )

                tree = await server.call_tool("fibmind_search_from", {"node_id": node_id})
                self.assertEqual(_result_json(tree)["root"], node_id)

                ctx = await server.call_tool("fibmind_context", {"goal": "login token expiry"})
                self.assertIn("Login bug", _result_json(ctx)["text"])

            asyncio.run(scenario())

    def test_parse_args_defaults_to_dot_fibmind(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.store, ".fibmind/memory.json")

    def test_parse_args_accepts_custom_store(self) -> None:
        args = _parse_args(["--store", "/tmp/custom.json"])
        self.assertEqual(args.store, "/tmp/custom.json")


if __name__ == "__main__":
    unittest.main()
