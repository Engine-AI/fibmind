"""Real independent MCP processes; client names stand for harnesses, not GUI tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.types import Implementation  # noqa: E402

from fibmind.integrations import server_config  # noqa: E402
from fibmind.service import MemoryService  # noqa: E402


class CrossApplicationProtocolTests(unittest.TestCase):
    def test_three_independent_stdio_clients_share_context_without_cross_project_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            store = root / "shared" / "memory.db"
            workspaces = {}
            for label in ("project-a", "project-b"):
                path = root / label
                (path / ".git").mkdir(parents=True)
                (path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
                workspaces[label] = path

            async def scenario() -> None:
                async with AsyncExitStack() as stack:

                    async def connect(host, suffix=""):
                        cwd = root / f"host-{host}{suffix}"
                        cwd.mkdir()
                        config = server_config(host, owner="abbila", store=store, python=sys.executable)
                        env = {key: value for key, value in os.environ.items() if not key.startswith("FIBMIND_")}
                        params = StdioServerParameters(
                            command=config["command"], args=config["args"], cwd=str(cwd), env={**env, **config["env"]}
                        )
                        read, write = await stack.enter_async_context(stdio_client(params))
                        client = await stack.enter_async_context(
                            ClientSession(read, write, client_info=Implementation(name=host, version="test"))
                        )
                        await client.initialize()
                        return client

                    codex, claude, agy = [await connect(host) for host in ("codex", "claude-code", "agy")]

                    async def call(client, tool, arguments):
                        result = await client.call_tool(tool, arguments)
                        self.assertFalse(result.is_error, result)
                        self.assertIsInstance(result.structured_content, dict)
                        return result.structured_content

                    state_a = {
                        "owner": "abbila",
                        "workspace_path": str(workspaces["project-a"]),
                        "session_id": "codex-session",
                    }
                    preference = await call(
                        codex,
                        "fibbrain_remember",
                        {
                            **state_a,
                            "category": "preference",
                            "title": "Response language",
                            "content": "Prefer concise Mandarin Chinese explanations.",
                            "context_level": "user",
                        },
                    )
                    decision = await call(
                        codex,
                        "fibbrain_remember",
                        {
                            **state_a,
                            "category": "decision",
                            "title": "Quasar upload recovery",
                            "content": "Quasar uploads retry twice with bounded exponential backoff.",
                            "context_level": "project",
                        },
                    )
                    revision = MemoryService(store)._store.revision()
                    context_a = await call(
                        claude,
                        "fibbrain_context",
                        {
                            **state_a,
                            "session_id": "claude-session",
                            "goal": "quasar upload recovery",
                            "budget_tokens": 300,
                        },
                    )
                    self.assertEqual(
                        {m["node_id"] for m in context_a["memories"]}, {preference["node_id"], decision["node_id"]}
                    )
                    self.assertTrue(context_a["has_task_context"])
                    self.assertTrue(all(m["hosts"] == ["codex"] for m in context_a["memories"]))
                    context_b = await call(
                        agy,
                        "fibbrain_context",
                        {
                            "owner": "abbila",
                            "workspace_path": str(workspaces["project-b"]),
                            "session_id": "agy-session",
                            "goal": "quasar upload recovery",
                        },
                    )
                    self.assertEqual([m["node_id"] for m in context_b["memories"]], [preference["node_id"]])
                    self.assertFalse(context_b["has_task_context"])
                    self.assertEqual(MemoryService(store)._store.revision(), revision)
                    stranger = await call(
                        agy,
                        "fibbrain_context",
                        {**context_a["state"], "owner": "bob", "goal": "quasar upload recovery"},
                    )
                    self.assertEqual(stranger["memories"], [])

                    # A later process has no shared Python objects or hot cache.
                    restarted = await connect("claude-code", "-restarted")
                    later = await call(
                        restarted,
                        "fibbrain_context",
                        {**context_a["state"], "session_id": "later-session", "goal": "quasar upload recovery"},
                    )
                    self.assertIn(decision["node_id"], [m["node_id"] for m in later["memories"]])
                    for host in ("codex", "claude-code", "agy", "claude-code-restarted"):
                        self.assertFalse((root / f"host-{host}" / ".fibmind" / "memory.db").exists())
                    self.assertNotIn("Quasar", json.dumps(stranger))

            asyncio.run(scenario())
