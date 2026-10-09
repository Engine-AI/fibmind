"""Tests for cross-AI tool session and memory sharing via task_id and unified owner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fibmind import FibBrain, MemoryScope, MemoryService
from fibmind.brain import BrainState
from fibmind.models import MemoryNode, MemoryStatus
from fibmind.visibility import is_visible


class CrossToolVisibilityTests(unittest.TestCase):
    def test_session_scope_visibility_by_task_id(self) -> None:
        node = MemoryNode(
            id="node_scratch_1",
            title="Codex decision",
            content="Use WAL mode for SQLite",
            category="episode",
            scope=MemoryScope.SESSION,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-codex-1",
            task_id="task-opt-storage",
            status=MemoryStatus.ACTIVE,
        )

        # 1. Same session -> visible
        self.assertTrue(
            is_visible(
                node,
                owner="abbila",
                workspace_id="fibmind",
                project_id="fibmind",
                session_id="session-codex-1",
            )
        )

        # 2. Different session without task_id -> invisible
        self.assertFalse(
            is_visible(
                node,
                owner="abbila",
                workspace_id="fibmind",
                project_id="fibmind",
                session_id="session-agy-2",
            )
        )

        # 3. Different session with matching task_id -> visible
        self.assertTrue(
            is_visible(
                node,
                owner="abbila",
                workspace_id="fibmind",
                project_id="fibmind",
                session_id="session-agy-2",
                task_id="task-opt-storage",
            )
        )

        # 4. Different session with mismatched task_id -> invisible
        self.assertFalse(
            is_visible(
                node,
                owner="abbila",
                workspace_id="fibmind",
                project_id="fibmind",
                session_id="session-agy-2",
                task_id="other-task",
            )
        )

        # 5. Matching task_id but mismatched owner -> invisible
        self.assertFalse(
            is_visible(
                node,
                owner="other_user",
                workspace_id="fibmind",
                project_id="fibmind",
                session_id="session-agy-2",
                task_id="task-opt-storage",
            )
        )

        # 6. Matching task_id but mismatched workspace -> invisible
        self.assertFalse(
            is_visible(
                node,
                owner="abbila",
                workspace_id="other_workspace",
                project_id="fibmind",
                session_id="session-agy-2",
                task_id="task-opt-storage",
            )
        )


class CrossToolBrainHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.db"
        self.service = MemoryService(self.db_path)
        self.brain = FibBrain(self.service)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cross_tool_episode_handoff(self) -> None:
        codex_state = BrainState(
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="sess-codex-001",
            task_id="task-cross-tool",
        )
        agy_state = BrainState(
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="sess-agy-002",
            task_id="task-cross-tool",
        )
        unrelated_state = BrainState(
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="sess-claude-003",
            task_id="unrelated-task",
        )

        # Tool 1 (Codex) observes a decision and a tool result
        self.brain.observe(
            kind="decision",
            summary="Identified bottleneck in SqliteStore _load: avoid full table scans",
            payload={"tool": "profiler"},
            state=codex_state,
        )
        self.brain.observe(
            kind="tool_result",
            summary="tests/test_storage.py passed with 274 tests",
            payload={"files": ["src/fibmind/storage.py"]},
            state=codex_state,
        )

        # Tool 2 (AGY) performs recall with the same task_id
        pack = self.brain.recall(
            goal="optimize storage load",
            state=agy_state,
            include_hot=False,
        )

        # AGY should have received Codex's episode events
        episode_summaries = [ev["summary"] for ev in pack["episode"]]
        self.assertTrue(any("Identified bottleneck in SqliteStore" in s for s in episode_summaries))
        self.assertTrue(any("test_storage.py passed" in s for s in episode_summaries))

        # Tool 3 (Unrelated task) should not see Codex's episode
        unrelated_pack = self.brain.recall(
            goal="optimize storage load",
            state=unrelated_state,
            include_hot=False,
        )
        unrelated_summaries = [ev["summary"] for ev in unrelated_pack["episode"]]
        self.assertFalse(any("Identified bottleneck in SqliteStore" in s for s in unrelated_summaries))

    def test_cross_tool_goal_sharing(self) -> None:
        codex_state = BrainState(
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="sess-codex-plan",
            task_id="task-shared-plan",
        )
        agy_state = BrainState(
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="sess-agy-exec",
            task_id="task-shared-plan",
        )

        # Tool 1 plans a multi-step objective
        plan = self.brain.plan(
            objective="Implement cross-AI session sync",
            state=codex_state,
        )
        self.assertEqual(plan["status"], "active")
        goal_id = plan["goal_id"]

        # Tool 2 resolves the plan via task_id without passing goal_id
        agy_plan = self.brain.plan(
            state=agy_state,
        )
        self.assertEqual(agy_plan["goal_id"], goal_id)
        self.assertEqual(agy_plan["objective"], "Implement cross-AI session sync")


class TaskOnlyScratchTests(unittest.TestCase):
    """A tool that knows the task but not another tool's session id."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = MemoryService(Path(self.temp_dir.name) / "memory.db")
        self.brain = FibBrain(self.service)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_observe_with_task_only_is_persisted_and_shared(self) -> None:
        writer = BrainState(owner="abbila", workspace_id="fibmind", project_id="fibmind", task_id="t-handoff")
        event = self.brain.observe(kind="decision", summary="Keep WAL mode on", state=writer)
        self.assertIsNotNone(event["node_id"])

        # A fresh brain (another process / AI tool) recalls it by task alone.
        reader = FibBrain(self.service)
        pack = reader.recall(
            goal="wal mode",
            state=BrainState(
                owner="abbila", workspace_id="fibmind", project_id="fibmind", session_id="s-other", task_id="t-handoff"
            ),
            include_hot=False,
        )
        self.assertIn("Keep WAL mode on", [ev["summary"] for ev in pack["episode"]])

    def test_remember_session_scope_with_task_only_writes(self) -> None:
        result = self.brain.remember(
            "decision",
            "Retry policy",
            "Retry the flaky upload step twice before failing the build.",
            scope="session",
            state=BrainState(owner="abbila", workspace_id="fibmind", project_id="fibmind", task_id="t-retry"),
        )
        self.assertEqual(result["verdict"], "write")


class IdentityDefaultsTests(unittest.TestCase):
    def test_apply_maps_host_owner_and_fills_missing(self) -> None:
        from fibmind.mcp_server import IdentityDefaults

        defaults = IdentityDefaults(owner="abbila", workspace_id="fibmind")
        accepted = {"owner", "workspace_id", "project_id", "task_id", "metadata"}
        self.assertEqual(defaults.apply({}, accepted)["owner"], "abbila")
        mapped = defaults.apply({"owner": "codex"}, accepted)
        self.assertEqual(mapped["owner"], "abbila")
        self.assertEqual(mapped["metadata"], {"host": "codex"})
        self.assertEqual(mapped["workspace_id"], "fibmind")
        # A real, different human owner is never overridden.
        self.assertEqual(defaults.apply({"owner": "alice"}, accepted)["owner"], "alice")
        # Explicit context wins over the default.
        self.assertEqual(defaults.apply({"workspace_id": "other"}, accepted)["workspace_id"], "other")

    def test_from_env_reads_aliases(self) -> None:
        from fibmind.mcp_server import IdentityDefaults

        defaults = IdentityDefaults.from_env({"FIBMIND_OWNER": "abbila", "FIBMIND_OWNER_ALIASES": "windsurf"})
        self.assertEqual(defaults.owner, "abbila")
        self.assertIn("windsurf", defaults.owner_aliases)
        self.assertIn("claude-code", defaults.owner_aliases)

    def test_two_hosts_share_memory_through_server_defaults(self) -> None:
        import asyncio

        from fibmind.mcp_server import IdentityDefaults, build_server

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "memory.db"
            defaults = IdentityDefaults(owner="abbila")
            codex = build_server(MemoryService(db), defaults=defaults)
            claude = build_server(MemoryService(db), defaults=defaults)

            async def scenario() -> None:
                await codex.call_tool(
                    "fibbrain_remember",
                    {
                        "category": "decision",
                        "title": "Storage loader",
                        "content": "SqliteStore._load reads columns by index to skip sqlite3.Row overhead.",
                        "owner": "codex",
                        "workspace_id": "fibmind",
                        "project_id": "fibmind",
                    },
                )
                recalled = await claude.call_tool(
                    "fibmind_search",
                    {"query": "SqliteStore load columns", "workspace_id": "fibmind", "project_id": "fibmind"},
                )
                results = recalled.structured_content["results"]
                self.assertTrue(results, "claude should see the memory codex wrote")

            asyncio.run(scenario())
            nodes = MemoryService(db).read(
                lambda memory: [n for n in memory.nodes.values() if n.title == "Storage loader"]
            )
            self.assertEqual(nodes[0].owner, "abbila")
            self.assertEqual(nodes[0].metadata.get("host"), "codex")


class UnifyOwnerMigrationTests(unittest.TestCase):
    def test_unify_owner_is_logged_and_replayable(self) -> None:
        from fibmind.graph import FibMind
        from fibmind.migrate import unify_owner

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "memory.db"
            service = MemoryService(db)
            for host in ("codex", "claude-code"):
                service.append(
                    "decision",
                    f"{host} note",
                    f"Decision recorded by {host} about the retrieval cache.",
                    owner=host,
                    workspace_id="fibmind",
                    project_id="fibmind",
                )
            service.append("decision", "alice note", "Alice keeps her own memory.", owner="alice")

            preview = unify_owner(db, "abbila", dry_run=True)
            self.assertEqual(preview["would_reassign"], 2)

            result = unify_owner(db, "abbila")
            self.assertEqual(result["reassigned"], 2)

            def owners(memory: FibMind) -> dict[str, str | None]:
                return {n.title: n.owner for n in memory.nodes.values() if n.title.endswith(" note")}

            live = MemoryService(db).read(owners)
            self.assertEqual(live, {"codex note": "abbila", "claude-code note": "abbila", "alice note": "alice"})
            replayed = MemoryService(db).read(lambda memory: owners(FibMind.rebuild_from_log(memory.events)))
            self.assertEqual(replayed, live)
            hosts = MemoryService(db).read(
                lambda memory: {n.metadata.get("host") for n in memory.nodes.values() if n.title.endswith(" note")}
            )
            self.assertEqual(hosts, {"codex", "claude-code", None})

            # Idempotent.
            self.assertEqual(unify_owner(db, "abbila")["reassigned"], 0)
