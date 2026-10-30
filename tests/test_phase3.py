"""Unit tests for Phase 3: AI-OS Entity World Model, Active Consolidation,
Skill Materialization & Revocation, and Proactive Background Warnings.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from fibmind import (
    BrainState,
    FibBrain,
    MemoryNode,
    MemoryService,
    compute_file_hashes,
)
from fibmind.retrieval import LexicalIndex


def _state(**overrides: str) -> BrainState:
    base = {
        "owner": "abbila",
        "workspace_id": "fibmind",
        "project_id": "fibmind",
        "session_id": "sess-phase3",
    }
    base.update(overrides)
    return BrainState(**base)


class Phase3EntityAndConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = MemoryService(self.root / "memory.db")
        self.brain = FibBrain(self.service)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lexical_index_anchor_inverted_index_o1(self) -> None:
        idx = LexicalIndex()
        node = MemoryNode(
            title="WAL Mode Storage",
            content="Use WAL journal mode",
            category="decision",
            metadata={"files": ["src/storage.py", "src/db/concurrency.py"], "symbols": ["open_wal_store"]},
        )
        idx.add(node)

        # 1. Exact path lookup
        self.assertEqual(idx.nodes_for_anchor("src/storage.py"), {node.id})
        # 2. Basename path lookup
        self.assertEqual(idx.nodes_for_anchor("storage.py"), {node.id})
        self.assertEqual(idx.nodes_for_anchor("concurrency.py"), {node.id})
        # 3. Symbol lookup
        self.assertEqual(idx.nodes_for_anchor("open_wal_store"), {node.id})
        # 4. Unknown anchor
        self.assertEqual(idx.nodes_for_anchor("unknown.py"), set())

        # 5. Removal cleans up inverted index
        idx.remove(node.id)
        self.assertEqual(idx.nodes_for_anchor("src/storage.py"), set())
        self.assertEqual(idx.nodes_for_anchor("storage.py"), set())
        self.assertEqual(len(idx.anchors), 0)

    def test_entity_profile_aggregates_complete_world_model(self) -> None:
        target_file = self.root / "src" / "storage.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text("wal_mode = True\n", encoding="utf-8")
        hashes = compute_file_hashes(["src/storage.py"], cwd=self.root)

        # 1. Invariant decision memory
        inv = self.brain.remember(
            "decision",
            "SQLite WAL journal mode",
            "Always enable PRAGMA journal_mode=WAL on connections.",
            metadata={
                "files": ["src/storage.py"],
                "file_hashes": hashes,
                "command": "pytest tests/test_storage.py",
            },
            state=_state(),
        )
        self.brain.reflect("confirmed", "pytest", node_id=inv["node_id"], state=_state())
        self.brain.reflect("confirmed", "bench", node_id=inv["node_id"], state=_state())

        # 2. Procedure memory
        proc = self.brain.remember_procedure(
            "Vacuum SQLite database",
            "Vacuum the SQLite database on startup",
            steps=["python -m fibmind.vacuum"],
            tools=["python"],
            when=["maintenance"],
            metadata={"files": ["src/storage.py"]},
            state=_state(),
        )

        # 3. Refuted / pitfall memory
        bad = self.brain.remember(
            "decision",
            "Use raw sqlite3 connect without timeout",
            "Do not pass timeout=5000 to sqlite3 connect.",
            metadata={"files": ["src/storage.py"]},
            state=_state(),
        )
        self.brain.reflect("refuted", "deadlock_detected", node_id=bad["node_id"], state=_state())

        # 4. Conflicted memory
        conf = self.brain.remember(
            "decision",
            "Delete WAL file on exit",
            "Manually unlink WAL file when connection closes.",
            metadata={
                "files": ["src/storage.py"],
                "conflicted": True,
                "conflict_reason": "causes database corruption",
            },
            state=_state(),
        )

        # Query entity profile for "src/storage.py"
        profile = self.brain.entity_profile("src/storage.py", state=_state(), workspace_root=self.root)

        self.assertEqual(profile["entity"], "src/storage.py")
        self.assertEqual(len(profile["invariants"]), 1)
        self.assertEqual(profile["invariants"][0]["node_id"], inv["node_id"])
        self.assertEqual(len(profile["procedures"]), 1)
        self.assertEqual(profile["procedures"][0]["node_id"], proc["node_id"])
        self.assertTrue(any(p["node_id"] == bad["node_id"] for p in profile["pitfalls"]))
        self.assertTrue(any(c["node_id"] == conf["node_id"] for c in profile["conflicts"]))
        self.assertIn("pytest tests/test_storage.py", profile["verification_commands"])
        self.assertEqual(len(profile["drift"]), 0)

        # Also test basename query
        profile_base = self.brain.entity_profile("storage.py", state=_state(), workspace_root=self.root)
        self.assertEqual(len(profile_base["invariants"]), 1)

    def test_consolidate_detects_environmental_drift_and_demotes(self) -> None:
        target_file = self.root / "src" / "worker.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text("concurrency = 4\n", encoding="utf-8")
        file_hash = hashlib.sha256(target_file.read_bytes()).hexdigest()

        written = self.brain.remember(
            "decision",
            "Worker concurrency settings",
            "Worker process concurrency is fixed at 4.",
            metadata={"files": ["src/worker.py"], "file_hashes": {"src/worker.py": file_hash}},
            state=_state(),
        )
        node_id = written["node_id"]
        # Confirm twice so confidence reaches 0.5
        self.brain.reflect("confirmed", "pytest", node_id=node_id, state=_state())
        self.brain.reflect("confirmed", "perf_test", node_id=node_id, state=_state())
        self.assertGreaterEqual(self.service.inspect(node_id, identity=_state().identity())["confidence"], 0.5)

        # 1. Consolidate before drift -> 0 drifted
        c1 = self.brain.consolidate(state=_state(), workspace_root=self.root)
        self.assertEqual(len(c1["drifted"]), 0)

        # 2. Induce drift by modifying file content on disk
        target_file.write_text("concurrency = 16\n", encoding="utf-8")

        # 3. Consolidate without auto_mark_drift -> reports drifted but confidence unchanged
        c_read = self.brain.consolidate(state=_state(), workspace_root=self.root, auto_mark_drift=False)
        self.assertEqual(len(c_read["drifted"]), 1)
        self.assertEqual(c_read["drifted"][0]["node_id"], node_id)
        node_mid = self.service.inspect(node_id, identity=_state().identity())
        self.assertGreaterEqual(node_mid["confidence"], 0.5)

        # 4. Consolidate with auto_mark_drift=True -> demotes confidence to 0.25 and marks drift_detected
        c2 = self.brain.consolidate(state=_state(), workspace_root=self.root, auto_mark_drift=True)
        self.assertEqual(len(c2["drifted"]), 1)
        node_after = self.service.inspect(node_id, identity=_state().identity())
        self.assertEqual(node_after["confidence"], 0.25)
        self.assertIn("file_hash_mismatch", node_after["metadata"]["drift_detected"])

    def test_sync_skills_materialization_and_auto_revocation(self) -> None:
        skills_dir = self.root / "skills"
        proc = self.brain.remember_procedure(
            "Run deploy verification smoke check",
            "Verify deployment health",
            steps=[f"{sys.executable} -c \"print('ok')\""],
            tools=["python"],
            when=["deployment", "verify"],
            state=_state(),
        )
        proc_id = proc["node_id"]

        # Initially candidate (0 confirmations) -> min_maturity="verified" does not materialize
        s0 = self.brain.sync_skills(skills_dir, state=_state(), min_maturity="verified")
        self.assertEqual(len(s0["materialized"]), 0)

        # Confirm procedure once -> maturity becomes "verified"
        self.brain.reflect("confirmed", "smoke_test", node_id=proc_id, state=_state(session_id="s1"))

        # Now sync_skills materializes it as SKILL.md
        s1 = self.brain.sync_skills(skills_dir, state=_state(), min_maturity="verified")
        self.assertEqual(len(s1["materialized"]), 1)
        skill_file = Path(s1["materialized"][0]["path"])
        self.assertTrue(skill_file.is_file())
        content = skill_file.read_text(encoding="utf-8")
        self.assertIn(f"FibBrain procedure `{proc_id}`", content)

        # Create a user-authored custom skill without FibBrain signature
        custom_skill_dir = skills_dir / "user-custom-skill"
        custom_skill_dir.mkdir(parents=True, exist_ok=True)
        custom_skill_file = custom_skill_dir / "SKILL.md"
        custom_skill_file.write_text("# User hand-written skill\nDo something custom.", encoding="utf-8")

        # Refute the procedure in memory
        self.brain.reflect("refuted", "failed_in_production", node_id=proc_id, state=_state())

        # Sync skills should auto-revoke the refuted FibBrain skill, but NEVER touch custom skill!
        s2 = self.brain.sync_skills(skills_dir, state=_state(), min_maturity="verified")
        self.assertEqual(len(s2["revoked"]), 1)
        self.assertEqual(s2["revoked"][0]["node_id"], proc_id)
        self.assertFalse(skill_file.exists())
        self.assertTrue(custom_skill_file.exists())

    def test_background_proactive_warnings_for_refuted_and_conflicted(self) -> None:
        # 1. Refuted memory about redis connection timeout
        bad = self.brain.remember(
            "decision",
            "Redis connection timeout setting",
            "Set redis timeout to 100 milliseconds.",
            state=_state(),
        )
        self.brain.reflect("refuted", "network_jitter_timeouts", node_id=bad["node_id"], state=_state())

        # 2. Conflicted memory about redis timeout
        self.brain.remember(
            "decision",
            "Redis connection timeout override",
            "Disable redis timeout completely.",
            metadata={"conflicted": True, "conflict_reason": "unbounded blocking"},
            state=_state(),
        )

        # When goal matches redis timeout -> background returns warnings
        ctx = self.brain.context("redis connection timeout", state=_state())
        self.assertIn("warnings", ctx)
        warnings = ctx["warnings"]
        self.assertEqual(len(warnings), 2)

        kinds = {w["kind"] for w in warnings}
        self.assertEqual(kinds, {"refuted", "conflict"})

        # When goal is empty -> warnings is empty list
        ctx_empty = self.brain.context("", state=_state())
        self.assertEqual(ctx_empty["warnings"], [])

    def test_mcp_entity_and_consolidate_tools(self) -> None:
        from fibmind.mcp_server import build_server

        server = build_server(self.service)
        self.assertIn("fibbrain_entity", server._tool_manager._tools)
        self.assertIn("fibbrain_consolidate", server._tool_manager._tools)


if __name__ == "__main__":
    unittest.main()
