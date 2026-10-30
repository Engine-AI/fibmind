"""Comprehensive tests for Track 2: World Model Projection & Multi-Host Skill Compiler."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fibmind.background import build_background
from fibmind.brain import BrainState, FibBrain
from fibmind.entity import (
    build_entity_profile,
    extract_code_imports,
    extract_entity_targets,
    extract_python_imports,
    resolve_transitive_imports,
)
from fibmind.graph import FibMind
from fibmind.models import (
    MemoryKind,
    MemoryNode,
    MemoryScope,
    MemoryStatus,
    RelationType,
    Verdict,
)
from fibmind.procedure import Procedure, parse_skill_markdown, sync_skills
from fibmind.reflex import (
    auto_ground_metadata,
    check_node_preconditions,
    compute_file_hashes,
    compute_symbol_hashes,
    diagnose_temporal_drift,
    locate_symbol_spans,
)
from fibmind.review import (
    SYMBOL_KEYS,
    EpisodeDigest,
    _node_anchors_meta,
    digest_episode,
    extract_candidates,
)
from fibmind.service import MemoryService
from fibmind.storage import SqliteStore
from fibmind.verify import revalidate_memories, run_command_and_reflect
from fibmind.working_context import resolve_git_commit, working_context


class QiVesselsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temp_dir.name)
        self.memory = FibMind()
        self.identity = {
            "owner": "abbila",
            "workspace_id": "fibmind",
            "project_id": "fibmind",
            "session_id": "session-main",
            "task_id": "track2",
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_transitive_drift_derived_from_and_related_to(self) -> None:
        """Transitive drift propagates across DERIVED_FROM and RELATED_TO edges."""
        # Node 1: active decision targeting src/fibmind/core.py without file preconditions
        node_1_id = self.memory.append(
            category="decision",
            title="Core module architecture src/fibmind/core.py",
            content="Maintain core architecture for src/fibmind/core.py",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Node 2: neighbor node with file preconditions on a non-existent file
        node_2_id = self.memory.append(
            category="decision",
            title="Helper precondition node",
            content="Precondition depends on src/fibmind/helper.py",
            metadata={"file_hashes": {"src/fibmind/helper.py": "abc123expected"}},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Link node 1 -> node 2 via DERIVED_FROM
        self.memory.link_nodes(node_1_id, node_2_id, RelationType.DERIVED_FROM)

        # Build entity profile for src/fibmind/core.py with cwd
        prof = build_entity_profile(
            self.memory,
            "src/fibmind/core.py",
            identity=self.identity,
            cwd=self.cwd,
        )

        # Node 1 should be reported in drift with transitive_drift via node 2
        drift_items = prof["drift"]
        self.assertEqual(len(drift_items), 1)
        self.assertEqual(drift_items[0]["node_id"], node_1_id)
        self.assertEqual(drift_items[0]["via_node_id"], node_2_id)
        self.assertTrue(drift_items[0]["reason"].startswith(f"transitive_drift via {node_2_id}: missing_hashed_file"))

        # Test RELATED_TO edge as well
        node_3_id = self.memory.append(
            category="requirement",
            title="Pipeline rule for src/fibmind/pipe.py",
            content="Pipeline specifications in src/fibmind/pipe.py",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        node_4_id = self.memory.append(
            category="requirement",
            title="Pipeline helper requirements",
            content="Requires file existence",
            metadata={"require_files_exist": True, "files": ["src/fibmind/missing_config.json"]},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        self.memory.link_nodes(node_3_id, node_4_id, RelationType.RELATED_TO)

        prof_pipe = build_entity_profile(
            self.memory,
            "src/fibmind/pipe.py",
            identity=self.identity,
            cwd=self.cwd,
        )
        self.assertEqual(len(prof_pipe["drift"]), 1)
        self.assertEqual(prof_pipe["drift"][0]["node_id"], node_3_id)
        self.assertEqual(prof_pipe["drift"][0]["via_node_id"], node_4_id)
        self.assertIn("missing_target_file", prof_pipe["drift"][0]["reason"])

        # Test that direct drift is NOT duplicated in transitive drift
        direct_drift_node_id = self.memory.append(
            category="decision",
            title="Direct drift node src/fibmind/direct.py",
            content="Direct drift",
            metadata={"file_hashes": {"src/fibmind/direct.py": "deadbeef"}},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        self.memory.link_nodes(direct_drift_node_id, node_2_id, RelationType.DERIVED_FROM)
        prof_direct = build_entity_profile(
            self.memory,
            "src/fibmind/direct.py",
            identity=self.identity,
            cwd=self.cwd,
        )
        direct_drift_entries = [d for d in prof_direct["drift"] if d["node_id"] == direct_drift_node_id]
        self.assertEqual(len(direct_drift_entries), 1)
        self.assertTrue(direct_drift_entries[0]["reason"].startswith("missing_hashed_file"))

    def test_cross_session_active_leases(self) -> None:
        """Active leases are detected for concurrent sessions on the same target entity."""
        target = "src/fibmind/compiler.py"

        # 1. Other session active goal mentioning target
        lease_goal_id = self.memory.append(
            category="goal",
            title="Implement compiler optimizations",
            content=json.dumps({"status": "active", "objective": f"Refactor {target}"}),
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-other-1",
            task_id="opt-task",
        )

        # 2. Other session active episode mentioning target
        lease_episode_id = self.memory.append(
            category="episode",
            title="Debugging session on compiler.py",
            content=f"Fixing AST generation in {target}",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-other-2",
            task_id="debug-task",
        )

        # 3. Other session completed goal mentioning target -> should NOT be included
        self.memory.append(
            category="goal",
            title="Completed compiler setup",
            content=json.dumps({"status": "complete", "objective": f"Setup {target}"}),
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-other-3",
            task_id="setup-task",
        )

        # 4. Same session active goal mentioning target -> should NOT be reported as foreign lease
        self.memory.append(
            category="goal",
            title="Current session goal",
            content=json.dumps({"status": "active", "objective": f"Work on {target}"}),
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # 5. Different workspace/project active goal -> should NOT be included
        self.memory.append(
            category="goal",
            title="Foreign workspace goal",
            content=json.dumps({"status": "active", "objective": f"Work on {target}"}),
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="other-workspace",
            project_id="other-project",
            session_id="session-foreign",
            task_id="foreign-task",
        )

        prof = build_entity_profile(self.memory, target, identity=self.identity)
        leases = prof.get("active_leases", [])
        self.assertEqual(len(leases), 2)
        lease_node_ids = {item["node_id"] for item in leases}
        self.assertEqual(lease_node_ids, {lease_goal_id, lease_episode_id})

        goal_lease = next(item for item in leases if item["node_id"] == lease_goal_id)
        self.assertEqual(goal_lease["session_id"], "session-other-1")
        self.assertEqual(goal_lease["kind"], "goal")
        self.assertEqual(goal_lease["task_id"], "opt-task")

        ep_lease = next(item for item in leases if item["node_id"] == lease_episode_id)
        self.assertEqual(ep_lease["session_id"], "session-other-2")
        self.assertEqual(ep_lease["kind"], "episode")

    def test_extract_entity_targets(self) -> None:
        """Entity targets are extracted from goal tokens, code extensions, anchors, and task metadata."""
        # Seed an anchor node in lexical index
        self.memory.append(
            category="decision",
            title="SpecialSymbolAnchor definition",
            content="Anchor symbol details",
            metadata={"symbols": ["SpecialSymbolAnchor"]},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
        )

        task_node = MemoryNode(
            title="Task node",
            content="Task content",
            category="task",
            metadata={
                "files": ["src/fibmind/service.py", "config/settings.json"],
                "changed_files": ["src/fibmind/service.py", "docs/architecture.md"],
            },
        )

        goal = "Refactor src/fibmind/models.py and inspect SpecialSymbolAnchor, plus check index.ts!"
        targets = extract_entity_targets(self.memory, goal, [task_node], limit=5)

        self.assertIn("src/fibmind/models.py", targets)
        self.assertIn("index.ts", targets)
        self.assertIn("SpecialSymbolAnchor", targets)
        self.assertIn("src/fibmind/service.py", targets)
        self.assertIn("config/settings.json", targets)
        self.assertEqual(len(targets), 5)
        # Deduplication check
        self.assertEqual(len(targets), len(set(targets)))

    def test_background_entities_projection_and_warnings(self) -> None:
        """build_background projects entities and surfaces drift and active lease warnings."""
        target = "src/fibmind/router.py"

        # Create active lease on router.py from another session
        self.memory.append(
            category="goal",
            title="Concurrent routing upgrade",
            content=json.dumps({"status": "active", "objective": f"Refactoring {target}"}),
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-worker-9",
            task_id="route-task",
        )

        # Create drifted invariant for router.py
        self.memory.append(
            category="decision",
            title="Router dispatch rule src/fibmind/router.py",
            content="Route dispatch specification",
            metadata={"file_hashes": {target: "bad_hash_1234"}},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Call build_background with goal referencing router.py
        bg = build_background(
            self.memory,
            f"Fix routing logic in {target}",
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
            cwd=self.cwd,
        )

        self.assertIn("entities", bg)
        self.assertTrue(len(bg["entities"]) >= 1)
        entity_profile = next((p for p in bg["entities"] if target in p["entity"]), None)
        self.assertIsNotNone(entity_profile)
        self.assertTrue(len(entity_profile["drift"]) >= 1)
        self.assertTrue(len(entity_profile["active_leases"]) >= 1)

        warnings = bg["warnings"]
        drift_warnings = [w for w in warnings if w.get("kind") == "drift"]
        lease_warnings = [w for w in warnings if w.get("kind") == "active_lease"]
        self.assertTrue(len(drift_warnings) >= 1)
        self.assertTrue(len(lease_warnings) >= 1)
        self.assertIn("session-worker-9", lease_warnings[0]["reason"])

    def test_background_exclude_node_ids(self) -> None:
        """build_background excludes specified node IDs and preserves has_task_context."""
        # 1. Profile memory
        self.memory.append(
            category="profile",
            title="Developer Role",
            content="Backend systems engineer working on memory engines",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            memory_kind=MemoryKind.PROFILE,
        )

        # 2. Task memory
        task_id = self.memory.append(
            category="decision",
            title="Cache layer strategy",
            content="Use LRU cache with Fibonacci layering for index lookups",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Normal run
        bg_normal = build_background(
            self.memory,
            "Fibonacci cache layer strategy",
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        rendered_ids_normal = {m["node_id"] for m in bg_normal["memories"]}
        self.assertIn(task_id, rendered_ids_normal)
        self.assertTrue(bg_normal["has_task_context"])

        # Exclude task_id
        bg_excluded = build_background(
            self.memory,
            "Fibonacci cache layer strategy",
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
            exclude_node_ids={task_id},
        )
        rendered_ids_excluded = {m["node_id"] for m in bg_excluded["memories"]}
        self.assertNotIn(task_id, rendered_ids_excluded)
        # has_task_context should still be True because task context was matched
        self.assertTrue(bg_excluded["has_task_context"])

    def test_sync_skills_multi_host_and_manifest(self) -> None:
        """sync_skills validates host, writes manifest, and updates manifest upon auto-revocation."""
        skills_dir = self.cwd / "skills"

        proc = Procedure(
            trigger="Deploy verification check",
            steps=("Step 1: Check health endpoint", "Step 2: Validate database"),
            tools=("curl", "pytest"),
            verify="curl -f http://localhost:8080/health",
        )
        proc_node_id = self.memory.append(
            category="procedure",
            title="Deploy verification check",
            content=proc.encode(),
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            memory_kind=MemoryKind.PROCEDURE,
        )

        # Host validation check
        with self.assertRaises(ValueError) as ctx:
            sync_skills(
                self.memory,
                skills_dir,
                identity=self.identity,
                host="unsupported-host",
            )
        self.assertIn("host must be one of", str(ctx.exception))

        # Initial state: candidate (no outcomes) -> not materialized for min_maturity="verified"
        res_cand = sync_skills(
            self.memory,
            skills_dir,
            identity=self.identity,
            min_maturity="verified",
            host="agy",
            write_manifest=True,
        )
        self.assertEqual(len(res_cand["materialized"]), 0)
        manifest_file = Path(res_cand["manifest_path"])
        self.assertTrue(manifest_file.is_file())
        cand_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(cand_manifest["host"], "agy")
        self.assertEqual(cand_manifest["skills"], [])

        # Confirm procedure once -> maturity becomes "verified"
        self.memory.record_outcome(proc_node_id, Verdict.CONFIRMED, "smoke_test", session_id="session-1")

        # Materialize for host="agy"
        res_agy = sync_skills(
            self.memory,
            skills_dir,
            identity=self.identity,
            min_maturity="verified",
            host="agy",
            write_manifest=True,
        )
        self.assertEqual(len(res_agy["materialized"]), 1)
        self.assertEqual(res_agy["host"], "agy")
        self.assertTrue(manifest_file.is_file())

        agy_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(agy_manifest["host"], "agy")
        self.assertEqual(agy_manifest["min_maturity"], "verified")
        self.assertEqual(len(agy_manifest["skills"]), 1)

        skill_entry = agy_manifest["skills"][0]
        self.assertEqual(skill_entry["node_id"], proc_node_id)
        self.assertEqual(skill_entry["title"], "Deploy verification check")
        self.assertEqual(skill_entry["trigger"], proc.trigger)
        self.assertEqual(skill_entry["maturity"], "verified")
        self.assertEqual(skill_entry["verify"], proc.verify)
        self.assertTrue(Path(skill_entry["path"]).is_file())

        # Test write_manifest=False
        res_no_manifest = sync_skills(
            self.memory,
            skills_dir,
            identity=self.identity,
            min_maturity="verified",
            host="codex",
            write_manifest=False,
        )
        self.assertEqual(res_no_manifest["host"], "codex")
        self.assertIsNone(res_no_manifest["manifest_path"])

        # Refute the procedure -> auto-revocation unlinks skill file and updates manifest
        self.memory.record_outcome(proc_node_id, Verdict.REFUTED, "failed_in_staging", session_id="session-2")
        self.memory.record_outcome(proc_node_id, Verdict.REFUTED, "failed_in_staging_again", session_id="session-3")
        # Mark status refuted to ensure ineligible
        self.memory.set_status(proc_node_id, MemoryStatus.REFUTED, reason="failed in staging")

        res_revoked = sync_skills(
            self.memory,
            skills_dir,
            identity=self.identity,
            min_maturity="verified",
            host="claude-code",
            write_manifest=True,
        )
        self.assertEqual(len(res_revoked["revoked"]), 1)
        self.assertEqual(res_revoked["revoked"][0]["node_id"], proc_node_id)
        self.assertFalse(Path(skill_entry["path"]).exists())

        # Manifest should be updated and skills list empty
        revoked_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(revoked_manifest["host"], "claude-code")
        self.assertEqual(revoked_manifest["skills"], [])

    def test_compute_symbol_hashes_and_precision_precondition(self) -> None:
        """compute_symbol_hashes extracts AST hashes for functions, methods, and constants."""
        code_file = self.cwd / "module_a.py"
        code_file.write_text(
            (
                "GLOBAL_LIMIT = 100\n\n"
                "def top_level_func(x, y):\n"
                "    # Inside function comment\n"
                "    return x + y\n\n"
                "class DataHandler:\n"
                "    def process(self, item):\n"
                "        return item.upper()\n"
            ),
            encoding="utf-8",
        )

        specs = [
            "module_a.py::top_level_func",
            "module_a.py::DataHandler.process",
            "module_a.py::GLOBAL_LIMIT",
            "module_a.py::NonExistentSymbol",
            "missing_file.py::foo",
        ]
        hashes = compute_symbol_hashes(specs, cwd=self.cwd)

        self.assertIn("module_a.py::top_level_func", hashes)
        self.assertIn("module_a.py::DataHandler.process", hashes)
        self.assertIn("module_a.py::GLOBAL_LIMIT", hashes)
        self.assertNotIn("module_a.py::NonExistentSymbol", hashes)
        self.assertNotIn("missing_file.py::foo", hashes)
        self.assertEqual(len(hashes["module_a.py::top_level_func"]), 64)

    def test_symbol_drift_avoids_false_positives_and_detects_symbol_edits(self) -> None:
        """Editing comments/unrelated code does not trip drift; editing symbol triggers mismatch & transitive drift."""
        worker_file = self.cwd / "worker.py"
        worker_file.write_text(
            ("def compute_hash(val):\n    return str(val)\n\ndef unrelated_helper():\n    return 1\n"),
            encoding="utf-8",
        )

        initial_file_hashes = compute_file_hashes(["worker.py"], cwd=self.cwd)
        initial_sym_hashes = compute_symbol_hashes(["worker.py::compute_hash"], cwd=self.cwd)
        self.assertIn("worker.py::compute_hash", initial_sym_hashes)

        # Node depending on worker.py::compute_hash
        node_id = self.memory.append(
            category="decision",
            title="Worker compute hash invariant",
            content="Invariant for worker.py::compute_hash",
            metadata={
                "file_hashes": initial_file_hashes,
                "symbol_hashes": initial_sym_hashes,
            },
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        node = self.memory.nodes[node_id]

        # 1. Initial check: preconditions satisfied
        ok, reason = check_node_preconditions(node, cwd=self.cwd)
        self.assertTrue(ok)
        self.assertEqual(reason, "preconditions_satisfied")

        # 2. Modify unrelated function and add comments
        worker_file.write_text(
            (
                "# New header comment\n"
                "def compute_hash(val):\n"
                "    return str(val)\n\n"
                "def unrelated_helper():\n"
                "    return 999999  # Modified unrelated function!\n"
            ),
            encoding="utf-8",
        )

        # Whole-file hash has changed, but symbol_hashes is intact -> NO false positive!
        ok, reason = check_node_preconditions(node, cwd=self.cwd)
        self.assertTrue(ok)
        self.assertEqual(reason, "preconditions_satisfied")

        # Direct entity profile for symbol target should have 0 drift
        prof_ok = build_entity_profile(self.memory, "worker.py::compute_hash", identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(prof_ok["drift"]), 0)

        # 3. Modify compute_hash itself -> triggers symbol_hash_mismatch
        worker_file.write_text(
            (
                "def compute_hash(val):\n"
                "    return int(val)  # Changed implementation!\n\n"
                "def unrelated_helper():\n"
                "    return 1\n"
            ),
            encoding="utf-8",
        )

        ok, reason = check_node_preconditions(node, cwd=self.cwd)
        self.assertFalse(ok)
        self.assertEqual(reason, "symbol_hash_mismatch: worker.py::compute_hash")

        # Direct drift in entity profile
        prof_drift = build_entity_profile(self.memory, "worker.py::compute_hash", identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(prof_drift["drift"]), 1)
        self.assertEqual(prof_drift["drift"][0]["node_id"], node_id)
        self.assertEqual(prof_drift["drift"][0]["reason"], "symbol_hash_mismatch: worker.py::compute_hash")

        # 4. Transitive drift: consumer node linked via DERIVED_FROM
        consumer_node_id = self.memory.append(
            category="requirement",
            title="Consumer requirement for pipeline",
            content="Depends on worker compute hash",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        self.memory.link_nodes(consumer_node_id, node_id, RelationType.DERIVED_FROM)

        prof_consumer = build_entity_profile(self.memory, "pipeline.py", identity=self.identity, cwd=self.cwd)
        # We need consumer_node to be matched to pipeline.py
        self.memory.nodes[consumer_node_id].title = "Pipeline requirement pipeline.py"
        prof_consumer = build_entity_profile(self.memory, "pipeline.py", identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(prof_consumer["drift"]), 1)
        self.assertEqual(prof_consumer["drift"][0]["node_id"], consumer_node_id)
        self.assertEqual(prof_consumer["drift"][0]["via_node_id"], node_id)
        self.assertIn("symbol_hash_mismatch: worker.py::compute_hash", prof_consumer["drift"][0]["reason"])

    def test_sqlite_store_settings_support(self) -> None:
        """SqliteStore supports read_setting / write_setting for JSON-serializable structures."""
        db_path = self.cwd / "test_store.db"
        store = SqliteStore(db_path)

        store.write_setting("string_val", "active_mode")
        store.write_setting("dict_val", {"k1": [1, 2, 3], "enabled": True})
        store.write_setting("num_val", 42.5)

        self.assertEqual(store.read_setting("string_val"), "active_mode")
        self.assertEqual(store.read_setting("dict_val"), {"k1": [1, 2, 3], "enabled": True})
        self.assertEqual(store.read_setting("num_val"), 42.5)
        self.assertIsNone(store.read_setting("non_existent"))

    def test_auto_ground_metadata_auto_capture_and_shorthand(self) -> None:
        """auto_ground_metadata computes file and symbol hashes for explicit and shorthand specs."""
        self.assertIsNone(auto_ground_metadata(None, cwd=self.cwd))

        # Without cwd: return copy of metadata
        raw_meta = {"files": ["mod_a.py"]}
        self.assertEqual(auto_ground_metadata(raw_meta, cwd=None), raw_meta)

        # Write test modules
        mod_a_file = self.cwd / "mod_a.py"
        mod_a_file.write_text(
            ("def calculate_tax(amount):\n    return amount * 0.2\n"),
            encoding="utf-8",
        )
        mod_b_file = self.cwd / "mod_b.py"
        mod_b_file.write_text(
            ("class OrderService:\n    def process(self):\n        return True\n"),
            encoding="utf-8",
        )

        # Explicit spec: mod_a.py::calculate_tax
        meta_explicit = {
            "files": ["mod_a.py"],
            "symbols": ["mod_a.py::calculate_tax"],
        }
        grounded_explicit = auto_ground_metadata(meta_explicit, cwd=self.cwd)
        self.assertIsNotNone(grounded_explicit)
        self.assertIn("file_hashes", grounded_explicit)
        self.assertIn("mod_a.py", grounded_explicit["file_hashes"])
        self.assertIn("symbol_hashes", grounded_explicit)
        self.assertIn("mod_a.py::calculate_tax", grounded_explicit["symbol_hashes"])

        # Shorthand spec: "OrderService.process" paired with files=["mod_b.py"]
        meta_shorthand = {
            "files": ["mod_b.py"],
            "symbols": ["OrderService.process"],
        }
        grounded_shorthand = auto_ground_metadata(meta_shorthand, cwd=self.cwd)
        self.assertIsNotNone(grounded_shorthand)
        self.assertIn("file_hashes", grounded_shorthand)
        self.assertIn("mod_b.py", grounded_shorthand["file_hashes"])
        self.assertIn("symbol_hashes", grounded_shorthand)
        self.assertIn("mod_b.py::OrderService.process", grounded_shorthand["symbol_hashes"])

        # If file_hashes or symbol_hashes already present, do not overwrite
        existing_meta = {
            "files": ["mod_a.py"],
            "symbols": ["mod_a.py::calculate_tax"],
            "file_hashes": {"mod_a.py": "preset_file_hash"},
            "symbol_hashes": {"mod_a.py::calculate_tax": "preset_sym_hash"},
        }
        grounded_exist = auto_ground_metadata(existing_meta, cwd=self.cwd)
        self.assertIsNotNone(grounded_exist)
        self.assertEqual(grounded_exist["file_hashes"]["mod_a.py"], "preset_file_hash")
        self.assertEqual(grounded_exist["symbol_hashes"]["mod_a.py::calculate_tax"], "preset_sym_hash")

    def test_physical_ast_import_graph_drift_propagation(self) -> None:
        """Physical AST import graph extracts dependencies and propagates drift without memory edges."""
        storage_file = self.cwd / "storage_mod.py"
        storage_file.write_text(
            ("def query_all():\n    return ['item1', 'item2']\n"),
            encoding="utf-8",
        )
        service_file = self.cwd / "service_mod.py"
        service_file.write_text(
            ("import storage_mod\n\ndef serve():\n    return storage_mod.query_all()\n"),
            encoding="utf-8",
        )

        # 1. extract_python_imports
        imports = extract_python_imports("service_mod.py", self.cwd)
        self.assertEqual(imports, ["storage_mod.py"])

        # 2. Build profile when no memories exist
        prof_empty = build_entity_profile(self.memory, "service_mod.py", identity=self.identity, cwd=self.cwd)
        self.assertEqual(prof_empty["imported_files"], ["storage_mod.py"])
        self.assertEqual(len(prof_empty["drift"]), 0)

        # 3. Create upstream memory node for storage_mod.py
        h = compute_file_hashes(["storage_mod.py"], cwd=self.cwd)["storage_mod.py"]
        storage_node_id = self.memory.append(
            category="decision",
            title="Storage module storage_mod.py invariants",
            content="Maintain invariants for storage_mod.py",
            metadata={"file_hashes": {"storage_mod.py": h}},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Profile before drift: 0 drift
        prof_before = build_entity_profile(self.memory, "service_mod.py", identity=self.identity, cwd=self.cwd)
        self.assertEqual(prof_before["imported_files"], ["storage_mod.py"])
        self.assertEqual(len(prof_before["drift"]), 0)

        # 4. Mutate storage_mod.py to trigger drift
        storage_file.write_text(
            ("def query_all():\n    return ['drifted_item']\n"),
            encoding="utf-8",
        )

        # 5. Profile for service_mod.py should catch upstream drift via AST import graph
        # Notice: there is NO edge between service_mod and storage_mod memory node!
        prof_after = build_entity_profile(self.memory, "service_mod.py", identity=self.identity, cwd=self.cwd)
        self.assertEqual(prof_after["imported_files"], ["storage_mod.py"])
        self.assertEqual(len(prof_after["drift"]), 1)
        drift_entry = prof_after["drift"][0]
        self.assertEqual(drift_entry["node_id"], storage_node_id)
        self.assertEqual(drift_entry["via_file"], "storage_mod.py")
        self.assertTrue(
            drift_entry["reason"].startswith("import_drift via storage_mod.py: file_hash_mismatch: storage_mod.py")
        )

    def test_sqlite_store_incremental_event_fast_path_and_forget(self) -> None:
        """SqliteStore handles incremental event log fast path and forget redaction fallback."""
        db_path = self.cwd / "incremental_events.db"
        store = SqliteStore(db_path)

        # 1. Initial transaction: append nodes
        with store.transaction() as mem:
            n1 = mem.append(
                category="profile",
                title="Profile node 1",
                content="Owner secret preference",
                owner="abbila",
                workspace_id="fibmind",
                project_id="fibmind",
            )
            n2 = mem.append(
                category="decision",
                title="Architecture decision 2",
                content="Architecture details",
                owner="abbila",
                workspace_id="fibmind",
                project_id="fibmind",
            )
            count_1 = len(mem.events)
            self.assertGreater(count_1, 0)

        # Reload and verify events match
        loaded = store.load()
        self.assertIn(n1, loaded.nodes)
        self.assertIn(n2, loaded.nodes)
        self.assertEqual(len(loaded.events), count_1)

        # 2. Incremental append transaction (fast path O(Delta E))
        with store.transaction() as mem:
            n3 = mem.append(
                category="requirement",
                title="Requirement node 3",
                content="Fast path verification",
                owner="abbila",
                workspace_id="fibmind",
                project_id="fibmind",
            )
            count_2 = len(mem.events)
            self.assertGreater(count_2, count_1)

        loaded2 = store.load()
        self.assertIn(n3, loaded2.nodes)
        self.assertEqual(len(loaded2.events), count_2)

        # 3. Forget transaction (redaction fallback: deleted_nodes triggers full check)
        with store.transaction() as mem:
            mem.forget(n1, reason="obsolete profile")
            self.assertNotIn(n1, mem.nodes)

        # Reload after forget: node 1 is gone, events redacted
        loaded3 = store.load()
        self.assertNotIn(n1, loaded3.nodes)
        self.assertIn(n2, loaded3.nodes)
        self.assertIn(n3, loaded3.nodes)

        # Verify historical events for n1 have been redacted on disk
        for ev in loaded3.events:
            payload = ev.payload
            if isinstance(payload, dict):
                node_data = payload.get("node")
                if isinstance(node_data, dict) and node_data.get("id") == n1:
                    self.assertEqual(node_data.get("content"), "[forgotten]")

        # 4. Subsequent incremental append continues working
        with store.transaction() as mem:
            n4 = mem.append(
                category="decision",
                title="Post-forget decision 4",
                content="Continued normal operation",
                owner="abbila",
                workspace_id="fibmind",
                project_id="fibmind",
            )
        loaded4 = store.load()
        self.assertIn(n4, loaded4.nodes)
        self.assertIn(n2, loaded4.nodes)
        self.assertIn(n3, loaded4.nodes)
        self.assertNotIn(n1, loaded4.nodes)

    def test_polyglot_symbol_hashing_ts_and_go(self) -> None:
        """Polyglot symbol hashing for TS and Go ignores comments and detects function body drift."""
        # 1. TypeScript test file
        ts_file = self.cwd / "api.ts"
        ts_file.write_text(
            (
                "// Header comment\n"
                "export function handleRequest(reqId: string): string {\n"
                "    // Internal comment\n"
                "    return 'res-' + reqId;\n"
                "}\n\n"
                "export function unrelatedHelper(): number {\n"
                "    return 42;\n"
                "}\n"
            ),
            encoding="utf-8",
        )

        h_ts = compute_symbol_hashes(["api.ts::handleRequest"], cwd=self.cwd)["api.ts::handleRequest"]
        node_ts_id = self.memory.append(
            category="decision",
            title="TypeScript API handler",
            content="Handle request specification",
            metadata={"symbol_hashes": {"api.ts::handleRequest": h_ts}},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        node_ts = self.memory.nodes[node_ts_id]
        ok, reason = check_node_preconditions(node_ts, cwd=self.cwd)
        self.assertTrue(ok)
        self.assertEqual(reason, "preconditions_satisfied")

        # Modifying comments and unrelated function does not drift handleRequest
        ts_file.write_text(
            (
                "// Completely new header comment\n"
                "/* Multi-line block comment explaining nothing */\n"
                "export function handleRequest(reqId: string): string {\n"
                "    // Different internal comment\n"
                "    return 'res-' + reqId;\n"
                "}\n\n"
                "export function unrelatedHelper(): number {\n"
                "    return 999; // Changed unrelated helper\n"
                "}\n"
            ),
            encoding="utf-8",
        )
        ok, reason = check_node_preconditions(node_ts, cwd=self.cwd)
        self.assertTrue(ok)
        self.assertEqual(reason, "preconditions_satisfied")

        # Modifying handleRequest body triggers symbol_hash_mismatch
        ts_file.write_text(
            ("export function handleRequest(reqId: string): string {\n    return 'drifted-' + reqId;\n}\n"),
            encoding="utf-8",
        )
        ok, reason = check_node_preconditions(node_ts, cwd=self.cwd)
        self.assertFalse(ok)
        self.assertEqual(reason, "symbol_hash_mismatch: api.ts::handleRequest")

        # 2. Go test file with receiver method
        go_file = self.cwd / "server.go"
        go_file.write_text(
            (
                "package main\n\n"
                "// Server struct doc\n"
                "type Server struct {\n"
                "    port int\n"
                "}\n\n"
                "func (s *Server) Start() error {\n"
                "    // Starting server\n"
                "    return nil\n"
                "}\n\n"
                "func (s *Server) Stop() {\n"
                "    // Stopping server\n"
                "}\n"
            ),
            encoding="utf-8",
        )

        h_go = compute_symbol_hashes(["server.go::Server.Start"], cwd=self.cwd)["server.go::Server.Start"]
        node_go_id = self.memory.append(
            category="decision",
            title="Go server runner",
            content="Go server lifecycle specification",
            metadata={"symbol_hashes": {"server.go::Server.Start": h_go}},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        node_go = self.memory.nodes[node_go_id]
        ok_go, reason_go = check_node_preconditions(node_go, cwd=self.cwd)
        self.assertTrue(ok_go)
        self.assertEqual(reason_go, "preconditions_satisfied")

        # Modifying comments and Stop() does not drift Server.Start
        go_file.write_text(
            (
                "package main\n\n"
                "// Changed doc comment\n"
                "type Server struct {\n"
                "    port int\n"
                "}\n\n"
                "func (s *Server) Start() error {\n"
                "    // New comment in Start\n"
                "    return nil\n"
                "}\n\n"
                "func (s *Server) Stop() {\n"
                '    println("Stop modified")\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        ok_go, reason_go = check_node_preconditions(node_go, cwd=self.cwd)
        self.assertTrue(ok_go)
        self.assertEqual(reason_go, "preconditions_satisfied")

        # Modifying Server.Start body triggers symbol_hash_mismatch
        go_file.write_text(
            (
                "package main\n\n"
                "type Server struct {\n"
                "    port int\n"
                "}\n\n"
                "func (s *Server) Start() error {\n"
                '    return errors.New("failed to start")\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        ok_go, reason_go = check_node_preconditions(node_go, cwd=self.cwd)
        self.assertFalse(ok_go)
        self.assertEqual(reason_go, "symbol_hash_mismatch: server.go::Server.Start")

    def test_post_verification_auto_regrounding(self) -> None:
        """run_command_and_reflect automatically refreshes hashes and clears drift on confirmed verification."""
        worker_file = self.cwd / "worker.py"
        worker_file.write_text(
            ("def compute_hash(val):\n    return val + '_v1'\n"),
            encoding="utf-8",
        )

        db_path = self.cwd / "reground.db"
        service = MemoryService(db_path)
        brain = FibBrain(service)

        h_file = compute_file_hashes(["worker.py"], cwd=self.cwd)["worker.py"]
        h_sym = compute_symbol_hashes(["worker.py::compute_hash"], cwd=self.cwd)["worker.py::compute_hash"]

        node_res = service.append(
            category="decision",
            title="Worker task worker.py::compute_hash",
            content="Worker hash calculation logic",
            metadata={
                "files": ["worker.py"],
                "symbols": ["worker.py::compute_hash"],
                "file_hashes": {"worker.py": h_file},
                "symbol_hashes": {"worker.py::compute_hash": h_sym},
                "drift_detected": True,
            },
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        node_id = node_res["node_id"]

        # Modify worker.py to cause symbol_hash_mismatch
        worker_file.write_text(
            ("def compute_hash(val):\n    return val + '_v2_drifted'\n"),
            encoding="utf-8",
        )

        node_before = service.read(lambda m: m.nodes[node_id])
        ok_before, reason_before = check_node_preconditions(node_before, cwd=self.cwd)
        self.assertFalse(ok_before)
        self.assertEqual(reason_before, "symbol_hash_mismatch: worker.py::compute_hash")
        self.assertTrue(node_before.metadata.get("drift_detected"))

        # Run verification command that touches worker.py and succeeds
        res = run_command_and_reflect(
            ["echo", "PASSED worker.py"],
            node_id=node_id,
            cwd=self.cwd,
            service=service,
            brain=brain,
        )
        self.assertEqual(res["verdict"], "confirmed")
        self.assertTrue(res["attribution"]["regrounded"])

        # Reload node and verify preconditions are satisfied and drift_detected is cleared
        node_reloaded = service.read(lambda m: m.nodes[node_id])
        ok_after, reason_after = check_node_preconditions(node_reloaded, cwd=self.cwd)
        self.assertTrue(ok_after)
        self.assertEqual(reason_after, "preconditions_satisfied")
        self.assertNotIn("drift_detected", node_reloaded.metadata)

        # Hash has been refreshed to match the new code on disk
        new_h_sym = compute_symbol_hashes(["worker.py::compute_hash"], cwd=self.cwd)["worker.py::compute_hash"]
        self.assertEqual(node_reloaded.metadata["symbol_hashes"]["worker.py::compute_hash"], new_h_sym)

    def test_locate_symbol_spans_and_entity_profile_projection(self) -> None:
        """locate_symbol_spans provides accurate line spans and signatures across languages."""
        # 1. Python file
        calc_file = self.cwd / "calc.py"
        calc_file.write_text(
            (
                "# Header\n"
                "def add(a: int, b: int) -> int:\n"
                "    return a + b\n\n"
                "class MathService:\n"
                "    def multiply(self, x: int, y: int) -> int:\n"
                "        return x * y\n"
            ),
            encoding="utf-8",
        )

        spans_py = locate_symbol_spans(["calc.py::add", "calc.py::MathService.multiply"], cwd=self.cwd)
        self.assertIn("calc.py::add", spans_py)
        self.assertEqual(spans_py["calc.py::add"]["file"], "calc.py")
        self.assertEqual(spans_py["calc.py::add"]["symbol"], "add")
        self.assertEqual(spans_py["calc.py::add"]["start_line"], 2)
        self.assertEqual(spans_py["calc.py::add"]["end_line"], 3)
        self.assertEqual(spans_py["calc.py::add"]["signature"], "def add(a: int, b: int) -> int:")
        self.assertTrue(bool(spans_py["calc.py::add"]["sha256"]))

        self.assertIn("calc.py::MathService.multiply", spans_py)
        self.assertEqual(spans_py["calc.py::MathService.multiply"]["start_line"], 6)
        self.assertEqual(spans_py["calc.py::MathService.multiply"]["end_line"], 7)
        self.assertEqual(
            spans_py["calc.py::MathService.multiply"]["signature"], "def multiply(self, x: int, y: int) -> int:"
        )

        # 2. TypeScript file
        ts_file = self.cwd / "utils.ts"
        ts_file.write_text(
            (
                "// Header comment\n"
                "/* Block comment */\n"
                "export function formatName(first: string, last: string): string {\n"
                "    return first + ' ' + last;\n"
                "}\n"
            ),
            encoding="utf-8",
        )
        spans_ts = locate_symbol_spans(["utils.ts::formatName"], cwd=self.cwd)
        self.assertIn("utils.ts::formatName", spans_ts)
        self.assertEqual(spans_ts["utils.ts::formatName"]["start_line"], 3)
        self.assertEqual(spans_ts["utils.ts::formatName"]["end_line"], 5)
        self.assertIn("export function formatName", spans_ts["utils.ts::formatName"]["signature"])

        # 3. Go file
        go_file = self.cwd / "handler.go"
        go_file.write_text(
            ('package main\n\ntype App struct {}\n\nfunc (a *App) Handle() {\n    println("handled")\n}\n'),
            encoding="utf-8",
        )
        spans_go = locate_symbol_spans(["handler.go::App.Handle"], cwd=self.cwd)
        self.assertIn("handler.go::App.Handle", spans_go)
        self.assertEqual(spans_go["handler.go::App.Handle"]["start_line"], 5)
        self.assertEqual(spans_go["handler.go::App.Handle"]["end_line"], 7)
        self.assertIn("func (a *App) Handle()", spans_go["handler.go::App.Handle"]["signature"])

        # 4. Profile integration: candidate symbols in matched nodes are localized
        self.memory.append(
            category="decision",
            title="Calculator logic calc.py",
            content="Maintain add function invariants",
            metadata={"symbols": ["add"], "files": ["calc.py"]},
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        prof = build_entity_profile(self.memory, "calc.py", identity=self.identity, cwd=self.cwd)
        self.assertIn("symbol_spans", prof)
        self.assertIn("calc.py::add", prof["symbol_spans"])
        self.assertEqual(prof["symbol_spans"]["calc.py::add"]["start_line"], 2)

    def test_bidirectional_skill_parse_and_ingest_modified(self) -> None:
        """sync_skills ingests user-modified SKILL.md, updates graph, and survives event replay."""
        nid = self.memory.append(
            category="procedure",
            title="Deploy App",
            content=Procedure(
                trigger="Deploy the web application",
                steps=("Run tests", "Build Docker image", "Push image"),
                verify="kubectl rollout status deployment/web",
                when=("release", "production"),
                tools=("kubectl", "docker"),
                inputs={"env": "target environment"},
            ).encode(),
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        self.memory.record_outcome(nid, Verdict.CONFIRMED, source="test", session_id="session-main")
        self.memory.record_outcome(nid, Verdict.CONFIRMED, source="test", session_id="session-2")
        self.memory.record_outcome(nid, Verdict.CONFIRMED, source="test", session_id="session-3")

        skill_dir = self.cwd / "skills"
        res1 = sync_skills(self.memory, skill_dir, identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(res1["materialized"]), 1)
        skill_file = Path(res1["materialized"][0]["path"])
        self.assertTrue(skill_file.exists())

        # Test parse_skill_markdown
        parsed = parse_skill_markdown(skill_file.read_text(encoding="utf-8"))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["node_id"], nid)
        self.assertEqual(parsed["title"], "Deploy App")
        self.assertEqual(parsed["procedure"].steps, ("Run tests", "Build Docker image", "Push image"))
        self.assertEqual(parsed["procedure"].verify, "kubectl rollout status deployment/web")

        # Modify on-disk SKILL.md (add step 4, update verify command)
        modified_text = (
            skill_file.read_text(encoding="utf-8")
            .replace(
                "3. Push image\n",
                "3. Push image\n4. Notify Slack\n",
            )
            .replace(
                "kubectl rollout status deployment/web",
                "curl -f https://app.example.com/health",
            )
        )
        skill_file.write_text(modified_text, encoding="utf-8")

        # Ingest modified skill
        res2 = sync_skills(self.memory, skill_dir, identity=self.identity, cwd=self.cwd, ingest_modified=True)
        self.assertEqual(len(res2["ingested"]), 1)
        self.assertEqual(res2["ingested"][0]["node_id"], nid)

        # In-memory node updated
        updated_proc = Procedure.parse(self.memory.nodes[nid].content)
        self.assertIsNotNone(updated_proc)
        self.assertEqual(len(updated_proc.steps), 4)
        self.assertEqual(updated_proc.steps[3], "Notify Slack")
        self.assertEqual(updated_proc.verify, "curl -f https://app.example.com/health")

        # Event log replay preserves the ingested procedure
        rebuilt = FibMind.rebuild_from_log(self.memory.events)
        rebuilt_proc = Procedure.parse(rebuilt.nodes[nid].content)
        self.assertIsNotNone(rebuilt_proc)
        self.assertEqual(rebuilt_proc.steps, updated_proc.steps)
        self.assertEqual(rebuilt_proc.verify, "curl -f https://app.example.com/health")

    def test_resolve_git_commit_and_auto_ground_temporal_anchor(self) -> None:
        """Resolve git commits from loose refs, packed-refs, and detached HEAD."""
        git_dir = self.cwd / ".git"
        git_dir.mkdir(parents=True, exist_ok=True)
        (git_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)

        # 1. Loose ref
        (git_dir / "HEAD").write_text("ref: refs/heads/feature-track2\n", encoding="utf-8")
        (git_dir / "refs" / "heads" / "feature-track2").write_text(
            "406fa9ad525a6781b312fc8d0000000000000001\n", encoding="utf-8"
        )
        self.assertEqual(resolve_git_commit(self.cwd), "406fa9ad525a")

        # 2. Packed refs
        (git_dir / "refs" / "heads" / "feature-track2").unlink()
        (git_dir / "packed-refs").write_text(
            "# pack-refs with: peeled-tags inline\n"
            "e09fa9ad525a6781b312fc8d0000000000000002 refs/heads/feature-track2\n"
            "^e09fa9ad525a6781b312fc8d0000000000000002\n",
            encoding="utf-8",
        )
        self.assertEqual(resolve_git_commit(self.cwd), "e09fa9ad525a")

        # 3. Detached HEAD
        (git_dir / "HEAD").write_text("d82fa9ad525a6781b312fc8d0000000000000003\n", encoding="utf-8")
        self.assertEqual(resolve_git_commit(self.cwd), "d82fa9ad525a")

        # 4. Non-git directory returns None
        with tempfile.TemporaryDirectory() as isolated:
            self.assertIsNone(resolve_git_commit(isolated))

        # 5. working_context retains exact 3 keys
        ctx = working_context(str(self.cwd))
        self.assertEqual(set(ctx.keys()), {"workspace_id", "project_id", "task_id"})
        self.assertEqual(ctx["task_id"], "detached:d82fa9ad525a")

        # 6. auto_ground_metadata captures git_commit
        meta = auto_ground_metadata({"title": "Temporal anchor"}, cwd=self.cwd)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("git_commit"), "d82fa9ad525a")

        # 7. auto_ground_metadata preserves existing git_commit
        meta_existing = auto_ground_metadata({"git_commit": "pinnedcommit12"}, cwd=self.cwd)
        self.assertIsNotNone(meta_existing)
        self.assertEqual(meta_existing.get("git_commit"), "pinnedcommit12")

    def test_polyglot_code_imports_and_drift_propagation(self) -> None:
        """extract_code_imports handles TS/JS, Rust, and Go with physical drift propagation."""
        # Setup TypeScript structure
        (self.cwd / "config.ts").write_text("export const APP_PORT = 8080;\n", encoding="utf-8")
        src_dir = self.cwd / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "utils.ts").write_text("export function format(s: string) { return s; }\n", encoding="utf-8")
        (src_dir / "index.ts").write_text(
            'import { format } from "./utils";\n'
            'import { APP_PORT } from "../config";\n'
            'export { format } from "./utils";\n'
            'const legacy = require("./utils");\n',
            encoding="utf-8",
        )
        ts_imports = extract_code_imports("src/index.ts", self.cwd)
        self.assertEqual(ts_imports, ["config.ts", "src/utils.ts"])

        # Setup Rust structure
        (src_dir / "engine.rs").write_text("pub struct Engine;\n", encoding="utf-8")
        (src_dir / "models.rs").write_text("pub struct User;\n", encoding="utf-8")
        (src_dir / "main.rs").write_text(
            "pub mod engine;\nuse crate::models::User;\nfn main() {}\n",
            encoding="utf-8",
        )
        rs_imports = extract_code_imports("src/main.rs", self.cwd)
        self.assertEqual(rs_imports, ["src/engine.rs", "src/models.rs"])

        # Setup Go structure
        pkg_store = self.cwd / "pkg" / "store"
        pkg_store.mkdir(parents=True, exist_ok=True)
        (pkg_store / "store.go").write_text("package store\n", encoding="utf-8")
        internal_dir = self.cwd / "internal"
        internal_dir.mkdir(parents=True, exist_ok=True)
        (internal_dir / "sub.go").write_text("package internal\n", encoding="utf-8")
        (self.cwd / "main.go").write_text(
            'package main\nimport (\n    "example.com/company/repo/pkg/store"\n    "./internal"\n)\n',
            encoding="utf-8",
        )
        go_imports = extract_code_imports("main.go", self.cwd)
        self.assertEqual(go_imports, ["internal/sub.go", "pkg/store/store.go"])

        # Physical AST import-graph drift propagation test via Step 6b
        web_dir = self.cwd / "web"
        web_components = web_dir / "components"
        web_components.mkdir(parents=True, exist_ok=True)
        btn_file = web_components / "button.tsx"
        btn_file.write_text("export const Button = () => <button>Click</button>;\n", encoding="utf-8")
        ui_file = web_dir / "ui.tsx"
        ui_file.write_text('import { Button } from "./components/button";\n', encoding="utf-8")

        initial_hashes = compute_file_hashes(["web/components/button.tsx"], cwd=self.cwd)
        self.assertIn("web/components/button.tsx", initial_hashes)

        # Upstream node bound to web/components/button.tsx with initial hash
        upstream_id = self.memory.append(
            category="decision",
            title="Button component invariants",
            content="Maintain Button styles for web/components/button.tsx",
            metadata={"file_hashes": initial_hashes},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Downstream node targeting web/ui.tsx
        self.memory.append(
            category="decision",
            title="UI page view web/ui.tsx",
            content="Page layout in web/ui.tsx",
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Before mutation: no drift on web/ui.tsx
        prof_clean = build_entity_profile(self.memory, "web/ui.tsx", identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(prof_clean["drift"]), 0)

        # Mutate button.tsx on disk
        btn_file.write_text("export const Button = () => <button>Mutated</button>;\n", encoding="utf-8")

        # After mutation: build_entity_profile for web/ui.tsx discovers upstream drift via import graph
        prof_drift = build_entity_profile(self.memory, "web/ui.tsx", identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(prof_drift["drift"]), 1)
        self.assertEqual(prof_drift["drift"][0]["node_id"], upstream_id)
        self.assertTrue(prof_drift["drift"][0]["reason"].startswith("import_drift via web/components/button.tsx"))

    def test_review_digest_and_candidates_extract_symbols(self) -> None:
        """Episode digest and review candidates extract and propagate symbols."""
        self.assertIn("symbols", SYMBOL_KEYS)
        self.assertIn("classes", SYMBOL_KEYS)

        # Create mock episode nodes with diverse symbol keys
        decision_node = MemoryNode(
            id="ep-dec-1",
            category="episode",
            title="Use LRU cache",
            content="Decided to use LRU cache for memory lookups",
            created_at=100.0,
            metadata={
                "kind": "decision",
                "payload": {
                    "symbols": ["LRUCache", "get_item"],
                    "files": ["src/cache.py"],
                },
            },
        )
        error_node = MemoryNode(
            id="ep-err-1",
            category="episode",
            title="Index out of bounds error",
            content="IndexError when retrieving negative key",
            created_at=101.0,
            metadata={
                "kind": "error",
                "payload": {
                    "symbol": "retrieve_key",
                    "files": ["src/cache.py"],
                },
            },
        )
        test_node = MemoryNode(
            id="ep-test-1",
            category="episode",
            title="Test LRU eviction",
            content="test_lru_eviction passes",
            created_at=102.0,
            metadata={
                "kind": "test",
                "payload": {
                    "command": "pytest tests/test_cache.py",
                    "functions": ["test_lru_eviction"],
                    "files": ["tests/test_cache.py"],
                },
            },
        )
        risk_node = MemoryNode(
            id="ep-risk-1",
            category="episode",
            title="Thread safety risk",
            content="Cache is not yet thread safe",
            created_at=103.0,
            metadata={
                "kind": "risk",
                "payload": {
                    "classes": ["ThreadSafeCache"],
                },
            },
        )

        episodes = [decision_node, error_node, test_node, risk_node]
        digest = digest_episode("session-track2", episodes)
        self.assertIsInstance(digest, EpisodeDigest)

        # Verify symbols in EpisodeDigest
        self.assertIn("LRUCache", digest.symbols)
        self.assertIn("get_item", digest.symbols)
        self.assertIn("retrieve_key", digest.symbols)
        self.assertIn("test_lru_eviction", digest.symbols)
        self.assertIn("ThreadSafeCache", digest.symbols)

        # Verify candidate extraction attaches symbols to metadata
        candidates = extract_candidates(digest)
        by_cat = {c.category: c for c in candidates}

        self.assertIn("decision", by_cat)
        self.assertEqual(by_cat["decision"].metadata.get("symbols"), ["LRUCache", "get_item"])

        self.assertIn("error", by_cat)
        self.assertEqual(by_cat["error"].metadata.get("symbols"), ["retrieve_key"])

        self.assertIn("verification", by_cat)
        self.assertEqual(by_cat["verification"].metadata.get("symbols"), ["test_lru_eviction"])

        self.assertIn("risk", by_cat)
        self.assertEqual(by_cat["risk"].metadata.get("symbols"), ["ThreadSafeCache"])

        self.assertIn("summary", by_cat)
        # Summary inherits symbols via _node_anchors_meta fallback
        self.assertTrue(bool(by_cat["summary"].metadata.get("symbols")))

        # Test _node_anchors_meta fallback behavior directly
        meta_with_fallback = _node_anchors_meta([], fallback_symbols=["global_helper"])
        self.assertEqual(meta_with_fallback.get("symbols"), ["global_helper"])

    def test_two_hop_transitive_code_imports_and_drift(self) -> None:
        """resolve_transitive_imports resolves 2-hop imports and surfaces transitive drift."""
        src_dir = self.cwd / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        c_file = src_dir / "c_leaf.py"
        c_file.write_text("def calculate_tax(amount: float) -> float:\n    return amount * 0.2\n", encoding="utf-8")

        b_file = src_dir / "b_mid.py"
        b_file.write_text(
            "from src.c_leaf import calculate_tax\n\n"
            "def process_invoice(amount: float) -> float:\n"
            "    return calculate_tax(amount)\n",
            encoding="utf-8",
        )

        a_file = src_dir / "a_root.py"
        a_file.write_text(
            "from src.b_mid import process_invoice\n\ndef main_handler():\n    return process_invoice(100.0)\n",
            encoding="utf-8",
        )

        # 1. Test resolve_transitive_imports
        transitive_list = resolve_transitive_imports("src/a_root.py", self.cwd, max_hops=2)
        self.assertEqual(len(transitive_list), 2)
        self.assertEqual(transitive_list[0], {"file": "src/b_mid.py", "hop": 1, "via": None})
        self.assertEqual(transitive_list[1], {"file": "src/c_leaf.py", "hop": 2, "via": "src/b_mid.py"})

        # 2. Append memory node anchored to c_leaf symbol calculate_tax
        initial_sym_hashes = compute_symbol_hashes(["src/c_leaf.py::calculate_tax"], cwd=self.cwd)
        self.assertIn("src/c_leaf.py::calculate_tax", initial_sym_hashes)

        leaf_node_id = self.memory.append(
            category="decision",
            title="Tax calculation invariant",
            content="Tax rate is 20% in src/c_leaf.py::calculate_tax",
            metadata={"symbol_hashes": initial_sym_hashes},
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Before mutation: 0 drift on a_root.py
        prof_before = build_entity_profile(self.memory, "src/a_root.py", identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(prof_before["drift"]), 0)
        self.assertEqual(prof_before["imported_files"], ["src/b_mid.py"])
        self.assertEqual(len(prof_before["transitive_imports"]), 2)

        # Mutate calculate_tax in c_leaf.py
        c_file.write_text("def calculate_tax(amount: float) -> float:\n    return amount * 0.25\n", encoding="utf-8")

        # After mutation: build_entity_profile for a_root.py discovers 2-hop drift via b_mid.py
        prof_after = build_entity_profile(self.memory, "src/a_root.py", identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(prof_after["drift"]), 1)
        drift_item = prof_after["drift"][0]
        self.assertEqual(drift_item["node_id"], leaf_node_id)
        self.assertEqual(drift_item["hop"], 2)
        self.assertEqual(drift_item["via"], "src/b_mid.py")
        self.assertEqual(drift_item["via_file"], "src/c_leaf.py")
        self.assertTrue(drift_item["reason"].startswith("import_drift via src/c_leaf.py: symbol_hash_mismatch"))

    def test_temporal_commit_drift_diagnostics_in_entity_profile(self) -> None:
        """diagnose_temporal_drift detects commit hash changes and attaches to drift warnings."""
        git_dir = self.cwd / ".git"
        git_dir.mkdir(parents=True, exist_ok=True)
        (git_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)

        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git_dir / "refs" / "heads" / "main").write_text("1111222233334444555566667777888899990000\n", encoding="utf-8")

        schema_file = self.cwd / "src" / "schema.py"
        schema_file.parent.mkdir(parents=True, exist_ok=True)
        schema_file.write_text("VERSION = 1\n", encoding="utf-8")

        initial_hashes = compute_file_hashes(["src/schema.py"], cwd=self.cwd)
        node_id = self.memory.append(
            category="decision",
            title="Database schema definition",
            content="Schema in src/schema.py",
            metadata={
                "file_hashes": initial_hashes,
                "git_commit": "111122223333",
            },
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        # Before commit advance: no temporal drift
        diag_none = diagnose_temporal_drift(self.memory.nodes[node_id], cwd=self.cwd)
        self.assertIsNone(diag_none)

        # Advance commit in git
        (git_dir / "refs" / "heads" / "main").write_text("aaaa9999bbbb8888cccc7777dddd6666eeee5555\n", encoding="utf-8")

        # Now temporal drift is reported
        diag = diagnose_temporal_drift(self.memory.nodes[node_id], cwd=self.cwd)
        self.assertIsNotNone(diag)
        self.assertEqual(diag["anchored_commit"], "111122223333")
        self.assertEqual(diag["current_commit"], "aaaa9999bbbb")
        self.assertTrue(diag["commit_changed"])

        # Mutate schema.py so physical drift occurs
        schema_file.write_text("VERSION = 2\n", encoding="utf-8")

        # build_entity_profile includes anchored_commit and current_commit in drift warning
        prof = build_entity_profile(self.memory, "src/schema.py", identity=self.identity, cwd=self.cwd)
        self.assertEqual(len(prof["drift"]), 1)
        warning = prof["drift"][0]
        self.assertEqual(warning["node_id"], node_id)
        self.assertEqual(warning.get("anchored_commit"), "111122223333")
        self.assertEqual(warning.get("current_commit"), "aaaa9999bbbb")

    def test_revalidate_memories_drift_recovery_and_command_execution(self) -> None:
        """revalidate_memories recovers restored memories and executes verify commands."""
        db_path = self.cwd / "reval.db"
        service = MemoryService(db_path)
        brain = FibBrain(service)
        state = BrainState(
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )

        svc_file = self.cwd / "pkg" / "service.py"
        svc_file.parent.mkdir(parents=True, exist_ok=True)
        svc_file.write_text("def run():\n    return 42\n", encoding="utf-8")

        initial_hashes = compute_file_hashes(["pkg/service.py"], cwd=self.cwd)

        node_res = service.append(
            category="decision",
            title="Service runner invariant",
            content="Service run returns 42 in pkg/service.py",
            metadata={
                "files": ["pkg/service.py"],
                "file_hashes": initial_hashes,
                "command": "echo 'PASSED pkg/service.py'",
                "drift_detected": "file_hash_mismatch",
            },
            owner="abbila",
            workspace_id="fibmind",
            project_id="fibmind",
            session_id="session-main",
            task_id="track2",
        )
        node_id = node_res["node_id"]

        # 1. Mutate file: precondition still fails
        svc_file.write_text("def run():\n    return 999\n", encoding="utf-8")
        res1 = revalidate_memories(brain, state=state, cwd=self.cwd, execute_commands=False)
        self.assertGreaterEqual(res1["checked"], 1)
        self.assertEqual(res1["recovered"], [])
        self.assertEqual(len(res1["still_drifted"]), 1)
        self.assertEqual(res1["still_drifted"][0]["node_id"], node_id)
        node_mid = service.read(lambda m: m.nodes[node_id])
        self.assertIn("drift_detected", node_mid.metadata)

        # 2. Restore file to original content: preconditions pass, drift cleared, command executed
        svc_file.write_text("def run():\n    return 42\n", encoding="utf-8")
        res2 = revalidate_memories(brain, state=state, cwd=self.cwd, execute_commands=True)
        self.assertIn(node_id, res2["recovered"])
        self.assertEqual(res2["still_drifted"], [])
        node_recovered = service.read(lambda m: m.nodes[node_id])
        self.assertNotIn("drift_detected", node_recovered.metadata)

        self.assertEqual(len(res2["executed"]), 1)
        exec_item = res2["executed"][0]
        self.assertEqual(exec_item["node_id"], node_id)
        self.assertEqual(exec_item["verdict"], "confirmed")
        self.assertEqual(exec_item["exit_code"], 0)

        # 3. Verify event log replay preserves drift recovery
        loaded_mem = service.read(lambda m: m)
        rebuilt = FibMind.rebuild_from_log(loaded_mem.events)
        rebuilt_node = rebuilt.nodes[node_id]
        self.assertNotIn("drift_detected", rebuilt_node.metadata or {})


if __name__ == "__main__":
    unittest.main()
