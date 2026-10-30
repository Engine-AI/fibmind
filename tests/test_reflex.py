"""Unit tests for System 1 Fast Reflex Engine, 4-Layer Anti-Degradation Audit,
Shadow Revision protection, and Cross-Host Near-Duplicate Admission.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from fibmind import (
    AdmitVerdict,
    BrainState,
    ConflictError,
    FibBrain,
    MemoryService,
    MemoryStatus,
    compute_file_hashes,
)


def _state(**overrides: str) -> BrainState:
    base = {
        "owner": "abbila",
        "workspace_id": "fibmind",
        "project_id": "fibmind",
        "session_id": "sess-1",
    }
    base.update(overrides)
    return BrainState(**base)


class ReflexAndAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = MemoryService(self.root / "memory.db")
        self.brain = FibBrain(self.service)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reflex_direct_answer_bypasses_llm_for_verified_memory(self) -> None:
        target_file = self.root / "src" / "storage.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text("pragma = 'WAL'\n", encoding="utf-8")

        hashes = compute_file_hashes(["src/storage.py"], cwd=self.root)
        written = self.brain.remember(
            "decision",
            "SQLite WAL concurrency configuration",
            (
                "Always enable PRAGMA journal_mode=WAL and busy_timeout=5000 on SQLite connections.\n"
                "When opening local SQLite stores across multiple processes.\n"
                "Avoid bare sqlite3.connect() without timeout as concurrent writers will fail."
            ),
            metadata={
                "files": ["src/storage.py"],
                "file_hashes": hashes,
                "conditions": ["SQLite >= 3.35"],
                "exceptions": ["Do not use on network NFS mounts"],
            },
            state=_state(),
        )
        node_id = written["node_id"]

        # Unverified memory (confidence 0.0) should NOT trigger System 1 direct answer
        unverified = self.brain.reflex(
            "how to configure SQLite WAL concurrency",
            state=_state(),
            min_confidence=0.2,
            cwd=self.root,
        )
        self.assertFalse(unverified["bypass_llm"])
        self.assertEqual(unverified["mode"], "llm_fallback")

        # Confirm the memory so confidence rises above threshold
        self.brain.reflect("confirmed", "pytest tests/test_storage.py", node_id=node_id, state=_state())

        reflex = self.brain.reflex(
            "how to configure SQLite WAL concurrency",
            state=_state(),
            min_confidence=0.2,
            cwd=self.root,
        )
        self.assertTrue(reflex["bypass_llm"])
        self.assertEqual(reflex["mode"], "direct_answer")
        self.assertEqual(reflex["node_id"], node_id)
        self.assertIn("[RULE] SQLite WAL concurrency configuration:", reflex["answer"])
        self.assertIn("Do not use on network NFS mounts", reflex["answer"])
        self.assertEqual(reflex["distillation"]["risk_level"], "high")

        # Mutate the file so SHA-256 drifts -> System 1 blocks and falls back to System 2
        target_file.write_text("pragma = 'DELETE'\n", encoding="utf-8")
        drifted = self.brain.reflex(
            "how to configure SQLite WAL concurrency",
            state=_state(),
            min_confidence=0.2,
            cwd=self.root,
        )
        self.assertFalse(drifted["bypass_llm"])
        self.assertEqual(drifted["mode"], "llm_fallback")
        self.assertEqual(len(drifted["precondition_warnings"]), 1)
        self.assertIn("file_hash_mismatch", drifted["precondition_warnings"][0]["reason"])

    def test_reflex_procedure_execution_and_circuit_breaker(self) -> None:
        proc = self.brain.remember_procedure(
            "Run quick smoke check",
            "Run the python smoke check script",
            steps=[f'{sys.executable} -c "import sys; print(sys.argv[1])" {{msg}}'],
            when=["smoke", "check"],
            tools=["python"],
            inputs={"msg": "greeting message"},
            state=_state(),
        )
        proc_id = proc["node_id"]
        # Confirm once so maturity becomes 'verified'
        self.brain.reflect("confirmed", "pytest", node_id=proc_id, state=_state(session_id="s1"))

        # 1. execute=False returns procedure_ready without running subprocess
        ready = self.brain.reflex(
            "run the python smoke check",
            state=_state(),
            execute=False,
            inputs={"msg": "hello_reflex"},
        )
        self.assertTrue(ready["bypass_llm"])
        self.assertEqual(ready["mode"], "procedure_ready")
        self.assertEqual(ready["node_id"], proc_id)

        # 2. execute=True runs the step and confirms outcome
        executed = self.brain.reflex(
            "run the python smoke check",
            state=_state(),
            execute=True,
            inputs={"msg": "hello_reflex"},
        )
        self.assertTrue(executed["bypass_llm"])
        self.assertEqual(executed["mode"], "procedure_executed")
        self.assertIn("hello_reflex", executed["answer"])

        # 3. Failing procedure trips circuit breaker and falls back to System 2
        bad_proc = self.brain.remember_procedure(
            "Run broken deployment script",
            "Execute the broken deployment check",
            steps=[f"{sys.executable} -c \"import sys; sys.stderr.write('deploy failed\\n'); sys.exit(2)\""],
            when=["broken", "deployment"],
            tools=["python"],
            state=_state(),
        )
        bad_id = bad_proc["node_id"]
        self.brain.reflect("confirmed", "initial_setup", node_id=bad_id, state=_state(session_id="s1"))

        tripped = self.brain.reflex(
            "execute the broken deployment check",
            state=_state(),
            execute=True,
        )
        self.assertFalse(tripped["bypass_llm"])
        self.assertEqual(tripped["mode"], "circuit_breaker_tripped")
        self.assertEqual(tripped["node_id"], bad_id)
        self.assertIsNotNone(tripped["fallback_context"])
        # Node should now be marked REFUTED in the store
        inspected = self.service.inspect(bad_id, identity=_state().identity())
        self.assertEqual(inspected["status"], MemoryStatus.REFUTED.value)

    def test_shadow_revision_protects_verified_memory_until_confirmed(self) -> None:
        written = self.brain.remember(
            "decision",
            "Database timeout setting",
            "Use 5000ms busy_timeout on all SQLite connections",
            state=_state(),
        )
        orig_id = written["node_id"]
        self.brain.reflect("confirmed", "pytest", node_id=orig_id, state=_state())
        orig_after_confirm = self.service.inspect(orig_id, identity=_state().identity())
        self.assertGreater(orig_after_confirm["confidence"], 0.0)

        # Revise with protect_verified=True -> creates a PENDING shadow candidate instead of wiping confidence
        shadow = self.service.revise(
            orig_id,
            content="Use 10000ms busy_timeout for heavy batch writes",
            protect_verified=True,
            identity=_state().identity(),
        )
        shadow_id = shadow["node_id"]
        self.assertNotEqual(shadow_id, orig_id)
        self.assertEqual(shadow["status"], MemoryStatus.PENDING.value)
        self.assertEqual(shadow["metadata"].get("supersedes_node_id"), orig_id)

        # Original verified memory is STILL active and retains its confidence!
        orig_still_active = self.service.inspect(orig_id, identity=_state().identity())
        self.assertEqual(orig_still_active["status"], MemoryStatus.ACTIVE.value)
        self.assertGreater(orig_still_active["confidence"], 0.0)

        # Once the shadow revision is confirmed by external verification, it becomes ACTIVE
        # and automatically retires the superseded original to STALE!
        promoted = self.brain.reflect("confirmed", "pytest tests/test_batch.py", node_id=shadow_id, state=_state())
        self.assertEqual(promoted["status"], MemoryStatus.ACTIVE.value)
        self.assertGreater(promoted["confidence"], 0.0)

        orig_retired = self.service.inspect(orig_id, identity=_state().identity())
        self.assertEqual(orig_retired["status"], MemoryStatus.STALE.value)
        self.assertIn(shadow_id, orig_retired["status_reason"])

    def test_model_generation_isolation_in_hot_memory_and_reflex(self) -> None:
        # Memory 1: model-specific heuristic for an older model
        heuristic = self.brain.remember(
            "preference",
            "XML tool call workaround",
            "Wrap tool arguments in manual XML tags to avoid parser truncation",
            metadata={"grounding": "model_heuristic", "model_id": "claude-3.5-sonnet"},
            state=_state(),
        )
        self.brain.reflect("confirmed", "manual_check", node_id=heuristic["node_id"], state=_state())

        # Memory 2: codebase-grounded rule (applies to all models)
        grounded = self.brain.remember(
            "decision",
            "Use RLock in MemoryService",
            "Guard store transactions with threading.RLock in MemoryService",
            metadata={"grounding": "codebase_fact"},
            state=_state(),
        )
        self.brain.reflect("confirmed", "pytest", node_id=grounded["node_id"], state=_state())

        # When a newer model (claude-4.5-sonnet) requests hot memory, the old model's heuristic is excluded
        hot_new_model = self.brain.hot_snapshot(
            state=_state(session_id="sess-new-model", model_id="claude-4.5-sonnet"),
            refresh=True,
        )
        self.assertNotIn(heuristic["node_id"], hot_new_model["node_ids"])
        self.assertIn(grounded["node_id"], hot_new_model["node_ids"])

        # Reflex also blocks the old model's heuristic when called by a newer model
        reflex_new = self.brain.reflex(
            "XML tool call workaround",
            state=_state(model_id="claude-4.5-sonnet"),
            min_confidence=0.2,
        )
        self.assertFalse(reflex_new["bypass_llm"])
        self.assertTrue(any("model_heuristic_mismatch" in w["reason"] for w in reflex_new["precondition_warnings"]))

    def test_near_duplicate_admission_across_hosts_and_polarity_guard(self) -> None:
        first = self.brain.remember(
            "decision",
            "SQLite WAL concurrency policy",
            "Enable WAL journal mode and 5000ms busy timeout for local SQLite database connections",
            metadata={"host": "codex"},
            state=_state(),
        )
        self.assertEqual(first["verdict"], AdmitVerdict.WRITE.value)

        # Near-duplicate from another host with minor punctuation/wording difference -> skipped
        near_dup = self.brain.remember(
            "decision",
            "SQLite WAL concurrency policy",
            "Enable WAL journal mode and 5000ms busy timeout for local SQLite database connection.",
            metadata={"host": "claude-code"},
            state=_state(),
        )
        self.assertEqual(near_dup["verdict"], AdmitVerdict.SKIP.value)
        self.assertEqual(near_dup["duplicate_node_id"], first["node_id"])

        # Opposite negation polarity ("Do not enable...") must NOT be skipped as duplicate
        negated = self.brain.remember(
            "decision",
            "SQLite WAL concurrency policy",
            "Do not enable WAL journal mode and 5000ms busy timeout for local SQLite database connections",
            metadata={"host": "agy"},
            state=_state(),
        )
        self.assertEqual(negated["verdict"], AdmitVerdict.WRITE.value)

    def test_cross_level_duplicate_admission(self) -> None:
        user_pref = self.brain.remember(
            "preference",
            "Concise Chinese responses",
            "Always reply in concise Chinese with clickable file links",
            context_level="user",
            state=_state(),
        )
        self.assertEqual(user_pref["verdict"], AdmitVerdict.WRITE.value)

        # Saving the exact same preference at project level should detect the user-level duplicate
        proj_dup = self.brain.remember(
            "preference",
            "Concise Chinese responses",
            "Always reply in concise Chinese with clickable file links",
            context_level="project",
            state=_state(),
        )
        self.assertEqual(proj_dup["verdict"], AdmitVerdict.SKIP.value)
        self.assertEqual(proj_dup["duplicate_node_id"], user_pref["node_id"])

    def test_reflex_procedure_verify_clause_and_inconclusive_execution(self) -> None:
        # 1. Procedure whose step passes (exit 0) but verify command fails (exit 1)
        proc_bad_verify = self.brain.remember_procedure(
            "Run migration with failing verify",
            "Execute schema migration and verify table count",
            steps=[f"{sys.executable} -c \"print('migration step ok')\""],
            verify=f"{sys.executable} -c \"import sys; sys.stderr.write('AssertionError: table missing\\n'); sys.exit(1)\"",
            when=["schema", "migration", "verify"],
            tools=["alembic"],
            state=_state(),
        )
        self.brain.reflect(
            "confirmed", "initial_seed", node_id=proc_bad_verify["node_id"], state=_state(session_id="s1")
        )

        tripped = self.brain.reflex(
            "execute schema migration and verify table count",
            state=_state(),
            execute=True,
        )
        self.assertFalse(tripped["bypass_llm"])
        self.assertEqual(tripped["mode"], "circuit_breaker_tripped")
        self.assertEqual(tripped["node_id"], proc_bad_verify["node_id"])

        # 2. Procedure whose step produces an inconclusive result ("collected 0 items")
        proc_inconc = self.brain.remember_procedure(
            "Run empty test suite",
            "Execute the empty unit test suite",
            steps=[f"{sys.executable} -c \"print('collected 0 items')\""],
            when=["empty", "unit", "suite"],
            tools=["pytest"],
            state=_state(),
        )
        self.brain.reflect("confirmed", "initial_seed", node_id=proc_inconc["node_id"], state=_state(session_id="s1"))
        before_conf = self.service.inspect(proc_inconc["node_id"], identity=_state().identity())["confidence"]

        inconc_res = self.brain.reflex(
            "execute the empty unit test suite",
            state=_state(),
            execute=True,
        )
        self.assertFalse(inconc_res["bypass_llm"])
        self.assertEqual(inconc_res["mode"], "inconclusive_execution")
        after_conf = self.service.inspect(proc_inconc["node_id"], identity=_state().identity())["confidence"]
        self.assertEqual(before_conf, after_conf)

    def test_shadow_revision_conflict_when_base_modified(self) -> None:
        written = self.brain.remember(
            "decision",
            "Connection pool limit",
            "Set max_connections=20 on the pool",
            state=_state(),
        )
        orig_id = written["node_id"]
        self.brain.reflect("confirmed", "pytest", node_id=orig_id, state=_state())

        shadow = self.service.revise(
            orig_id,
            content="Set max_connections=50 on the pool",
            protect_verified=True,
            identity=_state().identity(),
        )
        shadow_id = shadow["node_id"]

        # Another writer directly revises the original node (bumping its revision)
        self.service.revise(
            orig_id,
            content="Set max_connections=35 on the pool",
            protect_verified=False,
            identity=_state().identity(),
        )

        # Confirming the stale shadow revision must now raise ConflictError
        with self.assertRaises(ConflictError):
            self.brain.reflect("confirmed", "pytest", node_id=shadow_id, state=_state())

    def test_reflex_fast_path_a_endorsed_with_metacognition(self) -> None:
        written = self.brain.remember(
            "decision",
            "Redis cache TTL policy",
            "Always set TTL to 3600 seconds on user session tokens in Redis.",
            state=_state(),
        )
        node_id = written["node_id"]
        # Confirm twice so confidence reaches 0.5 >= trust_floor (0.5)
        self.brain.reflect("confirmed", "pytest", node_id=node_id, state=_state())
        self.brain.reflect("confirmed", "integration_test", node_id=node_id, state=_state())

        reflex = self.brain.reflex(
            "Redis cache TTL policy",
            state=_state(),
            min_confidence=0.2,
        )
        self.assertTrue(reflex["bypass_llm"])
        self.assertEqual(reflex["mode"], "direct_answer")
        self.assertIn("assessment_verdict", reflex["distillation"])
        self.assertEqual(reflex["distillation"]["assessment_verdict"], "known")
        self.assertIn("known_coverage", reflex["distillation"])
        self.assertGreater(reflex["distillation"]["known_coverage"], 0.0)

    def test_reflex_fast_path_a_vetoed_by_contradicted_memory(self) -> None:
        # Active verified memory
        active = self.brain.remember(
            "decision",
            "Server timeout limit",
            "Configure server timeout to 60 seconds for gateway.",
            state=_state(),
        )
        self.brain.reflect("confirmed", "manual_check", node_id=active["node_id"], state=_state())

        # Contradicted / refuted memory matching the same goal
        bad = self.brain.remember(
            "decision",
            "Server timeout deprecated setting",
            "Configure server timeout to 5 seconds for gateway.",
            state=_state(),
        )
        self.brain.reflect("refuted", "load_test_failure", node_id=bad["node_id"], state=_state())

        # Calling reflex on server timeout should be vetoed by contradicted memory
        res = self.brain.reflex(
            "Configure server timeout for gateway",
            state=_state(),
            min_confidence=0.2,
        )
        self.assertFalse(res["bypass_llm"])
        self.assertEqual(res["mode"], "llm_fallback")
        veto_warnings = [
            w for w in res["precondition_warnings"] if "metacognition veto: contradicted memory" in w.get("reason", "")
        ]
        self.assertTrue(len(veto_warnings) >= 1)

    def test_reflex_fast_path_a_vetoed_by_conflicted_uncertain_candidate(self) -> None:
        # Create an active memory marked as conflicted in metadata
        conflicted_node = self.brain.remember(
            "decision",
            "Database connection pooling size",
            "Set connection pool size to 50 connections.",
            metadata={"conflicted": True, "conflict_reason": "opposing spec"},
            state=_state(),
        )
        self.brain.reflect("confirmed", "doc_check", node_id=conflicted_node["node_id"], state=_state())

        res = self.brain.reflex(
            "Database connection pooling size",
            state=_state(),
            min_confidence=0.2,
        )
        self.assertFalse(res["bypass_llm"])
        self.assertEqual(res["mode"], "llm_fallback")
        warnings = [w for w in res["precondition_warnings"] if "conflicted" in w.get("reason", "")]
        self.assertTrue(len(warnings) >= 1)


if __name__ == "__main__":
    unittest.main()
