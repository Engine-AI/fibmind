"""Unit tests for fibmind.verify automated verification loop."""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

from fibmind import FibMind, MemoryNode, NodeType
from fibmind.mcp_server import build_server
from fibmind.service import MemoryService
from fibmind.verify import main, run_command_and_reflect


def _result_json(result: object) -> dict:
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, dict):
        raise AssertionError("tool call did not return structured content")
    return structured


class VerifyLoopTests(unittest.TestCase):
    def test_verify_command_success_confirms_memory(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_fix_1",
                title="Fix Concurrency Deadlock",
                content="Applied WAL mode and 5s timeout to resolve contention",
                category="decision",
                node_type=NodeType.CONCEPT,
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            # Run a successful python command
            res = run_command_and_reflect(
                [sys.executable, "-c", "import sys; print('All tests passed'); sys.exit(0)"],
                node_id="mem_fix_1",
                source="pytest",
                store_path=store_path,
            )

            self.assertEqual(res["exit_code"], 0)
            self.assertEqual(res["verdict"], "confirmed")
            self.assertEqual(res["source"], "pytest")
            self.assertIn("All tests passed", res["stdout"])
            self.assertGreater(res["reflection"]["confidence"], 0.0)

    def test_verify_command_failure_refutes_memory(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_fix_2",
                title="Cache Invalidation Strategy",
                content="Invalidate on every read",
                category="decision",
                node_type=NodeType.CONCEPT,
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            # Run a failing command with stderr
            res = run_command_and_reflect(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('AssertionError: benchmark failed\\n'); sys.exit(1)",
                ],
                node_id="mem_fix_2",
                store_path=store_path,
            )

            self.assertEqual(res["exit_code"], 1)
            self.assertEqual(res["verdict"], "refuted")
            self.assertIn("python", res["source"].lower())
            self.assertIn("AssertionError: benchmark failed", res["stderr"])
            self.assertIn("AssertionError: benchmark failed", res["note"])

    def test_verify_command_resolves_goal(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_search_target",
                title="Token Expiration Handling",
                content="Refresh expired auth tokens automatically",
                category="solution",
                node_type=NodeType.CONCEPT,
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            res = run_command_and_reflect(
                [sys.executable, "-c", "import sys; sys.exit(0)"],
                goal="Token Expiration Handling",
                store_path=store_path,
            )

            self.assertEqual(res["exit_code"], 0)
            self.assertEqual(res["reflection"]["node_id"], "mem_search_target")

    def test_verify_environment_failure_is_inconclusive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_env_test",
                title="Redis connection pool",
                content="Use connection pool to avoid port exhaustion",
                category="decision",
                node_type=NodeType.CONCEPT,
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            # ModuleNotFoundError in python
            res = run_command_and_reflect(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('ModuleNotFoundError: No module named foobar\\n'); sys.exit(1)",
                ],
                node_id="mem_env_test",
                store_path=store_path,
            )

            self.assertEqual(res["exit_code"], 1)
            self.assertEqual(res["verdict"], "inconclusive")
            # Inconclusive should not degrade confidence
            self.assertEqual(res["reflection"]["confidence"], 0.0)

    def test_verify_unrelated_file_failure_is_inconclusive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_unrelated_test",
                title="Auth token refresh",
                content="Refresh expired auth tokens automatically",
                category="decision",
                node_type=NodeType.CONCEPT,
                metadata={"files": ["src/fibmind/auth.py"]},
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            # Failure occurs in test_unrelated.py, not mentioning auth.py
            res = run_command_and_reflect(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('AssertionError: tests/test_payment.py failed\\n'); sys.exit(1)",
                ],
                node_id="mem_unrelated_test",
                store_path=store_path,
            )

            self.assertEqual(res["exit_code"], 1)
            self.assertEqual(res["verdict"], "inconclusive")
            self.assertIn("unrelated", res["note"])

    def test_verify_zero_tests_collected_is_inconclusive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_zero_test",
                title="Zero Test Target",
                content="Feature with no tests",
                category="decision",
                node_type=NodeType.CONCEPT,
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            res = run_command_and_reflect(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('collected 0 items'); sys.exit(0)",
                ],
                node_id="mem_zero_test",
                store_path=store_path,
            )

            self.assertEqual(res["verdict"], "inconclusive")

    def test_verify_cli_main(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_cli_test",
                title="CLI Test Execution",
                content="Verify CLI wrapper propagates exit code and records outcome",
                category="test",
                node_type=NodeType.CONCEPT,
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            # Test exit code 0
            rc = main(
                [
                    "--node-id",
                    "mem_cli_test",
                    "--store",
                    str(store_path),
                    "--",
                    sys.executable,
                    "-c",
                    "import sys; sys.exit(0)",
                ]
            )
            self.assertEqual(rc, 0)

            # Test exit code 2
            rc_err = main(
                [
                    "--node-id",
                    "mem_cli_test",
                    "--store",
                    str(store_path),
                    "--",
                    sys.executable,
                    "-c",
                    "import sys; sys.exit(2)",
                ]
            )
            self.assertEqual(rc_err, 2)

    def test_mcp_fibbrain_verify_tool(self) -> None:
        pytest.importorskip("mcp")

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_mcp_test",
                title="MCP Verification Integration",
                content="MCP tool fibbrain_verify executes command and updates confidence",
                category="decision",
                node_type=NodeType.CONCEPT,
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)
            server = build_server(service)

            async def scenario():
                tool_res = await server.call_tool(
                    "fibbrain_verify",
                    {
                        "command": f'{sys.executable} -c "import sys; sys.exit(0)"',
                        "node_id": "mem_mcp_test",
                        "source": "ci_check",
                    },
                )
                payload = _result_json(tool_res)
                self.assertEqual(payload["exit_code"], 0)
                self.assertEqual(payload["verdict"], "confirmed")
                self.assertEqual(payload["source"], "ci_check")

            asyncio.run(scenario())

    def test_verify_success_unrelated_target_is_inconclusive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_target_scope_test",
                title="Auth Token Expiry",
                content="Refresh expired tokens automatically",
                category="decision",
                node_type=NodeType.CONCEPT,
                metadata={"files": ["src/fibmind/auth.py"]},
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            # Succeeded command, but output only mentions payment module, NOT auth.py
            res = run_command_and_reflect(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('tests/test_payment.py passed successfully'); sys.exit(0)",
                ],
                node_id="mem_target_scope_test",
                store_path=store_path,
            )

            self.assertEqual(res["exit_code"], 0)
            self.assertEqual(res["verdict"], "inconclusive")
            self.assertIn("did not touch target files", res["note"])
            # Confidence must not be elevated
            self.assertEqual(res["reflection"]["confidence"], 0.0)
            self.assertEqual(res["attribution"]["touched_targets"], [])

    def test_verify_success_matching_target_is_confirmed(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_target_touch_test",
                title="Auth Token Expiry",
                content="Refresh expired tokens automatically",
                category="decision",
                node_type=NodeType.CONCEPT,
                metadata={"files": ["src/fibmind/auth.py"]},
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            # Succeeded command that explicitly verifies auth.py
            res = run_command_and_reflect(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('tests/test_auth.py: PASSED (verified src/fibmind/auth.py)'); sys.exit(0)",
                ],
                node_id="mem_target_touch_test",
                store_path=store_path,
            )

            self.assertEqual(res["exit_code"], 0)
            self.assertEqual(res["verdict"], "confirmed")
            self.assertGreater(res["reflection"]["confidence"], 0.0)
            self.assertIn("src/fibmind/auth.py", res["attribution"]["touched_targets"])

    def test_verify_assertion_patterns(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store_path = Path(f.name)
            service = MemoryService(store_path)
            node = MemoryNode(
                id="mem_assert_test",
                title="Bulk Query Benchmark",
                content="Query latency must be under 50ms",
                category="decision",
                node_type=NodeType.CONCEPT,
            )

            def _init(m: FibMind) -> None:
                m.nodes[node.id] = node

            service.run(_init)

            # Command succeeds but lacks required assertion pattern
            res_fail = run_command_and_reflect(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('Done without benchmark numbers'); sys.exit(0)",
                ],
                node_id="mem_assert_test",
                assertion_patterns=[r"latency:\s*\d+ms"],
                store_path=store_path,
            )
            self.assertEqual(res_fail["verdict"], "inconclusive")
            self.assertIn("missed required assertion pattern", res_fail["note"])

            # Command succeeds and matches assertion pattern
            res_ok = run_command_and_reflect(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('Benchmark completed: latency: 25ms'); sys.exit(0)",
                ],
                node_id="mem_assert_test",
                assertion_patterns=[r"latency:\s*\d+ms"],
                store_path=store_path,
            )
            self.assertEqual(res_ok["verdict"], "confirmed")

    def test_short_or_generic_stems_do_not_false_positive_match(self) -> None:
        from fibmind.verify import diagnose_verification_outcome

        # 'io.py' stem 'io' appears inside 'execution', and 'test.py' stem 'test' appears inside 'test session'
        verdict, _ = diagnose_verification_outcome(
            0,
            "=== test session starts ===\nexecution finished in 0.12s",
            "",
            target_files=["src/fibmind/io.py", "src/fibmind/test.py"],
        )
        self.assertEqual(verdict, "inconclusive")

        # Explicit filename 'io.py' does match
        verdict_ok, _ = diagnose_verification_outcome(
            0,
            "tests/test_io.py verified src/fibmind/io.py",
            "",
            target_files=["src/fibmind/io.py"],
        )
        self.assertEqual(verdict_ok, "confirmed")


if __name__ == "__main__":
    unittest.main()
