"""Unit tests for AI-OS Cognitive MMU, Unified Kernel Dispatcher, and MCP server extensions."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from fibmind import BrainState, FibBrain, MemoryService, compute_file_hashes
from fibmind.mcp_server import build_server


def _state(**overrides: str) -> BrainState:
    base = {
        "owner": "abbila",
        "workspace_id": "fibmind",
        "project_id": "fibmind",
        "session_id": "sess-kernel-1",
    }
    base.update(overrides)
    return BrainState(**base)


def _result_json(result: object) -> dict:
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, dict):
        raise AssertionError("tool call did not return structured content")
    return structured


class CognitiveMMUTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = MemoryService(self.root / "memory.db")
        self.brain = FibBrain(self.service)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_session_page_table_hits_and_delta_paging(self) -> None:
        m1 = self.brain.remember(
            "decision",
            "Python packaging setup",
            "Always use pyproject.toml and hatchling as the build backend for packaging.",
            state=_state(),
        )["node_id"]
        m2 = self.brain.remember(
            "decision",
            "Docker compose orchestration",
            "Use compose v2 specification with healthchecks for local service orchestration.",
            state=_state(),
        )["node_id"]

        st = _state(session_id="session-mmu-test")

        # Call 1: First time in session -> Page Faults
        res1 = self.brain.context("Python packaging and Docker compose", state=st)
        mmu1 = res1.get("mmu")
        self.assertIsNotNone(mmu1)
        self.assertEqual(mmu1["page_hits"], [])
        self.assertIn(m1, mmu1["page_faults"])
        self.assertIn(m2, mmu1["page_faults"])
        self.assertEqual(mmu1["dirty_pages"], [])
        self.assertFalse(mmu1["delta_only"])
        self.assertIn("Python packaging setup", res1["text"])
        self.assertIn("Docker compose orchestration", res1["text"])

        # Call 2: Second call in same session without delta_only -> Page Hits
        res2 = self.brain.context("Python packaging and Docker compose", state=st, delta_only=False)
        mmu2 = res2["mmu"]
        self.assertIn(m1, mmu2["page_hits"])
        self.assertIn(m2, mmu2["page_hits"])
        self.assertEqual(mmu2["page_faults"], [])
        self.assertEqual(mmu2["dirty_pages"], [])

        # Call 3: Third call with delta_only=True -> Already paged nodes omitted from text
        res3 = self.brain.context("Python packaging and Docker compose", state=st, delta_only=True)
        mmu3 = res3["mmu"]
        self.assertTrue(mmu3["delta_only"])
        self.assertIn(m1, mmu3["page_hits"])
        self.assertIn(m2, mmu3["page_hits"])
        self.assertEqual(mmu3["page_faults"], [])
        # Both nodes already paged, so delta text and memories should omit them
        self.assertNotIn("Python packaging setup", res3["text"])
        self.assertNotIn("Docker compose orchestration", res3["text"])
        self.assertEqual(len(res3["memories"]), 0)

        # Call 4: Add new memory, delta_only should yield only the new node
        m3 = self.brain.remember(
            "decision",
            "Kubernetes ingress controller",
            "Deploy ingress-nginx for routing cluster ingress traffic.",
            state=_state(),
        )["node_id"]

        res4 = self.brain.context("Python packaging and Kubernetes ingress", state=st, delta_only=True)
        mmu4 = res4["mmu"]
        self.assertIn(m1, mmu4["page_hits"])
        self.assertIn(m3, mmu4["page_faults"])
        self.assertIn("Kubernetes ingress controller", res4["text"])
        self.assertNotIn("Python packaging setup", res4["text"])
        rendered_ids = [m["node_id"] for m in res4["memories"]]
        self.assertIn(m3, rendered_ids)
        self.assertNotIn(m1, rendered_ids)

    def test_dirty_page_invalidation_on_revision(self) -> None:
        m1 = self.brain.remember(
            "decision",
            "FastAPI background tasks",
            "Use BackgroundTasks for fire-and-forget lightweight email tasks.",
            state=_state(),
        )["node_id"]

        st = _state(session_id="session-dirty-rev")

        # Page m1
        res1 = self.brain.context("FastAPI background tasks", state=st)
        self.assertIn(m1, res1["mmu"]["page_faults"])

        # Revise m1 -> revision changes
        self.service.revise(
            m1,
            content="Use Celery or ARQ instead of FastAPI BackgroundTasks for critical tasks.",
            identity=st.identity(),
        )

        # Next call in same session -> Dirty page detected and surfaced in warnings
        res2 = self.brain.context("FastAPI background tasks", state=st)
        mmu2 = res2["mmu"]
        dirty_ids = [d["node_id"] for d in mmu2["dirty_pages"]]
        self.assertIn(m1, dirty_ids)
        self.assertTrue(any("revision_changed" in d["reason"] for d in mmu2["dirty_pages"]))

        # Also check warnings
        dirty_warnings = [w for w in res2.get("warnings", []) if w.get("kind") == "dirty_page"]
        self.assertTrue(any(w["node_id"] == m1 for w in dirty_warnings))

    def test_dirty_page_invalidation_on_refutation(self) -> None:
        m1 = self.brain.remember(
            "decision",
            "Postgres connection pool max size",
            "Set connection pool size to 500 for high concurrency web requests.",
            state=_state(),
        )["node_id"]

        st = _state(session_id="session-dirty-refute")

        # Page m1
        res1 = self.brain.context("Postgres connection pool size", state=st)
        self.assertIn(m1, res1["mmu"]["page_faults"])

        # Refute m1
        self.brain.reflect("refuted", "load test exhausted database connection limits", node_id=m1, state=st)

        # Next call in session-dirty-refute
        res2 = self.brain.context("Postgres connection pool", state=st)
        mmu2 = res2["mmu"]
        dirty_ids = [d["node_id"] for d in mmu2["dirty_pages"]]
        self.assertIn(m1, dirty_ids)
        self.assertTrue(any(d["reason"] == "status_refuted" for d in mmu2["dirty_pages"]))

        dirty_warnings = [w for w in res2.get("warnings", []) if w.get("kind") == "dirty_page"]
        self.assertTrue(any(w["node_id"] == m1 for w in dirty_warnings))

    def test_recall_tracks_mmu_session(self) -> None:
        m1 = self.brain.remember(
            "decision",
            "Redis cache TTL policy",
            "Set standard TTL of 3600 seconds on all user session tokens.",
            state=_state(),
        )["node_id"]

        st = _state(session_id="session-recall-mmu")

        pack1 = self.brain.recall("Redis cache TTL policy", state=st)
        self.assertIn("mmu", pack1)
        self.assertIn(m1, pack1["mmu"]["page_faults"])
        self.assertEqual(pack1["mmu"]["page_hits"], [])

        pack2 = self.brain.recall("Redis cache TTL policy", state=st, delta_only=True)
        self.assertIn(m1, pack2["mmu"]["page_hits"])
        self.assertTrue(pack2["mmu"]["delta_only"])
        # Hits filtered out of pack["hits"] when delta_only=True
        self.assertNotIn(m1, [h["node_id"] for h in pack2.get("hits", [])])

    def test_cross_process_persistent_mmu_page_table(self) -> None:
        m1 = self.brain.remember(
            "decision",
            "Cross process persistent MMU",
            "Store MMU page tables in sqlite metadata setting for zero-loss daemon restarts.",
            state=_state(),
        )["node_id"]

        st = _state(session_id="session-cross-proc")

        # Process 1 / Instance 1 pages m1
        res1 = self.brain.context("Cross process persistent MMU", state=st)
        self.assertIn(m1, res1["mmu"]["page_faults"])

        # Process 2 / Instance 2 on same database file restores page table
        brain2 = FibBrain(MemoryService(self.root / "memory.db"))
        res2 = brain2.context("Cross process persistent MMU", state=st, delta_only=True)
        self.assertIn(m1, res2["mmu"]["page_hits"])
        self.assertEqual(len(res2["mmu"]["page_faults"]), 0)
        self.assertNotIn("Cross process persistent MMU", res2["text"])

        # Instance 2 revises memory
        self.service.revise(m1, content="Updated persistent MMU implementation details.", identity=st.identity())

        # Process 3 / Instance 3 recognizes dirty page across processes
        brain3 = FibBrain(MemoryService(self.root / "memory.db"))
        res3 = brain3.context("Cross process persistent MMU", state=st)
        self.assertIn(m1, [d["node_id"] for d in res3["mmu"]["dirty_pages"]])
        dirty_warn = [w for w in res3.get("warnings", []) if w.get("kind") == "dirty_page"]
        self.assertTrue(any(w["node_id"] == m1 for w in dirty_warn))

    def test_tlb_ttl_and_capacity_eviction(self) -> None:
        # Create brain with max_tlb_pages=3 and tlb_ttl_steps=2
        tlb_brain = FibBrain(self.service, max_tlb_pages=3, tlb_ttl_steps=2)
        m1 = tlb_brain.remember("decision", "Alpha architecture", "Alpha details", state=_state())["node_id"]
        m2 = tlb_brain.remember("decision", "Beta framework", "Beta details", state=_state())["node_id"]
        m3 = tlb_brain.remember("decision", "Gamma engine", "Gamma details", state=_state())["node_id"]
        m4 = tlb_brain.remember("decision", "Delta protocol", "Delta details", state=_state())["node_id"]

        st = _state(session_id="session-tlb-test")

        # Step 1: Page m1 -> step 1, m1 last_step = 1
        res1 = tlb_brain.context("Alpha architecture", state=st)
        self.assertEqual(res1["mmu"]["step"], 1)
        self.assertIn(m1, res1["mmu"]["page_faults"])

        # Step 2: Page m2 -> step 2
        res2 = tlb_brain.context("Beta framework", state=st)
        self.assertEqual(res2["mmu"]["step"], 2)
        self.assertIn(m2, res2["mmu"]["page_faults"])

        # Step 3: Page m3 -> step 3 (m1 last_step=1, step=3; diff=2 <= ttl=2, not yet expired)
        res3 = tlb_brain.context("Gamma engine", state=st)
        self.assertEqual(res3["mmu"]["step"], 3)
        self.assertIn(m3, res3["mmu"]["page_faults"])

        # Step 4: Advance step to 4 by requesting m1. (4 - 1 = 3 > ttl=2). Attention decay!
        res4 = tlb_brain.context("Alpha architecture", state=st, delta_only=True)
        self.assertEqual(res4["mmu"]["step"], 4)
        self.assertIn(m1, res4["mmu"]["evicted_pages"])
        self.assertIn(m1, res4["mmu"]["page_faults"])
        self.assertIn("Alpha architecture", res4["text"])

        # Capacity Eviction:
        # Currently page_table has {m2, m3, m1} (size 3 == max_tlb_pages).
        # Paging m4 will cause size 4 > 3, oldest accessed should be evicted!
        res5 = tlb_brain.context("Delta protocol", state=st)
        self.assertIn(m4, res5["mmu"]["page_faults"])
        pages = tlb_brain._load_session_pages("session-tlb-test")
        self.assertLessEqual(len(pages), 3)
        self.assertTrue(len(res5["mmu"]["evicted_pages"]) > 0)

    def test_mmu_page_pinning_and_swap_index(self) -> None:
        tlb_brain = FibBrain(self.service, max_tlb_pages=2, tlb_ttl_steps=2)
        st = _state(session_id="session-pinning-test")

        m1 = tlb_brain.remember(
            "decision",
            "Postgres max connections pool",
            "Set connection pool size to 100.",
            state=st,
        )["node_id"]

        m2 = tlb_brain.remember(
            "decision",
            "Pinned auth invariant",
            "Always verify JWT signature with HMAC-SHA256.",
            state=st,
        )["node_id"]

        m3 = tlb_brain.remember(
            "requirement",
            "PCI DSS requirement",
            "Never store card verification value in persistence.",
            state=st,
        )["node_id"]
        # Reflect confirmed twice to achieve confidence >= 0.5 (auto-pin threshold)
        tlb_brain.reflect("confirmed", "compliance check 1", node_id=m3, state=st, trust_ring="ring0")
        tlb_brain.reflect("confirmed", "compliance check 2", node_id=m3, state=st, trust_ring="ring0")
        self.assertGreaterEqual(tlb_brain.memory.inspect(m3)["confidence"], 0.5)

        # Explicitly pin m2
        pin_res = tlb_brain.pin_page("session-pinning-test", m2)
        self.assertIn(m2, pin_res["pinned_pages"])

        # Page m1, m2, m3 in session
        tlb_brain.context("Postgres max connections pool", state=st)
        tlb_brain.context("Pinned auth invariant", state=st)
        res_m3 = tlb_brain.context("PCI DSS requirement", state=st)
        mmu_m3 = res_m3["mmu"]
        # m2 was manually pinned, m3 auto-pinned because requirement with confidence >= 0.5
        self.assertIn(m2, mmu_m3["pinned_pages"])
        self.assertIn(m3, mmu_m3["pinned_pages"])
        self.assertNotIn(m1, mmu_m3["pinned_pages"])

        # Advance session clock past TTL steps (tlb_ttl_steps=2) and capacity (max_tlb_pages=2)
        tlb_brain.context("Unrelated query step 2", state=st)
        tlb_brain.context("Unrelated query step 3", state=st)
        tlb_brain.context("Unrelated query step 4", state=st)

        # Call context with delta_only=True
        # m1 is unpinned and expired -> evicted and page fault
        r1 = tlb_brain.context("Postgres max connections pool", state=st, delta_only=True)
        self.assertIn(m1, r1["mmu"].get("evicted_pages", []))
        self.assertIn(m1, r1["mmu"].get("page_faults", []))

        # m2 and m3 are pinned -> survived TTL & capacity eviction, hit as valid pages
        r2 = tlb_brain.context("Pinned auth invariant", state=st, delta_only=True)
        self.assertNotIn(m2, r2["mmu"].get("evicted_pages", []))
        self.assertIn(m2, r2["mmu"].get("page_hits", []))

        r3 = tlb_brain.context("PCI DSS requirement", state=st, delta_only=True)
        self.assertNotIn(m3, r3["mmu"].get("evicted_pages", []))
        self.assertIn(m3, r3["mmu"].get("page_hits", []))

        # Test unpin_page
        unpin_res = tlb_brain.unpin_page("session-pinning-test", m2)
        self.assertNotIn(m2, unpin_res["pinned_pages"])

        # Test swap_index with small budget_tokens
        st_swap = _state(session_id="session-swap-test")
        res_swap = tlb_brain.context(
            "Postgres max connections and Pinned auth invariant and PCI DSS requirement",
            state=st_swap,
            budget_tokens=10,
        )
        swap_index = res_swap["mmu"].get("swap_index", [])
        self.assertTrue(len(swap_index) > 0)
        for item in swap_index:
            self.assertIn("node_id", item)
            self.assertIn("title", item)
            self.assertIn("category", item)
            self.assertIn("confidence", item)
            self.assertEqual(item["uri"], f"fibmind://node/{item['node_id']}")

    def test_mmu_demand_page_in_and_pin(self) -> None:
        session_id = "demand-page-sess"
        st = _state(session_id=session_id)

        m1 = self.brain.remember(
            "decision",
            "Redis cluster sharding topology",
            "Use 3 masters and 3 replicas for high availability Redis cluster deployment.",
            state=st,
        )["node_id"]

        # 1. Truncate memory into swap_index via small budget_tokens
        res_small = self.brain.context("Redis cluster sharding topology", state=st, budget_tokens=10)
        swap_index = res_small["mmu"].get("swap_index", [])
        self.assertTrue(len(swap_index) > 0)
        swapped_id = swap_index[0]["node_id"]
        self.assertEqual(swapped_id, m1)

        # 2. Demand page in with pin=True
        res_page_in = self.brain.page_in(session_id, [swapped_id], pin=True, state=st)
        self.assertEqual(res_page_in["session_id"], session_id)
        paged_node_ids = [n["node_id"] for n in res_page_in["paged_in"]]
        self.assertIn(swapped_id, paged_node_ids)

        # 3. Verify in session page table and pinned_pages
        page_table = self.brain._load_session_pages(session_id)
        self.assertIn(swapped_id, page_table)
        pinned = self.brain._load_session_pinned(session_id)
        self.assertIn(swapped_id, pinned)
        self.assertIn(swapped_id, res_page_in["mmu"]["pinned_pages"])


class KernelStepRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = MemoryService(self.root / "memory.db")
        self.brain = FibBrain(self.service)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mode_system1_reflex_bypasses_llm(self) -> None:
        target_file = self.root / "src" / "storage.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text("pragma = 'WAL'\n", encoding="utf-8")

        hashes = compute_file_hashes(["src/storage.py"], cwd=self.root)
        written = self.brain.remember(
            "decision",
            "Configure SQLite WAL concurrency",
            (
                "Configure SQLite WAL concurrency: Always enable PRAGMA journal_mode=WAL and busy_timeout=5000 on SQLite connections.\n"
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

        # Confirm memory 3 times so confidence rises above TRUST_FLOOR (0.50) and assessment verdict is "known"
        self.brain.reflect("confirmed", "pytest tests/test_storage.py 1", node_id=node_id, state=_state())
        self.brain.reflect("confirmed", "pytest tests/test_storage.py 2", node_id=node_id, state=_state())
        self.brain.reflect("confirmed", "pytest tests/test_storage.py 3", node_id=node_id, state=_state())

        step = self.brain.kernel_step(
            "Configure SQLite WAL concurrency",
            state=_state(),
            workspace_root=self.root,
        )

        self.assertEqual(step["dispatch_mode"], "system1_reflex")
        self.assertTrue(step["bypass_llm"])
        self.assertIsNone(step["context"])
        self.assertIsNotNone(step["reflex"])
        self.assertEqual(step["next_actions"][0]["route"], "reflex")

    def test_mode_interrupt_verify(self) -> None:
        # Unconfirmed memory with verification command
        self.brain.remember(
            "decision",
            "Payment stripe webhooks validation",
            "Verify stripe-signature header using stripe webhook construct_event helper.",
            metadata={
                "command": "pytest tests/test_webhook.py",
            },
            state=_state(),
        )

        step = self.brain.kernel_step(
            "Payment stripe webhooks validation",
            state=_state(),
            workspace_root=self.root,
        )

        self.assertEqual(step["dispatch_mode"], "interrupt_verify")
        self.assertFalse(step["bypass_llm"])
        self.assertIsNotNone(step["context"])
        self.assertIn("mmu", step)

    def test_mode_interrupt_conflict(self) -> None:
        m1 = self.brain.remember(
            "decision",
            "GraphQL subscription transport",
            "Use graphql-ws library for websockets instead of legacy subscriptions-transport-ws.",
            state=_state(),
        )["node_id"]

        # Refute it to create a contradiction
        self.brain.reflect(
            "refuted", "client library incompatible with graphql-ws protocol", node_id=m1, state=_state()
        )

        step = self.brain.kernel_step(
            "GraphQL subscription transport protocol",
            state=_state(),
            workspace_root=self.root,
        )

        self.assertEqual(step["dispatch_mode"], "interrupt_conflict")
        self.assertFalse(step["bypass_llm"])
        self.assertIsNotNone(step["context"])

    def test_mode_system2_reason(self) -> None:
        # Normal query without reflex, without verify command, without conflict
        step = self.brain.kernel_step(
            "Design new user authentication microservice architecture",
            state=_state(),
            workspace_root=self.root,
        )

        self.assertEqual(step["dispatch_mode"], "system2_reason")
        self.assertFalse(step["bypass_llm"])
        self.assertIsNotNone(step["context"])
        self.assertIn("mmu", step)
        self.assertIn("state", step)

    def test_auto_grounding_on_remember(self) -> None:
        target_file = self.root / "pkg" / "engine.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(
            "class Engine:\n    def start(self):\n        return True\n",
            encoding="utf-8",
        )

        written = self.brain.remember(
            "decision",
            "Engine startup architecture",
            "Always call Engine.start() before serving traffic.",
            metadata={
                "files": ["pkg/engine.py"],
                "symbols": ["pkg/engine.py::Engine.start"],
            },
            cwd=self.root,
            state=_state(),
        )
        node_id = written["node_id"]
        node = self.service.inspect(node_id)
        meta = node["metadata"]
        self.assertIn("file_hashes", meta)
        self.assertIn("pkg/engine.py", meta["file_hashes"])
        self.assertIn("symbol_hashes", meta)
        self.assertIn("pkg/engine.py::Engine.start", meta["symbol_hashes"])

    def test_advise_lease_enforcement(self) -> None:
        # Create an active lease from another session targeting "src/auth.py"
        self.brain.observe(
            "lease",
            "Holding exclusive write lock on src/auth.py",
            {"files": ["src/auth.py"]},
            state=_state(session_id="session-holding-lease"),
        )

        # Session B advises editing "src/auth.py" with enforce_leases=False -> allowed
        adv1 = self.brain.advise(
            "edit_file",
            kind="tool",
            planned_files=["src/auth.py"],
            enforce_leases=False,
            state=_state(session_id="session-requesting"),
        )
        self.assertTrue(len(adv1["active_leases"]) > 0)
        self.assertEqual(adv1["verdict"], "allow")

        # Session B advises editing "src/auth.py" with enforce_leases=True -> REJECT!
        adv2 = self.brain.advise(
            "edit_file",
            kind="tool",
            planned_files=["src/auth.py"],
            enforce_leases=True,
            state=_state(session_id="session-requesting"),
        )
        self.assertEqual(adv2["verdict"], "reject")
        self.assertIn("active lease conflict on src/auth.py held by session(s) session-holding-lease", adv2["reason"])

    def test_multi_agent_ipc_bus_and_kernel_step_interrupt(self) -> None:
        parent_sess = "parent-sess-ipc"
        child_sess = "child-sess-ipc"
        st_parent = _state(session_id=parent_sess)

        # 1. Send IPC from child to parent
        sent_msg = self.brain.send_ipc(
            child_sess,
            parent_sess,
            "child_finished",
            {"exit_code": 0, "summary": "subtask complete"},
        )
        self.assertEqual(sent_msg["sender"], child_sess)
        self.assertEqual(sent_msg["recipient"], parent_sess)
        self.assertEqual(sent_msg["signal"], "child_finished")

        # 2. Run kernel_step on parent session
        k_res = self.brain.kernel_step("Continue execution after subtask", state=st_parent)
        self.assertIn("ipc_messages", k_res)
        self.assertEqual(len(k_res["ipc_messages"]), 1)
        self.assertEqual(k_res["ipc_messages"][0]["signal"], "child_finished")

        # 3. Verify interrupt surfaced in interrupts list
        ipc_interrupts = [item for item in k_res.get("interrupts", []) if item.get("kind") == "ipc"]
        self.assertEqual(len(ipc_interrupts), 1)
        self.assertEqual(ipc_interrupts[0]["target"], child_sess)
        self.assertEqual(ipc_interrupts[0]["reason"], "child_finished")

        # 4. Verify subsequent poll_ipc unread_only=True returns empty list
        unread_after = self.brain.poll_ipc(parent_sess, unread_only=True)
        self.assertEqual(unread_after, [])

        # 5. But unread_only=False still returns the history
        all_messages = self.brain.poll_ipc(parent_sess, unread_only=False)
        self.assertEqual(len(all_messages), 1)
        self.assertTrue(all_messages[0]["read"])


class McpServerAiosTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = MemoryService(self.root / "memory.db")
        self.brain = FibBrain(self.service)
        self.server = build_server(self.service)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mcp_fibbrain_context_kernel_routing_and_delta(self) -> None:
        # Remember a memory
        self.brain.remember(
            "decision",
            "AsyncIO event loop uvloop policy",
            "Install uvloop and set event loop policy on Unix production systems.",
            state=_state(),
        )

        async def scenario() -> None:
            # First call with include_kernel_routing=True and delta_only=False
            res1 = await self.server.call_tool(
                "fibbrain_context",
                {
                    "goal": "AsyncIO event loop uvloop policy",
                    "session_id": "mcp-sess-1",
                    "owner": "abbila",
                    "workspace_id": "fibmind",
                    "project_id": "fibmind",
                    "workspace_path": str(self.root),
                    "include_kernel_routing": True,
                    "delta_only": False,
                },
            )
            data1 = _result_json(res1)
            self.assertIn("kernel_routing", data1)
            kr = data1["kernel_routing"]
            self.assertIn(kr["dispatch_mode"], {"system1_reflex", "system2_reason", "interrupt_verify"})
            self.assertIn("mmu", data1)
            self.assertEqual(len(data1["mmu"]["page_faults"]), 1)

            # Second call with delta_only=True
            res2 = await self.server.call_tool(
                "fibbrain_context",
                {
                    "goal": "AsyncIO event loop uvloop policy",
                    "session_id": "mcp-sess-1",
                    "owner": "abbila",
                    "workspace_id": "fibmind",
                    "project_id": "fibmind",
                    "workspace_path": str(self.root),
                    "delta_only": True,
                },
            )
            data2 = _result_json(res2)
            self.assertIn("mmu", data2)
            self.assertTrue(data2["mmu"]["delta_only"])
            self.assertEqual(len(data2["mmu"]["page_hits"]), 1)
            self.assertEqual(len(data2["mmu"]["page_faults"]), 0)

        asyncio.run(scenario())

    def test_mcp_fibbrain_reflect_trust_ring(self) -> None:
        written = self.brain.remember(
            "decision",
            "Zero downtime migration locks",
            "Use CREATE INDEX CONCURRENTLY to avoid exclusive table locks during migrations.",
            state=_state(),
        )
        node_id = written["node_id"]

        async def scenario() -> None:
            reflected = await self.server.call_tool(
                "fibbrain_reflect",
                {
                    "node_id": node_id,
                    "verdict": "confirmed",
                    "source": "ci test",
                    "owner": "abbila",
                    "workspace_id": "fibmind",
                    "project_id": "fibmind",
                    "trust_ring": "ring0",
                },
            )
            data = _result_json(reflected)
            self.assertEqual(data["node_id"], node_id)
            self.assertGreater(data["confidence"], 0.0)
            self.assertEqual(data["confidence_source"], "ci test")

        asyncio.run(scenario())

    def test_mcp_read_resources(self) -> None:
        async def scenario() -> None:
            # 1. Status resource
            res_status = await self.server.read_resource("fibmind://status")
            self.assertTrue(len(res_status) > 0)
            status_json = json.loads(res_status[0].content)
            self.assertIn("store", status_json)
            self.assertIn("nodes", status_json)

            # 2. Hot snapshot resource
            res_hot = await self.server.read_resource("fibmind://hot/abbila/fibmind/fibmind")
            self.assertTrue(len(res_hot) > 0)
            hot_json = json.loads(res_hot[0].content)
            self.assertIn("node_ids", hot_json)

            # 3. Entity profile resource
            res_entity = await self.server.read_resource("fibmind://entity/abbila/fibmind/fibmind/storage.py")
            self.assertTrue(len(res_entity) > 0)
            entity_json = json.loads(res_entity[0].content)
            self.assertEqual(entity_json["entity"], "storage.py")

            # 4. Procedures resource
            res_proc = await self.server.read_resource("fibmind://procedures/abbila/fibmind/fibmind")
            self.assertTrue(len(res_proc) > 0)
            proc_json = json.loads(res_proc[0].content)
            self.assertEqual(proc_json["format"], "skill")

            # 5. MMU resource
            res_mmu = await self.server.read_resource("fibmind://mmu/mcp-sess-1")
            self.assertTrue(len(res_mmu) > 0)
            mmu_json = json.loads(res_mmu[0].content)
            self.assertEqual(mmu_json["session_id"], "mcp-sess-1")
            self.assertIn("pages", mmu_json)
            self.assertIn("pinned_pages", mmu_json)
            self.assertIn("step", mmu_json)

            # 6. Conflicts resource
            res_conf = await self.server.read_resource("fibmind://conflicts/abbila/fibmind/fibmind")
            self.assertTrue(len(res_conf) > 0)
            conf_json = json.loads(res_conf[0].content)
            self.assertIn("conflicts", conf_json)

            # 7. Clusters resource
            res_clust = await self.server.read_resource("fibmind://clusters/abbila/fibmind/fibmind")
            self.assertTrue(len(res_clust) > 0)
            clust_json = json.loads(res_clust[0].content)
            self.assertIn("clusters", clust_json)

            # 8. Node inspect resource
            st_res = _state()
            mem_node = self.brain.remember("decision", "MCP test node", "MCP inspect details", state=st_res)
            nid = mem_node["node_id"]
            res_node = await self.server.read_resource(f"fibmind://node/{nid}")
            self.assertTrue(len(res_node) > 0)
            node_json = json.loads(res_node[0].content)
            self.assertEqual(node_json["node_id"], nid)
            self.assertEqual(node_json["title"], "MCP test node")

            # 9. Plan resource
            self.brain.plan("Test MCP plan resource objective", state=st_res)
            res_plan = await self.server.read_resource("fibmind://plan/abbila/fibmind/fibmind")
            self.assertTrue(len(res_plan) > 0)
            plan_json = json.loads(res_plan[0].content)
            self.assertIn("goals", plan_json)
            self.assertIn("state", plan_json)
            self.assertTrue(any(g["objective"] == "Test MCP plan resource objective" for g in plan_json["goals"]))

            # 10. IPC resource
            self.brain.send_ipc("agent-a", "mcp-sess-1", "ping", {"seq": 42})
            res_ipc = await self.server.read_resource("fibmind://ipc/mcp-sess-1")
            self.assertTrue(len(res_ipc) > 0)
            ipc_json = json.loads(res_ipc[0].content)
            self.assertEqual(ipc_json["session_id"], "mcp-sess-1")
            self.assertIn("messages", ipc_json)
            self.assertTrue(any(m["signal"] == "ping" for m in ipc_json["messages"]))

            # 11. Processes table resource
            self.brain.acquire_lease("mcp-sess-1", ["src/mcp_server.py"], ttl_minutes=15)
            res_proc_tbl = await self.server.read_resource("fibmind://processes/fibmind/fibmind")
            self.assertTrue(len(res_proc_tbl) > 0)
            proc_tbl_json = json.loads(res_proc_tbl[0].content)
            self.assertIn("processes", proc_tbl_json)
            self.assertIn("leases", proc_tbl_json)
            self.assertTrue(any(p["session_id"] == "mcp-sess-1" for p in proc_tbl_json["processes"]))
            self.assertTrue(any(lease_row["target"] == "src/mcp_server.py" for lease_row in proc_tbl_json["leases"]))

        asyncio.run(scenario())


class SessionProcessTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = MemoryService(self.root / "memory.db")
        self.brain = FibBrain(self.service)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fork_and_merge_session_lifecycle(self) -> None:
        # 1. Parent remembers memories and pages them
        m1 = self.brain.remember(
            "decision",
            "Postgres transaction isolation level",
            "Use Read Committed isolation level as default for web request transactions.",
            state=_state(session_id="parent-sess-1"),
        )["node_id"]

        parent_st = _state(session_id="parent-sess-1")
        res_p = self.brain.context("Postgres transaction isolation", state=parent_st)
        self.assertIn(m1, res_p["mmu"]["page_faults"])

        # 2. Fork child session
        fork_res = self.brain.fork_session(
            "parent-sess-1",
            "child-sess-1",
            objective="Audit database deadlock risks",
            state=parent_st,
        )
        self.assertEqual(fork_res["parent_session_id"], "parent-sess-1")
        self.assertEqual(fork_res["child_session_id"], "child-sess-1")
        self.assertEqual(fork_res["inherited_pages"], 1)
        self.assertIn(m1, fork_res["page_ids"])

        # Child session starts with page hits from parent
        child_st = _state(session_id="child-sess-1")
        res_c = self.brain.context("Postgres transaction isolation", state=child_st, delta_only=True)
        self.assertIn(m1, res_c["mmu"]["page_hits"])
        self.assertEqual(len(res_c["mmu"]["page_faults"]), 0)

        # 3. Child discovers a new fact and records an episode event
        self.brain.observe(
            "decision",
            "Deadlock resolution: always order table updates alphabetically by table name.",
            {"rule": "ordered_locks"},
            state=child_st,
        )
        m2 = self.brain.remember(
            "decision",
            "Deadlock lock ordering protocol",
            "Always acquire row locks in deterministic ascending order of primary keys.",
            state=child_st,
        )["node_id"]
        # Child pages m2
        self.brain.context("Deadlock lock ordering protocol", state=child_st)

        # 4. Merge child session back into parent session
        merge_res = self.brain.merge_session(
            "parent-sess-1",
            "child-sess-1",
            mode="auto",
            close_child=True,
            state=parent_st,
        )
        self.assertEqual(merge_res["parent_session_id"], "parent-sess-1")
        self.assertEqual(merge_res["child_session_id"], "child-sess-1")
        self.assertTrue(merge_res["closed_child"])
        self.assertGreaterEqual(merge_res["merged_pages"], 2)

        # Parent's page table now includes both m1 and m2
        parent_pages = self.brain._load_session_pages("parent-sess-1")
        self.assertIn(m1, parent_pages)
        self.assertIn(m2, parent_pages)

        # Parent context call recognizes m2 as a page hit!
        res_p2 = self.brain.context("Deadlock lock ordering protocol", state=parent_st, delta_only=True)
        self.assertIn(m2, res_p2["mmu"]["page_hits"])

    def test_fork_and_merge_session_hierarchical_goals(self) -> None:
        parent_sess = "parent-sess-subgoal"
        child_sess = "child-sess-subgoal"
        st_parent = _state(session_id=parent_sess)

        # 1. Parent plans root goal
        parent_plan = self.brain.plan("Implement microservice authentication", state=st_parent)
        parent_goal_id = parent_plan["goal_id"]

        # 2. Fork child session with objective
        fork_res = self.brain.fork_session(
            parent_sess,
            child_sess,
            objective="Implement OAuth2 token verification",
            state=st_parent,
        )
        self.assertEqual(fork_res["parent_goal_id"], parent_goal_id)
        child_goal_id = fork_res["child_goal_id"]
        self.assertIsNotNone(child_goal_id)

        # 3. Verify graph edge DERIVED_FROM between child goal and parent goal
        edges = self.service.read(
            lambda m: [(e.from_node_id, e.to_node_id, e.relation_type.value) for e in m.edges.values()]
        )
        self.assertIn((child_goal_id, parent_goal_id, "derived_from"), edges)

        # 4. Merge child session
        merge_res = self.brain.merge_session(parent_sess, child_sess, state=st_parent)
        self.assertEqual(merge_res["completed_child_goal_id"], child_goal_id)

        # 5. Verify child goal status is COMPLETE
        st_child = _state(session_id=child_sess)
        child_goal_after = self.brain._resolve_goal(None, child_goal_id, st_child, allow_missing=False)
        self.assertEqual(child_goal_after.status.value, "complete")

        # 6. Verify parent goal steps advanced
        parent_goal_after = self.brain._resolve_goal(None, parent_goal_id, st_parent, allow_missing=False)
        self.assertTrue(
            any(
                s.status.value == "done"
                for s in parent_goal_after.steps
                if s.title.startswith("Do the work:") or s.title == "Reflect and remember"
            )
        )

    def test_review_session_auto_grounds_files_and_symbols(self) -> None:
        target_file = self.root / "auth.py"
        target_file.write_text("def verify_token(token):\n    return bool(token)\n", encoding="utf-8")

        st = _state(session_id="sess-auto-ground")
        self.brain.observe(
            "decision",
            "Implement token verification in auth.py",
            payload={"files": ["auth.py"], "symbols": ["auth.py::verify_token"]},
            state=st,
        )

        reviewed = self.brain.review_session(st, mode="auto", cwd=self.root)
        written = reviewed.get("written", [])
        self.assertGreater(len(written), 0)
        node_id = written[0]["node_id"]
        node_meta = self.service.inspect(node_id)["metadata"]
        self.assertIn("file_hashes", node_meta)
        self.assertIn("auth.py", node_meta["file_hashes"])
        self.assertIn("symbol_hashes", node_meta)
        self.assertIn("auth.py::verify_token", node_meta["symbol_hashes"])

    def test_brain_synthesize_conflict_and_consolidate_clusters(self) -> None:
        # 1. Dialectical synthesis via FibBrain
        st1 = _state(session_id="sess-a")
        st2 = _state(session_id="sess-b")
        st3 = _state(session_id="sess-c")
        n_a = self.brain.remember(
            "decision",
            "Thread pool vs asyncio concurrency",
            "Use ThreadPoolExecutor for CPU-bound or blocking I/O tasks.",
            metadata={"files": ["src/fibmind/worker.py"]},
            state=st1,
        )["node_id"]
        n_b = self.brain.remember(
            "decision",
            "Asyncio concurrency for high-throughput network I/O",
            "Use native asyncio event loop for non-blocking socket I/O.",
            metadata={"files": ["src/fibmind/worker.py"]},
            state=st2,
        )["node_id"]

        synth = self.brain.synthesize_conflict(
            n_a,
            n_b,
            title="Unified concurrency policy for worker.py",
            content="When blocking I/O: ThreadPoolExecutor; when non-blocking network I/O: native asyncio.",
            conditions_a=["when_blocking_io"],
            conditions_b=["when_nonblocking_network_io"],
            source="bench_concurrency",
            state=st1,
        )
        self.assertEqual(synth["title"], "Unified concurrency policy for worker.py")
        self.assertIn("synthesized", synth["tags"])

        # 2. Cross-session verified memories for knowledge cluster discovery in consolidate()
        c1 = self.brain.remember(
            "pattern",
            "SQLite WAL busy timeout configuration",
            "Always configure PRAGMA busy_timeout = 5000 on SQLite connections before queries.",
            metadata={"files": ["src/fibmind/storage.py"]},
            state=st1,
        )["node_id"]
        c2 = self.brain.remember(
            "pattern",
            "SQLite foreign keys enforcement rule",
            "Enable PRAGMA foreign_keys = ON whenever opening a fresh SQLite connection.",
            metadata={"files": ["src/fibmind/storage.py"]},
            state=st2,
        )["node_id"]
        c3 = self.brain.remember(
            "pattern",
            "SQLite immediate transaction locking invariant",
            "Use BEGIN IMMEDIATE transactions for write blocks to avoid SQLITE_BUSY deadlocks.",
            metadata={"files": ["src/fibmind/storage.py"]},
            state=st3,
        )["node_id"]
        self.brain.reflect("confirmed", "pytest tests/test_storage.py", node_id=c1, state=st1)
        self.brain.reflect("confirmed", "pytest tests/test_storage.py", node_id=c2, state=st2)
        self.brain.reflect("confirmed", "pytest tests/test_storage.py", node_id=c3, state=st3)

        consolidated = self.brain.consolidate(state=st1, workspace_root=self.root)
        self.assertIn("knowledge_clusters", consolidated)
        self.assertGreaterEqual(len(consolidated["knowledge_clusters"]), 1)
        cluster_ids = set(consolidated["knowledge_clusters"][0]["supporting_node_ids"])
        self.assertTrue({c1, c2, c3}.issubset(cluster_ids))

    def test_dynamic_plan_step_progression(self) -> None:
        st = _state(session_id="fsm-sess-1")
        # 1. Create an unverified memory with verify command and a conflict pair
        m_uncertain = self.brain.remember(
            "decision",
            "Verify cache eviction policy",
            "Ensure LRU cache evicts oldest keys first.",
            metadata={"verify": "pytest tests/test_cache.py"},
            state=st,
        )["node_id"]
        c_a = self.brain.remember(
            "decision",
            "Write-through cache strategy",
            "Write synchronously to database and cache.",
            state=st,
        )["node_id"]
        c_b = self.brain.remember(
            "decision",
            "Write-back cache strategy",
            "Write asynchronously to database from cache.",
            state=st,
        )["node_id"]
        self.service.mark_conflict(c_a, c_b, reason="contradictory cache write policies", identity=st.identity())

        # 2. Plan objective -> "Recall related history" becomes DONE, next step becomes ACTIVE
        planned = self.brain.plan("Verify cache eviction policy and write-through cache strategy", state=st)
        goal_id = planned["goal_id"]
        steps_by_title = {s["title"]: s["status"] for s in planned["steps"]}
        self.assertEqual(steps_by_title.get("Recall related history"), "done")
        self.assertTrue(any(s["status"] == "active" for s in planned["steps"]))

        # 3. Resolve conflict -> "Resolve: ..." step becomes DONE
        self.brain.resolve_conflict(c_a, c_b, reason="Write-through required for durability", source="audit", state=st)
        goal_after_resolve = self.brain._resolve_goal(None, goal_id, st, allow_missing=False)
        resolve_steps = [s for s in goal_after_resolve.steps if s.title.startswith("Resolve:")]
        for rs in resolve_steps:
            self.assertEqual(rs.status.value, "done")

        # 4. Reflect confirmed -> "Verify..." steps become DONE
        self.brain.reflect("confirmed", "pytest tests/test_cache.py", node_id=m_uncertain, state=st)
        goal_after_reflect = self.brain._resolve_goal(None, goal_id, st, allow_missing=False)
        verify_steps = [s for s in goal_after_reflect.steps if s.title.startswith("Verify")]
        self.assertTrue(len(verify_steps) > 0)
        for vs in verify_steps:
            self.assertEqual(vs.status.value, "done")

        # 5. Remember -> "Do the work: ..." and "Reflect and remember" become DONE
        self.brain.remember(
            "decision",
            "Finalized cache eviction and write-through policy",
            "Verified LRU eviction with synchronous write-through durability.",
            state=st,
        )
        goal_after_rem = self.brain._resolve_goal(None, goal_id, st, allow_missing=False)
        for s in goal_after_rem.steps:
            if s.title.startswith("Do the work:") or s.title == "Reflect and remember":
                self.assertEqual(s.status.value, "done")

        # 6. Complete goal -> all steps DONE and goal status COMPLETE
        completed = self.brain.complete_goal(goal_id=goal_id, state=st)
        self.assertEqual(completed["status"], "complete")
        self.assertTrue(all(s["status"] == "done" for s in completed["steps"]))

    def test_kernel_step_interrupt_vector_table(self) -> None:
        st = _state(session_id="ivt-sess-1")
        # 1. Refuted memory -> conflict interrupt
        m_ref = self.brain.remember(
            "decision",
            "Legacy MD5 password hashing",
            "Use MD5 for hashing user passwords.",
            state=st,
        )["node_id"]
        self.brain.reflect("refuted", "security audit", node_id=m_ref, state=st)

        # 2. Uncertain memory with verify command -> verify interrupt
        self.brain.remember(
            "decision",
            "Argon2id password hashing",
            "Use Argon2id with memory cost 64MB for password hashing.",
            metadata={"verify": "pytest tests/test_auth.py"},
            state=st,
        )

        res = self.brain.kernel_step("Legacy MD5 and Argon2id password hashing", state=st)
        self.assertIn("interrupts", res)
        kinds = {item["kind"] for item in res["interrupts"]}
        self.assertIn("conflict", kinds)
        self.assertIn("verify", kinds)

    def test_kernel_explicit_leases_acquire_conflict_and_auto_release(self) -> None:
        st_a = _state(session_id="sess-worker-a")
        st_b = _state(session_id="sess-worker-b")

        # 1. Session A acquires lease on storage.py
        acq_a = self.brain.acquire_lease(
            "sess-worker-a",
            ["src/fibmind/storage.py"],
            ttl_minutes=20,
            reason="Refactoring storage engine",
            state=st_a,
        )
        self.assertTrue(acq_a["acquired"])
        self.assertEqual(len(acq_a["leases"]), 1)
        self.assertEqual(acq_a["leases"][0]["target"], "src/fibmind/storage.py")

        # 2. Session B attempts to acquire conflicting lease on storage.py
        acq_b_conflict = self.brain.acquire_lease("sess-worker-b", ["storage.py"], ttl_minutes=15, state=st_b)
        self.assertFalse(acq_b_conflict["acquired"])
        self.assertGreaterEqual(len(acq_b_conflict["conflicts"]), 1)
        self.assertEqual(acq_b_conflict["conflicts"][0]["requested_target"], "storage.py")

        # 3. Session B succeeds on non-conflicting target
        acq_b_ok = self.brain.acquire_lease("sess-worker-b", ["evals/gate.py"], ttl_minutes=15, state=st_b)
        self.assertTrue(acq_b_ok["acquired"])

        # 4. Advise rejects when enforce_leases=True for conflicting session
        advise_b = self.brain.advise(
            "modify storage",
            kind="tool",
            planned_files=["src/fibmind/storage.py"],
            enforce_leases=True,
            state=st_b,
        )
        self.assertEqual(advise_b["verdict"], "reject")
        self.assertIn("active lease conflict", advise_b["reason"])
        self.assertIn("sess-worker-a", advise_b["reason"])

        # 5. Process table query returns active processes and leases
        ptable = self.brain.process_table(workspace_id="fibmind", project_id="fibmind")
        self.assertGreaterEqual(ptable["total_processes"], 2)
        sess_ids = {p["session_id"] for p in ptable["processes"]}
        self.assertIn("sess-worker-a", sess_ids)
        self.assertIn("sess-worker-b", sess_ids)
        self.assertGreaterEqual(len(ptable["leases"]), 2)

        # 6. Session B auto-releases lease on review_session with close=True
        self.brain.observe("decision", "Evaluated gate benchmarks", state=st_b)
        self.brain.review_session(st_b, mode="auto", close=True)
        ptable_after_b = self.brain.process_table()
        b_leases = [lease_row for lease_row in ptable_after_b["leases"] if lease_row["session_id"] == "sess-worker-b"]
        self.assertEqual(len(b_leases), 0)

        # 7. Session A releases lease manually
        rel_a = self.brain.release_lease("sess-worker-a")
        self.assertEqual(rel_a["released"], 1)
        self.assertEqual(len(self.brain._load_kernel_leases()), 0)

    def test_child_session_sigchld_ipc_bubbling_on_block_and_merge(self) -> None:
        parent_sess = "parent-sigchld-1"
        child_sess = "child-sigchld-1"
        st_parent = _state(session_id=parent_sess)
        st_child = _state(session_id=child_sess)

        # 1. Fork child session
        self.brain.fork_session(
            parent_sess,
            child_sess,
            objective="Execute critical migration",
            state=st_parent,
        )
        self.assertEqual(self.brain._load_session_parent(child_sess), parent_sess)

        # 2. Block child session in coordinate via a refuted memory constraint
        m_blocker = self.brain.remember(
            "decision",
            "Disallow database migration on replica",
            "Refuted: migrations must never run on read replicas",
            state=st_child,
        )["node_id"]
        self.brain.reflect("refuted", "audit", node_id=m_blocker, state=st_child)

        coord = self.brain.coordinate("Disallow database migration on replica", state=st_child)
        self.assertEqual(coord["status"], "blocked")

        # 3. Verify SIGCHLD child_blocked signal bubbled to parent
        parent_inbox = self.brain.poll_ipc(parent_sess, unread_only=True)
        self.assertTrue(any(m["signal"] == "child_blocked" and m["sender"] == child_sess for m in parent_inbox))

        # 4. Now child session completes and is merged into parent
        merge_res = self.brain.merge_session(parent_sess, child_sess, close_child=True, state=st_parent)
        self.assertEqual(merge_res["parent_session_id"], parent_sess)

        # 5. Verify SIGCHLD child_merged signal bubbled to parent
        parent_inbox_merged = self.brain.poll_ipc(parent_sess, unread_only=True)
        self.assertTrue(any(m["signal"] == "child_merged" and m["sender"] == child_sess for m in parent_inbox_merged))

    def test_consolidate_synthesis_proposals_and_auto_promote_clusters(self) -> None:
        st = _state(session_id="sess-consolidate-synth")

        # 1. Create two actively contradicting memories
        n1 = self.brain.remember(
            "decision",
            "Use SQLite WAL journal mode",
            "When in write concurrency mode: use WAL journal mode for SQLite.",
            metadata={"files": ["src/fibmind/storage.py"]},
            state=st,
        )["node_id"]
        n2 = self.brain.remember(
            "decision",
            "Use SQLite DELETE journal mode",
            "When on network filesystem storage: use DELETE journal mode for SQLite.",
            metadata={"files": ["src/fibmind/storage.py"]},
            state=st,
        )["node_id"]
        self.service.mark_conflict(n1, n2, reason="Opposing SQLite journal modes", identity=st.identity())

        # 2. Create 3 supporting observations across sessions for knowledge clustering
        s1 = _state(session_id="sess-induct-1")
        s2 = _state(session_id="sess-induct-2")
        s3 = _state(session_id="sess-induct-3")
        k1 = self.brain.remember(
            "pattern",
            "Postgres connection pool max overflow",
            "Configure max_overflow to 10 for spike traffic tolerance.",
            metadata={"files": ["src/db.py"]},
            state=s1,
        )["node_id"]
        k2 = self.brain.remember(
            "pattern",
            "Postgres connection pool recycle duration",
            "Always recycle idle pool connections after 1800 seconds.",
            metadata={"files": ["src/db.py"]},
            state=s2,
        )["node_id"]
        k3 = self.brain.remember(
            "pattern",
            "Postgres connection pool pre-ping healthcheck",
            "Enable pool_pre_ping=True to discard stale dropped connections.",
            metadata={"files": ["src/db.py"]},
            state=s3,
        )["node_id"]
        self.brain.reflect("confirmed", "unit test", node_id=k1, state=s1)
        self.brain.reflect("confirmed", "unit test", node_id=k2, state=s2)
        self.brain.reflect("confirmed", "unit test", node_id=k3, state=s3)

        # 3. Call consolidate with auto_promote_clusters=True
        cons = self.brain.consolidate(state=st, auto_promote_clusters=True)
        self.assertIn("synthesis_proposals", cons)
        self.assertGreaterEqual(len(cons["synthesis_proposals"]), 1)
        prop = next(p for p in cons["synthesis_proposals"] if {p["node_id_a"], p["node_id_b"]} == {n1, n2})
        self.assertIn("suggested_title", prop)
        self.assertIn("suggested_content", prop)
        self.assertIn("conditions_a", prop)
        self.assertIn("conditions_b", prop)

        self.assertIn("knowledge_clusters", cons)
        self.assertIn("promoted_knowledge", cons)
        self.assertGreaterEqual(len(cons["knowledge_clusters"]), 1)
        self.assertGreaterEqual(len(cons["promoted_knowledge"]), 1)

    def test_assess_metacognition_and_revalidate_facade(self) -> None:
        st = _state(session_id="sess-assess-meta")
        assessed = self.brain.assess("Implement connection pooling", state=st)
        self.assertIn("verdict", assessed)
        self.assertIn("known", assessed)

        reval = self.brain.revalidate_memories(state=st, older_than_days=90)
        self.assertIn("checked", reval)
        self.assertIn("recovered", reval)
        self.assertIn("still_drifted", reval)
        self.assertIn("executed", reval)

    def test_kernel_evaluation_runner(self) -> None:
        from evals.kernel_runner import run_kernel_evaluation

        scores = run_kernel_evaluation()
        self.assertEqual(scores["dao_epistemology_score"], 1.0)
        self.assertEqual(scores["qi_grounding_score"], 1.0)
        self.assertEqual(scores["aios_mmu_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
