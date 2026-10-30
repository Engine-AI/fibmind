"""Evaluation runner for AI-OS Cognitive Kernel (Track 3).

Measures three core cognitive runtime dimensions:
1. dao_epistemology_score:
   - Ring 3 confidence cap (0.60)
   - Ring 0 physical confirmation breakthrough (> 0.60)
   - Dialectical conflict synthesis (Dao dialectics)
   - Cross-session knowledge cluster discovery
2. qi_grounding_score:
   - AST-level symbol hashing stability (code vs comment edits)
   - Auto-grounding metadata capture
   - Transitive drift propagation across knowledge graphs
3. aios_mmu_score:
   - Cognitive MMU delta-only page hits and prompt token economy
   - Dirty page invalidation on memory revisions
   - Bounded Cognitive TLB step TTL and capacity eviction
   - Session process tree fork and merge state propagation
   - Structured kernel_step Interrupt Vector Table (IVT)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fibmind.brain import BrainState, FibBrain
from fibmind.entity import build_entity_profile
from fibmind.graph import FibMind
from fibmind.models import MemoryScope, RelationType, Verdict
from fibmind.reflex import auto_ground_metadata, compute_symbol_hashes
from fibmind.service import MemoryService


def evaluate_dao_epistemology(tmp_path: Path) -> float:
    checks_passed = 0
    total_checks = 4

    # 1. Ring 3 confidence cap
    mind = FibMind()
    nid = mind.append(
        "decision",
        "Architecture Clean Boundaries",
        "Maintain clean module boundaries.",
        scope=MemoryScope.PERSONAL,
        owner="abbila",
    )
    for _ in range(4):
        mind.record_outcome(nid, Verdict.CONFIRMED, source="self_eval", trust_ring="ring3")
    if mind.nodes[nid].confidence == 0.60:
        checks_passed += 1

    # 2. Ring 0 physical confirmation breakthrough
    mind.record_outcome(nid, Verdict.CONFIRMED, source="pytest", trust_ring="ring0")
    if mind.nodes[nid].confidence > 0.60:
        checks_passed += 1

    # 3. Dialectical conflict synthesis
    db_path = tmp_path / "dao_store.db"
    service = MemoryService(db_path)
    brain = FibBrain(service)
    st = BrainState(owner="abbila", workspace_id="fibmind", project_id="fibmind", session_id="s1")
    na = brain.remember("decision", "Concurrency threadpool", "Use threadpool for blocking I/O", state=st)["node_id"]
    nb = brain.remember("decision", "Concurrency async", "Use native asyncio for network I/O", state=st)["node_id"]
    service.mark_conflict(na, nb, reason="contradictory concurrency styles", identity=st.identity())
    synth = brain.synthesize_conflict(
        na,
        nb,
        title="Unified concurrency policy",
        content="Use threadpool for blocking I/O; native asyncio for network I/O.",
        conditions_a=["blocking"],
        conditions_b=["network"],
        source="dialectical_synthesis",
        state=st,
    )
    if synth.get("node_id") is not None and "synthesized" in synth.get("tags", []):
        checks_passed += 1

    # 4. Cross-session knowledge cluster discovery
    st2 = BrainState(owner="abbila", workspace_id="fibmind", project_id="fibmind", session_id="s2")
    st3 = BrainState(owner="abbila", workspace_id="fibmind", project_id="fibmind", session_id="s3")
    c1 = brain.remember(
        "pattern",
        "WAL timeout config",
        "PRAGMA busy_timeout=5000",
        metadata={"files": ["src/storage.py"]},
        state=st,
    )["node_id"]
    c2 = brain.remember(
        "pattern",
        "WAL foreign keys",
        "PRAGMA foreign_keys=ON",
        metadata={"files": ["src/storage.py"]},
        state=st2,
    )["node_id"]
    c3 = brain.remember(
        "pattern",
        "WAL immediate locks",
        "BEGIN IMMEDIATE",
        metadata={"files": ["src/storage.py"]},
        state=st3,
    )["node_id"]
    brain.reflect("confirmed", "pytest 1", node_id=c1, state=st)
    brain.reflect("confirmed", "pytest 2", node_id=c2, state=st2)
    brain.reflect("confirmed", "pytest 3", node_id=c3, state=st3)
    clusters_res = service.discover_knowledge_clusters(identity=st.identity(), auto_promote=False)
    if len(clusters_res.get("clusters", [])) > 0:
        checks_passed += 1

    return checks_passed / total_checks


def evaluate_qi_grounding(tmp_path: Path) -> float:
    checks_passed = 0
    total_checks = 3

    # 1. AST symbol hashes stability (comment edit vs body edit)
    code_file = tmp_path / "worker.py"
    code_file.write_text("def compute_hash(val):\n    return str(val)\n", encoding="utf-8")
    h1 = compute_symbol_hashes(["worker.py::compute_hash"], cwd=tmp_path)
    code_file.write_text(
        "# Header comment\ndef compute_hash(val):\n    return str(val)\n\ndef unrelated():\n    return 42\n",
        encoding="utf-8",
    )
    h2 = compute_symbol_hashes(["worker.py::compute_hash"], cwd=tmp_path)
    code_file.write_text("def compute_hash(val):\n    return int(val)\n", encoding="utf-8")
    h3 = compute_symbol_hashes(["worker.py::compute_hash"], cwd=tmp_path)
    if (
        "worker.py::compute_hash" in h1
        and h1.get("worker.py::compute_hash") == h2.get("worker.py::compute_hash")
        and h1.get("worker.py::compute_hash") != h3.get("worker.py::compute_hash")
    ):
        checks_passed += 1

    # 2. Auto-grounding metadata capture
    code_file.write_text("def compute_hash(val):\n    return str(val)\n", encoding="utf-8")
    raw_meta = {"files": ["worker.py"], "symbols": ["worker.py::compute_hash"]}
    grounded = auto_ground_metadata(raw_meta, cwd=tmp_path)
    if (
        grounded is not None
        and "worker.py" in grounded.get("file_hashes", {})
        and "worker.py::compute_hash" in grounded.get("symbol_hashes", {})
    ):
        checks_passed += 1

    # 3. Transitive drift propagation
    qi_mem = FibMind()
    n1 = qi_mem.append(
        "decision",
        "Core architecture core.py",
        "Maintain core architecture in core.py",
        metadata={"files": ["core.py"]},
        scope=MemoryScope.PERSONAL,
        owner="abbila",
        workspace_id="fibmind",
        project_id="fibmind",
    )
    n2 = qi_mem.append(
        "decision",
        "Helper invariant",
        "Depends on missing file",
        metadata={"file_hashes": {"missing.py": "abc123expected"}},
        scope=MemoryScope.PERSONAL,
        owner="abbila",
        workspace_id="fibmind",
        project_id="fibmind",
    )
    qi_mem.link_nodes(n1, n2, RelationType.DERIVED_FROM)
    prof = build_entity_profile(
        qi_mem,
        "core.py",
        identity={"owner": "abbila", "workspace_id": "fibmind", "project_id": "fibmind"},
        cwd=tmp_path,
    )
    if any(d.get("node_id") == n1 and d.get("via_node_id") == n2 for d in prof.get("drift", [])):
        checks_passed += 1

    return checks_passed / total_checks


def evaluate_aios_mmu(tmp_path: Path) -> float:
    checks_passed = 0
    total_checks = 8

    mmu_db = tmp_path / "mmu_store.db"
    mmu_brain = FibBrain(MemoryService(mmu_db), max_tlb_pages=3, tlb_ttl_steps=2)
    st_mmu = BrainState(owner="abbila", workspace_id="fibmind", project_id="fibmind", session_id="sess_mmu")

    # 1. MMU delta_only page hits
    mid = mmu_brain.remember("decision", "Postgres pool size", "Use 20 connections max.", state=st_mmu)["node_id"]
    r1 = mmu_brain.context("Postgres pool size", state=st_mmu, delta_only=False)
    r2 = mmu_brain.context("Postgres pool size", state=st_mmu, delta_only=True)
    if mid in r1["mmu"]["page_faults"] and mid in r2["mmu"]["page_hits"] and "Postgres pool size" not in r2["text"]:
        checks_passed += 1

    # 2. Dirty page invalidation on revise
    mmu_brain.memory.revise(mid, content="Updated to 50 connections max.", identity=st_mmu.identity())
    r3 = mmu_brain.context("Postgres pool size", state=st_mmu)
    if mid in [d.get("node_id") for d in r3["mmu"]["dirty_pages"]]:
        checks_passed += 1

    # 3. TLB step TTL eviction
    m1 = mmu_brain.remember("decision", "Alpha engine", "Alpha details", state=st_mmu)["node_id"]
    mmu_brain.remember("decision", "Beta cache", "Beta details", state=st_mmu)
    st_ttl = BrainState(owner="abbila", workspace_id="fibmind", project_id="fibmind", session_id="sess_ttl")
    mmu_brain.context("Alpha engine", state=st_ttl)
    mmu_brain.context("Beta cache", state=st_ttl)
    mmu_brain.context("Gamma query non-matching", state=st_ttl)
    r_ttl = mmu_brain.context("Alpha engine", state=st_ttl, delta_only=True)
    if m1 in r_ttl["mmu"].get("evicted_pages", []) and m1 in r_ttl["mmu"].get("page_faults", []):
        checks_passed += 1

    # 4. Fork and merge session
    forked = mmu_brain.fork_session("sess_ttl", "sess_child")
    merged = mmu_brain.merge_session("sess_ttl", "sess_child")
    if forked.get("child_session_id") == "sess_child" and merged.get("parent_session_id") == "sess_ttl":
        checks_passed += 1

    # 5. kernel_step interrupt vector
    m_ref = mmu_brain.remember("decision", "Old Redis strategy", "Never use Redis cluster", state=st_mmu)["node_id"]
    mmu_brain.reflect("refuted", "Redis cluster is standard now", node_id=m_ref, state=st_mmu)
    k_step = mmu_brain.kernel_step("Old Redis strategy", state=st_mmu)
    if "interrupts" in k_step and any(
        item.get("kind") == "conflict" and item.get("target") == m_ref for item in k_step["interrupts"]
    ):
        checks_passed += 1

    # 6. Page pinning protects from TTL eviction & hierarchical sub-goal fork/merge
    m_pin = mmu_brain.remember("decision", "Pinned engine", "Pinned details", state=st_mmu)["node_id"]
    st_pin = BrainState(owner="abbila", workspace_id="fibmind", project_id="fibmind", session_id="sess_pin")
    mmu_brain.pin_page("sess_pin", m_pin)
    mmu_brain.context("Pinned engine", state=st_pin)
    mmu_brain.context("Beta cache", state=st_pin)
    mmu_brain.context("Gamma query non-matching", state=st_pin)
    r_pin = mmu_brain.context("Pinned engine", state=st_pin, delta_only=True)
    pin_protected = m_pin in r_pin["mmu"].get("page_hits", []) and m_pin not in r_pin["mmu"].get("evicted_pages", [])

    mmu_brain.plan("Parent session goal", state=st_pin)
    fork_sub = mmu_brain.fork_session("sess_pin", "sess_pin_child", objective="Child goal")
    merge_sub = mmu_brain.merge_session("sess_pin", "sess_pin_child")
    goal_completed = fork_sub.get("child_goal_id") is not None and merge_sub.get(
        "completed_child_goal_id"
    ) == fork_sub.get("child_goal_id")
    if pin_protected and goal_completed:
        checks_passed += 1

    # 7. Demand paging (page_in) & multi-agent IPC signal bus
    m_paged = mmu_brain.remember("decision", "Demand paged item", "Demand paging content", state=st_mmu)["node_id"]
    page_in_res = mmu_brain.page_in("sess_mmu", [m_paged], pin=True, state=st_mmu)
    paged_ok = any(
        item.get("node_id") == m_paged for item in page_in_res.get("paged_in", [])
    ) and m_paged in page_in_res.get("mmu", {}).get("pinned_pages", [])
    mmu_brain.send_ipc("sess_child", "sess_mmu", "ready", {"step": 1})
    polled = mmu_brain.poll_ipc("sess_mmu", unread_only=True)
    ipc_ok = len(polled) == 1 and polled[0]["signal"] == "ready" and polled[0]["sender"] == "sess_child"
    if paged_ok and ipc_ok:
        checks_passed += 1

    # 8. Explicit leases, process table, and lease release
    lease_res1 = mmu_brain.acquire_lease("sess_worker_1", ["src/core.py"], ttl_minutes=10, state=st_mmu)
    lease_res2 = mmu_brain.acquire_lease("sess_worker_2", ["src/core.py"], ttl_minutes=10, state=st_mmu)
    lease_conflict_ok = lease_res1.get("acquired") is True and lease_res2.get("acquired") is False
    proc_tbl = mmu_brain.process_table(workspace_id="fibmind", project_id="fibmind")
    proc_ok = (
        any(p.get("session_id") == "sess_worker_1" for p in proc_tbl.get("processes", []))
        and len(proc_tbl.get("leases", [])) >= 1
    )
    mmu_brain.release_lease("sess_worker_1")
    rel_ok = len(mmu_brain._load_kernel_leases()) == 0
    if lease_conflict_ok and proc_ok and rel_ok:
        checks_passed += 1

    return checks_passed / total_checks


def run_kernel_evaluation() -> dict[str, float]:
    """Run full cognitive kernel evaluation across Dao, Qi, and AIOS MMU dimensions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        return {
            "dao_epistemology_score": evaluate_dao_epistemology(tmp_path),
            "qi_grounding_score": evaluate_qi_grounding(tmp_path),
            "aios_mmu_score": evaluate_aios_mmu(tmp_path),
        }


if __name__ == "__main__":
    results = run_kernel_evaluation()
    for key, score in sorted(results.items()):
        print(f"kernel.{key} = {score:.4f}")
