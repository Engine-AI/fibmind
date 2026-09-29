"""Identity checks on node-ID operations and compression boundaries."""

from pathlib import Path

import pytest

from fibmind import BrainState, FibBrain, FibMind, MemoryScope, MemoryService

ALICE = BrainState(owner="alice", workspace_id="ws", project_id="project", session_id="a")
BOB = BrainState(owner="bob", workspace_id="ws", project_id="project", session_id="b")


def test_node_id_does_not_bypass_identity_for_mutations(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "memory.db")
    brain = FibBrain(service)
    private = brain.remember("decision", "Retry budget", "Retry idempotent uploads three times", state=ALICE)["node_id"]
    other = brain.remember("decision", "Bob decision", "Use an explicit timeout", state=BOB)["node_id"]

    with pytest.raises(ValueError, match="not visible"):
        brain.reflect("refuted", "Bob's assertion", node_id=private, state=BOB)
    with pytest.raises(ValueError, match="not visible"):
        service.search_from(private, owner="bob", workspace_id="ws", project_id="project")
    for mutate in (
        lambda: service.revise(private, content="corrupted", identity=BOB.identity()),
        lambda: service.mark_stale(private, "corrupted", identity=BOB.identity()),
        lambda: service.forget(private, "corrupted", identity=BOB.identity()),
        lambda: service.link(private, other, "related_to", identity=BOB.identity()),
    ):
        with pytest.raises(ValueError, match="not visible"):
            mutate()

    assert service.inspect(private)["status"] == "active"
    assert service.inspect(private)["content"] == "Retry idempotent uploads three times"
    assert not service.read(
        lambda memory: any({edge.from_node_id, edge.to_node_id} == {private, other} for edge in memory.edges.values())
    )


def test_goal_and_review_node_ids_require_the_callers_identity(tmp_path: Path) -> None:
    brain = FibBrain(MemoryService(tmp_path / "memory.db"))
    planned = brain.plan("Keep the retry budget", state=ALICE)
    with pytest.raises(ValueError, match="not visible"):
        brain.complete_goal(goal_id=planned["goal_id"], state=BOB)

    brain.observe("decision", "keep the retry budget", state=ALICE)
    reviewed = brain.review_session(ALICE, mode="approve")
    pending_id = reviewed["written"][0]["node_id"]
    with pytest.raises(ValueError, match="not visible"):
        brain.approve_memory(pending_id, state=BOB)
    assert brain.memory.inspect(pending_id)["status"] == "pending"


def test_session_scratch_is_not_merged_during_compression() -> None:
    memory = FibMind()
    for number in range(22):
        session = "a" if number % 2 == 0 else "b"
        marker = "redsession" if session == "a" else "bluesession"
        memory.append(
            "scratch",
            f"{marker} {number}",
            f"{marker} only content {number}",
            scope=MemoryScope.SESSION,
            owner="alice",
            workspace_id="ws",
            project_id="project",
            session_id=session,
            auto_link=False,
        )

    summaries = [node for node in memory.nodes.values() if node.metadata.get("source_node_ids")]
    assert summaries
    for summary in summaries:
        sources = [memory.nodes[node_id] for node_id in summary.metadata["source_node_ids"]]
        assert {source.session_id for source in sources} == {summary.session_id}
    for session, foreign in (("a", "bluesession"), ("b", "redsession")):
        visible = (
            node
            for node in memory.nodes.values()
            if memory.is_visible(node, owner="alice", workspace_id="ws", project_id="project", session_id=session)
        )
        assert all(foreign not in node.content for node in visible)
