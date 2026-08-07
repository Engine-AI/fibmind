from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fibmind import EdgeDirection, FibMind, JsonStore, RelationType, Verdict


def main() -> None:
    memory = FibMind()

    code_bug = memory.append(
        "code",
        title="Login API returns 401",
        content="The login endpoint returns 401 after token expiry.",
        tags={"login", "auth", "bug"},
        owner="dev-a",
    )
    requirement = memory.append(
        "requirement",
        title="Refresh token automatically",
        content="The product requires silent token refresh before API calls fail.",
        tags={"login", "auth", "token"},
        owner="dev-a",
    )
    fix = memory.append(
        "code",
        title="Add request interceptor",
        content="Generate a request interceptor that refreshes tokens before retrying requests.",
        tags={"login", "auth", "fix"},
        owner="dev-a",
    )

    memory.link_nodes(code_bug, requirement, RelationType.CAUSED_BY, weight=0.85)
    memory.link_nodes(fix, code_bug, RelationType.ANSWER_TO, weight=0.95)
    memory.link_nodes(fix, requirement, RelationType.DERIVED_FROM, weight=0.9)
    memory.link_nodes(
        code_bug, fix, RelationType.RELATED_TO, weight=0.7, direction=EdgeDirection.BIDIRECTIONAL
    )

    # Recall does not make a memory more trusted; only an outside check does.
    # Skip this step and the fix ranks no better than an unverified guess, however
    # often it gets read back.
    memory.record_outcome(
        fix,
        Verdict.CONFIRMED,
        source="pytest tests/test_auth.py::test_silent_refresh",
        note="interceptor removed the 401 retry loop",
    )
    memory.promote_node(code_bug)

    print("Search from fix node:")
    for hit in memory.search_from(fix, depth=2):
        relation = hit.via_relation.value if hit.via_relation else "self"
        print(f"- depth={hit.depth} via={relation}: {hit.node.title} [{hit.node.node_type}]")

    print("\nRanked recall for 'login token expiry':")
    for hit in memory.search("login token expiry", owner="dev-a"):
        node = hit.node
        print(f"- {hit.score:.2f} confidence={node.confidence:.2f} {node.title}")

    # Three independent observations of one failure mode are what license a shared
    # claim. With fewer than three, promotion is refused.
    timeout_reports = [
        memory.append(
            "code",
            title=f"Upload times out ({owner})",
            content="Large uploads time out because the client has no retry budget.",
            tags={"upload", "timeout"},
            owner=owner,
        )
        for owner in ("dev-a", "dev-b", "dev-c")
    ]
    claim = memory.promote_to_knowledge(
        title="Uploads need an explicit retry budget",
        content=(
            "Clients uploading large files must set a retry budget; without one the "
            "request dies at the first transient timeout."
        ),
        supporting_node_ids=timeout_reports,
        tags={"upload", "timeout"},
    )
    claim_node = memory.nodes[claim]
    print(f"\nPromoted to shared knowledge: {claim_node.title}")
    print(f"  scope={claim_node.scope.value} confidence={claim_node.confidence}")
    print("  provenance:")
    for edge in memory.edges.values():
        if edge.from_node_id == claim and edge.relation_type == RelationType.DERIVED_FROM:
            print(f"    <- {memory.nodes[edge.to_node_id].title}")

    # dev-b sees the shared claim but none of dev-a's personal memories.
    print("\nWhat dev-b can recall:")
    for hit in memory.search("login upload timeout", top_k=10, owner="dev-b"):
        print(f"- [{hit.node.scope.value}] {hit.node.title}")

    # The log is the source of truth; nodes and edges are a cache built from it.
    replayed = FibMind.rebuild_from_log(memory.events)
    print(f"\nEvents logged: {len(memory.events)}")
    print(
        "Replayed from log: "
        f"{len(replayed.nodes)} nodes, {len(replayed.edges)} edges, "
        f"{len(replayed.trees)} trees"
    )

    output_path = PROJECT_ROOT / "data" / "demo-memory.json"
    JsonStore(output_path).save(memory)
    print(f"\nSaved demo memory to {output_path}")


if __name__ == "__main__":
    main()
