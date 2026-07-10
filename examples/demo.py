from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fibmind import EdgeDirection, FibMind, JsonStore, RelationType


def main() -> None:
    memory = FibMind()

    code_bug = memory.append(
        "code",
        title="Login API returns 401",
        content="The login endpoint returns 401 after token expiry.",
        tags={"login", "auth", "bug"},
    )
    requirement = memory.append(
        "requirement",
        title="Refresh token automatically",
        content="The product requires silent token refresh before API calls fail.",
        tags={"login", "auth", "token"},
    )
    fix = memory.append(
        "code",
        title="Add request interceptor",
        content="Generate a request interceptor that refreshes tokens before retrying requests.",
        tags={"login", "auth", "fix"},
    )

    memory.link_nodes(code_bug, requirement, RelationType.CAUSED_BY, weight=0.85)
    memory.link_nodes(fix, code_bug, RelationType.ANSWER_TO, weight=0.95)
    memory.link_nodes(fix, requirement, RelationType.DERIVED_FROM, weight=0.9)
    memory.link_nodes(code_bug, fix, RelationType.RELATED_TO, weight=0.7, direction=EdgeDirection.BIDIRECTIONAL)

    memory.promote_node(code_bug)

    print("Search from fix node:")
    for hit in memory.search_from(fix, depth=2):
        relation = hit.via_relation.value if hit.via_relation else "self"
        print(f"- depth={hit.depth} via={relation}: {hit.node.title} [{hit.node.node_type}]")

    output_path = PROJECT_ROOT / "data" / "demo-memory.json"
    JsonStore(output_path).save(memory)
    print(f"\nSaved demo memory to {output_path}")


if __name__ == "__main__":
    main()
