"""Command-line store maintenance: JSON → SQLite migration and owner unification."""

from __future__ import annotations

import argparse
from pathlib import Path

from fibmind.models import NodeType
from fibmind.storage import JsonStore, SqliteStore, open_store

# Owner values earlier AGENTS.md / CLAUDE.md told each host to use for itself.
# Memories stored under them are invisible to every other AI tool, which is
# what ``unify_owner`` repairs.
DEFAULT_HOST_OWNERS: tuple[str, ...] = (
    "codex",
    "claude-code",
    "claude",
    "agy",
    "antigravity",
    "gemini",
    "cursor",
    "localhost",
)


def migrate_json_to_sqlite(source: str | Path, target: str | Path) -> dict[str, int]:
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        raise FileNotFoundError(f"JSON store does not exist: {source_path}")
    if target_path.exists():
        raise FileExistsError(f"SQLite target already exists: {target_path}")

    memory = JsonStore(source_path).load()
    SqliteStore(target_path).save(memory)
    return {
        "nodes": len(memory.nodes),
        "edges": len(memory.edges),
        "trees": len(memory.trees),
    }


def unify_owner(
    store: str | Path,
    to_owner: str,
    from_owners: list[str] | tuple[str, ...] = DEFAULT_HOST_OWNERS,
    *,
    workspace_id: str | None = None,
    project_id: str | None = None,
    include_unowned: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Reassign memories written under host names to one human ``to_owner``.

    Each change is a ``reassign_owner`` log event, so a replay reproduces it;
    the previous owner is kept as ``metadata.host``. ``workspace_id`` /
    ``project_id`` restrict the change to one working context, and
    ``include_unowned`` also claims owner-less memories inside that context
    (it requires ``workspace_id`` so unrelated projects are left alone).
    """
    to_owner = to_owner.strip()
    if not to_owner:
        raise ValueError("to_owner must be a non-empty owner id")
    if include_unowned and workspace_id is None:
        raise ValueError("include_unowned requires workspace_id")
    sources = {owner for owner in from_owners if owner and owner != to_owner}

    def selected(memory) -> list[str]:  # type: ignore[no-untyped-def]
        ids = []
        for node in memory.nodes.values():
            if node.node_type == NodeType.ROOT:
                continue
            if workspace_id is not None and node.workspace_id != workspace_id:
                continue
            if project_id is not None and node.project_id != project_id:
                continue
            if node.owner in sources or (include_unowned and node.owner is None):
                ids.append(node.id)
        return sorted(ids)

    backing = open_store(store)
    if dry_run:
        memory = backing.load()
        ids = selected(memory)
        by_owner: dict[str, int] = {}
        for node_id in ids:
            key = memory.nodes[node_id].owner or "<none>"
            by_owner[key] = by_owner.get(key, 0) + 1
        return {"to_owner": to_owner, "reassigned": 0, "would_reassign": len(ids), "by_owner": by_owner}

    by_owner = {}
    with backing.transaction() as memory:
        ids = selected(memory)
        for node_id in ids:
            key = memory.nodes[node_id].owner or "<none>"
            by_owner[key] = by_owner.get(key, 0) + 1
            memory.reassign_owner(node_id, to_owner, f"unify owner for cross-tool sharing ({key} -> {to_owner})")
    return {"to_owner": to_owner, "reassigned": len(ids), "by_owner": by_owner}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fibmind-migrate",
        description="Migrate a FibMind JSON store to SQLite, or unify memory owners across AI tools.",
    )
    parser.add_argument("--from-json", dest="source")
    parser.add_argument("--to-sqlite", dest="target")
    parser.add_argument("--store", help="Store to rewrite with --unify-owner")
    parser.add_argument("--unify-owner", dest="to_owner", help="Owner every host-named memory should move to")
    parser.add_argument(
        "--from-owners",
        default=",".join(DEFAULT_HOST_OWNERS),
        help="Comma-separated owners to merge (default: known AI host names)",
    )
    parser.add_argument("--workspace-id")
    parser.add_argument("--project-id")
    parser.add_argument("--include-unowned", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.to_owner is None and (args.source is None or args.target is None):
        parser.error("either --from-json/--to-sqlite or --store/--unify-owner is required")
    if args.to_owner is not None and args.store is None:
        parser.error("--unify-owner requires --store")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.to_owner is not None:
        result = unify_owner(
            args.store,
            args.to_owner,
            [owner.strip() for owner in args.from_owners.split(",") if owner.strip()],
            workspace_id=args.workspace_id,
            project_id=args.project_id,
            include_unowned=args.include_unowned,
            dry_run=args.dry_run,
        )
        print(result)
        return
    counts = migrate_json_to_sqlite(args.source, args.target)
    print(f"Migrated {counts['nodes']} nodes, {counts['edges']} edges, and {counts['trees']} trees to {args.target}")


if __name__ == "__main__":
    main()
