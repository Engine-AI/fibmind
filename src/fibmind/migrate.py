"""Command-line migration from a FibMind JSON store to SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path

from fibmind.storage import JsonStore, SqliteStore


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fibmind-migrate",
        description="Migrate a FibMind JSON memory store to SQLite.",
    )
    parser.add_argument("--from-json", required=True, dest="source")
    parser.add_argument("--to-sqlite", required=True, dest="target")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    counts = migrate_json_to_sqlite(args.source, args.target)
    print(
        "Migrated "
        f"{counts['nodes']} nodes, {counts['edges']} edges, and {counts['trees']} trees "
        f"to {args.target}"
    )


if __name__ == "__main__":
    main()
