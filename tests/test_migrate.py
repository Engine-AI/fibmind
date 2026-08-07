"""Tests for JSON-to-SQLite migration."""

import tempfile
import unittest
from pathlib import Path

from fibmind import FibMind, JsonStore, SqliteStore
from fibmind.migrate import migrate_json_to_sqlite


class MigrationTests(unittest.TestCase):
    def test_migrate_json_to_sqlite(self) -> None:
        memory = FibMind()
        node_id = memory.append("代码", "登录故障", "令牌过期")

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "memory.json"
            target = Path(tmp) / "memory.db"
            JsonStore(source).save(memory)

            counts = migrate_json_to_sqlite(source, target)
            restored = SqliteStore(target).load()

            self.assertEqual(counts["nodes"], len(memory.nodes))
            self.assertIn(node_id, restored.nodes)

    def test_migration_refuses_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "memory.json"
            target = Path(tmp) / "memory.db"
            JsonStore(source).save(FibMind())
            target.touch()

            with self.assertRaises(FileExistsError):
                migrate_json_to_sqlite(source, target)


if __name__ == "__main__":
    unittest.main()
