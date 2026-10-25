"""Protect existing host settings while registering the shared memory server."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fibmind.hosts import END, START, connect, load_bundle, managed_instructions, registration_command
from fibmind.integrations import server_config, write_bundle


class HostSetupTests(unittest.TestCase):
    def test_preview_includes_existing_claude_local_override_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(directory / "bundle", owner="abbila", store=directory / "shared.db")
            config = directory / ".claude.json"
            original = json.dumps({"projects": {str(Path.cwd()): {"mcpServers": {"fibmind": {"env": {}}}}}})
            config.write_text(original)
            with patch("fibmind.hosts.Path.home", return_value=directory), patch.dict("os.environ", {"CODEX_HOME": ""}):
                preview = connect(directory / "bundle")
            command = preview["commands"]["claude-code-local"]
            self.assertEqual(command[command.index("--scope") + 1], "local")
            self.assertEqual(json.loads(command[-1])["env"]["FIBMIND_OWNER"], "abbila")
            self.assertEqual(config.read_text(), original)
            self.assertFalse((directory / ".claude" / "CLAUDE.md").exists())

    def test_managed_block_updates_preserve_other_instructions(self) -> None:
        original = "Keep my existing rules.\n"
        added = managed_instructions(original, "First memory instructions")
        with_suffix = added + "\nKeep my later rules.\n"
        updated = managed_instructions(with_suffix, "Updated memory instructions")
        self.assertTrue(updated.startswith(original))
        self.assertTrue(updated.endswith("\nKeep my later rules.\n"))
        self.assertNotIn("First memory instructions", updated)
        self.assertEqual(updated.count(START), 1)
        self.assertEqual(managed_instructions(updated, "Updated memory instructions"), updated)

    def test_malformed_managed_blocks_are_never_overwritten(self) -> None:
        for existing in (START, END, END + START, START + START + END):
            with self.subTest(existing=existing), self.assertRaises(ValueError):
                managed_instructions(existing, "replacement")

    def test_native_registration_uses_argv_and_claude_user_scope(self) -> None:
        server = server_config("claude-code", owner='用户 "name"', store='memory/with "quotes".db')
        command = registration_command("claude-code", server)
        self.assertEqual(command[:6], ["claude", "mcp", "add-json", "--scope", "user", "fibmind"])
        self.assertEqual(json.loads(command[6]), server)
        for host in ("codex", "agy"):
            server = server_config(host, owner="abbila", store="memory.db")
            command = registration_command(host, server)
            self.assertLess(command.index("--env"), command.index("fibmind"))
            self.assertEqual(command[command.index("--") + 1 :], [server["command"], *server["args"]])

    def test_mixed_store_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(directory, owner="abbila", store=directory / "shared.db")
            self.assertEqual(len(load_bundle(directory)), 3)
            path = directory / "agy.json"
            content = json.loads(path.read_text())
            args = content["mcpServers"]["fibmind"]["args"]
            args[args.index("--store") + 1] = str(directory / "different.db")
            path.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                load_bundle(directory)
