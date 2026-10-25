"""Portable configuration and workspace identity shared by all harnesses."""

from __future__ import annotations

import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from fibmind.integrations import HOSTS, render_config, server_config, write_bundle
from fibmind.mcp_server import IdentityDefaults
from fibmind.working_context import working_context


class ConfigurationTests(unittest.TestCase):
    def test_all_hosts_point_at_one_absolute_store_and_keep_the_venv_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "venv" / "bin" / "python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            configs = []
            for host in HOSTS:
                server = server_config(host, owner="abbila", store=root / "shared.db", python=executable)
                rendered = render_config(host, server)
                if host == "codex":
                    parsed = tomllib.loads(rendered)["mcp_servers"]["fibmind"]
                else:
                    parsed = json.loads(rendered)["mcpServers"]["fibmind"]
                self.assertEqual(parsed, server)
                self.assertEqual(parsed["command"], str(executable))
                self.assertEqual(parsed["env"]["FIBMIND_OWNER"], "abbila")
                self.assertEqual(parsed["env"]["FIBMIND_APPLICATION"], host)
                configs.append(parsed)
            self.assertEqual(len({tuple(config["args"]) for config in configs}), 1)

    def test_escaping_unicode_paths_and_existing_project_labels(self) -> None:
        server = server_config(
            "codex",
            owner='张三 "owner"',
            store='用户记忆/with "quotes".db',
            workspace_id="old-workspace",
            project_id="old-project",
        )
        parsed = tomllib.loads(render_config("codex", server))["mcp_servers"]["fibmind"]
        self.assertEqual(parsed, server)
        defaults = IdentityDefaults.from_env(parsed["env"])
        resolved = defaults.apply({}, {"owner", "metadata", "workspace_id", "project_id"})
        self.assertEqual(resolved["metadata"]["host"], "codex")
        self.assertEqual(resolved["workspace_id"], "old-workspace")
        self.assertEqual(resolved["project_id"], "old-project")

    def test_generated_files_do_not_touch_host_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root / "config.toml"
            sentinel.write_text('model = "keep-my-model"\n')
            paths = write_bundle(root / "generated", owner="abbila", store=root / "shared.db")
            self.assertEqual(
                {path.name for path in paths}, {"codex.toml", "claude-code.json", "agy.json", "memory-instructions.md"}
            )
            self.assertEqual(sentinel.read_text(), 'model = "keep-my-model"\n')
            instructions = (root / "generated" / "memory-instructions.md").read_text()
            self.assertIn("fibbrain_context", instructions)
            self.assertIn("workspace_path", instructions)
            self.assertIn('context_level="user"', instructions)

    def test_application_names_are_not_human_owners(self) -> None:
        for owner in ("", "codex", "claude-code", "agy"):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                server_config("codex", owner=owner, store="shared.db")


class WorkingContextTests(unittest.TestCase):
    def test_declared_project_labels_preserve_existing_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".fibmind-project.json").write_text(
                json.dumps({"workspace_id": "existing-workspace", "project_id": "existing-project"})
            )
            (root / "src").mkdir()
            identity = working_context(str(root / "src"))
            self.assertEqual(identity["workspace_id"], "existing-workspace")
            self.assertEqual(identity["project_id"], "existing-project")

    def test_git_root_symlinks_and_subdirectories_have_the_same_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "project"
            (root / ".git").mkdir(parents=True)
            (root / ".git" / "HEAD").write_text("ref: refs/heads/feature/context\n")
            (root / "src").mkdir()
            alias = Path(tmp) / "linked-project"
            alias.symlink_to(root, target_is_directory=True)
            expected = {"workspace_id": str(root), "project_id": "project", "task_id": "feature/context"}
            for path in (root, root / "src", alias):
                self.assertEqual(working_context(str(path)), expected)

    def test_same_basename_projects_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / parent / "api" for parent in ("first", "second")]
            for path in paths:
                path.mkdir(parents=True)
            first, second = (working_context(str(path)) for path in paths)
            self.assertNotEqual(first["workspace_id"], second["workspace_id"])

    def test_worktree_branch_and_detached_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "checkout").mkdir()
            (root / "git-data").mkdir()
            (root / "checkout" / ".git").write_text("gitdir: ../git-data\n")
            head = root / "git-data" / "HEAD"
            head.write_text("ref: refs/heads/worktree-branch\n")
            self.assertEqual(working_context(str(root / "checkout"))["task_id"], "worktree-branch")
            head.write_text("1234567890abcdef1234567890abcdef12345678\n")
            self.assertEqual(working_context(str(root / "checkout"))["task_id"], "detached:1234567890ab")

    def test_invalid_workspace_is_rejected(self) -> None:
        for value in ("relative/path", "/nonexistent-fibmind-workspace"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                working_context(value)
