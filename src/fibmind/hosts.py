"""Preview and apply native host registrations from a generated config bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fibmind.integrations import HOSTS

START = "<!-- fibmind:context:start -->"
END = "<!-- fibmind:context:end -->"


def managed_instructions(existing: str, body: str) -> str:
    block = f"{START}\n{body.strip()}\n{END}"
    if not existing.count(START) and not existing.count(END):
        return existing.rstrip() + ("\n\n" if existing.strip() else "") + block + "\n"
    if existing.count(START) != 1 or existing.count(END) != 1 or existing.index(START) > existing.index(END):
        raise ValueError("Malformed existing FibMind instruction block; refusing to overwrite it")
    left, right = existing.index(START), existing.index(END) + len(END)
    return existing[:left] + block + existing[right:]


def load_bundle(directory: Path) -> dict[str, dict[str, Any]]:
    servers = {}
    for host in HOSTS:
        if host == "codex":
            server = tomllib.loads((directory / "codex.toml").read_text())["mcp_servers"]["fibmind"]
        else:
            server = json.loads((directory / f"{host}.json").read_text())["mcpServers"]["fibmind"]
        if server.get("args", [])[:2] != ["-m", "fibmind.mcp_server"]:
            raise ValueError("Bundle must launch the FibMind Python module")
        if not Path(server["command"]).is_absolute() or not Path(server["command"]).is_file():
            raise ValueError("Bundle interpreter must be an existing absolute path")
        args = server["args"]
        store = args[args.index("--store") + 1]
        owner = args[args.index("--owner") + 1]
        if not Path(store).is_absolute() or owner != server["env"].get("FIBMIND_OWNER"):
            raise ValueError("Bundle requires an absolute store and consistent human owner")
        if server["env"].get("FIBMIND_APPLICATION") != host:
            raise ValueError("Bundle application provenance does not match its host")
        servers[host] = server
    signatures = {(s["command"], tuple(s["args"])) for s in servers.values()}
    if len(signatures) != 1:
        raise ValueError("All hosts must use the same interpreter, store, and owner")
    return servers


def registration_command(host: str, server: dict[str, Any]) -> list[str]:
    if host == "claude-code":
        return ["claude", "mcp", "add-json", "--scope", "user", "fibmind", json.dumps(server, ensure_ascii=False)]
    program = "codex" if host == "codex" else "agy"
    args = [program, "mcp", "add"]
    for key, value in server["env"].items():
        args.extend(["--env", f"{key}={value}"])
    return [*args, "fibmind", "--", server["command"], *server["args"]]


def instruction_targets(user_directory: Path) -> dict[str, Path]:
    codex_root = Path(os.environ.get("CODEX_HOME") or user_directory / ".codex")
    override = codex_root / "AGENTS.override.md"
    return {
        "codex": override if override.is_file() and override.stat().st_size else codex_root / "AGENTS.md",
        "claude-code": user_directory / ".claude" / "CLAUDE.md",
        "agy": user_directory / ".gemini" / "GEMINI.md",
    }


def connect(directory: Path, *, apply: bool = False) -> dict[str, Any]:
    directory = directory.expanduser().resolve()
    servers = load_bundle(directory)
    user_directory = Path.home()
    targets = instruction_targets(user_directory)
    body = (directory / "memory-instructions.md").read_text(encoding="utf-8")
    # Validate every managed edit before invoking a host or writing anything.
    edits = {
        host: managed_instructions(path.read_text(encoding="utf-8") if path.exists() else "", body)
        for host, path in targets.items()
    }
    commands = {host: registration_command(host, server) for host, server in servers.items()}
    claude_config = user_directory / ".claude.json"
    old_claude = (
        json.loads(claude_config.read_text()) if claude_config.is_file() and claude_config.stat().st_size else {}
    )
    local_servers = old_claude.get("projects", {}).get(str(Path.cwd()), {}).get("mcpServers", {})
    if local_servers.get("fibmind"):
        command = registration_command("claude-code", servers["claude-code"])
        command[command.index("--scope") + 1] = "local"
        commands["claude-code-local"] = command
    preview = {
        "commands": commands,
        "instruction_files": {host: str(path) for host, path in targets.items()},
        "applied": False,
    }
    if not apply:
        return preview
    for command in commands.values():
        if shutil.which(command[0]) is None:
            raise ValueError(f"Host CLI is not installed: {command[0]}")
    backup = directory / "backups" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    backup.mkdir(parents=True)
    paths = [
        *targets.values(),
        targets["codex"].parent / "config.toml",
        user_directory / ".claude.json",
        user_directory / ".gemini" / "config" / "mcp_config.json",
        user_directory / ".gemini" / "antigravity" / "mcp_config.json",
    ]
    manifest = []
    for index, path in enumerate(paths):
        item = {"path": str(path), "existed": path.is_file()}
        if path.is_file():
            saved = backup / f"{index}-{path.name}"
            shutil.copy2(path, saved)
            saved.chmod(0o600)
            item["backup"] = str(saved)
        manifest.append(item)
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Claude's native add-json refuses an existing named server. Its current
    # project-local entry can override user scope, so sync that entry too.
    # Other projects and all other named entries remain untouched.
    for host, command in commands.items():
        if host.startswith("claude-code"):
            scope = "local" if host.endswith("-local") else "user"
            previous = (
                local_servers.get("fibmind") if scope == "local" else old_claude.get("mcpServers", {}).get("fibmind")
            )
            if previous == servers["claude-code"]:
                continue
            if previous:
                subprocess.run(
                    ["claude", "mcp", "remove", "--scope", scope, "fibmind"],
                    check=True,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    for host, path in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(edits[host], encoding="utf-8")
    return {
        "applied": True,
        "hosts": list(servers),
        "instruction_files": {host: str(path) for host, path in targets.items()},
        "backup": str(backup),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preview or apply a generated FibMind host integration bundle")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Register fibmind through each native CLI and install managed context instructions",
    )
    args = parser.parse_args(argv)
    try:
        print(json.dumps(connect(args.bundle, apply=args.apply), indent=2, ensure_ascii=False))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Host setup failed: {exc}\nBackups, if created, remain in the bundle's backups directory.\n")


if __name__ == "__main__":
    main()
