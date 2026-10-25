# Cross-application memory

FibMind supplies historical user context to Codex, Claude Code, Antigravity
(AGY), and other MCP harnesses. Each harness retains its own conversation,
planning, permissions, and tool execution. `fibbrain_context` is the normal
entry point; the older cognitive workflow remains available with `--profile full`.

## Connect the same human and store

Use the Python environment containing FibMind and one absolute SQLite path.
For this checkout, keep the existing database:

```bash
.venv/bin/fibmind-config --owner abbila --store .fibmind/memory.db \
  --output-dir .fibmind/integrations
.venv/bin/fibmind-connect --bundle .fibmind/integrations
# Apply the reviewed preview through the installed native CLIs:
.venv/bin/fibmind-connect --bundle .fibmind/integrations --apply
```

The generator defaults to `--profile memory`, exposing context, admission,
reflection, retrieval, and upkeep without the planning/execution tools. This
repository's existing AGENTS workflow requires `--profile full` instead.
All three configs pin the human owner and preserve the venv executable path.
`FIBMIND_APPLICATION` records the source application, never a different owner.
The store path is absolute, so different launch directories share the same data.

The connector previews commands by default. Applying uses `codex mcp add`,
`claude mcp add-json --scope user`, and `agy mcp add`. It backs up existing
settings in the bundle's `backups/` directory before changing the named
`fibmind` registration and a marked instruction block. Other server entries,
models, and approval settings remain intact. A failed native CLI can leave a
partial registration; the backup manifest lists the original files for recovery.
An existing Claude Code local `fibmind` entry in the current project is synced
too, because it takes precedence over user scope. Other projects are untouched.
Reopen existing host sessions to load their new server/instructions. Project
MCP settings may take precedence over a global registration.

Native configuration references:
[Codex MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli),
[Claude Code MCP](https://code.claude.com/docs/en/mcp), and
[Antigravity MCP](https://antigravity.google/docs/mcp).

## Request background

Before handling the user's request, call:

```json
{
  "goal": "the current user request",
  "owner": "abbila",
  "workspace_path": "/absolute/path/to/the/current/project",
  "session_id": "the calling host's actual conversation ID",
  "budget_tokens": 500
}
```

The absolute canonical git root becomes `workspace_id`, its basename becomes
`project_id`, and the git branch becomes `task_id` unless an explicit label
overrides it. Same-named projects at different paths remain separate. A root
`.fibmind-project.json` can preserve established labels, as this repository does:

```json
{"workspace_id": "fibmind", "project_id": "fibmind"}
```

Reuse returned `state` for later calls. The response's `text` contains standing
profile/preferences and relevant task facts. `memories` contains only rendered
IDs with evidence and source-host metadata. `has_task_context=false` can still
include useful profile; it means no task facts were rendered. The budget bounds
rendered text using the configured token counter, not the full JSON response or
provider billing. Current user instructions outrank stored historical context.
`confirmed` describes past external evidence, not a fresh check of today's state.
If a host does not expose its native conversation ID, omit `session_id` and
avoid session-scope operations. User/project background remains usable; a
fabricated or reused conversation ID would break scratch-state isolation.

Context does not create goals, hot snapshots, episodes, or ranking evidence.
It excludes inactive, pending, conflicted, unsafe, and internal planning records.
An optional embedding provider may update its derived retrieval cache.

## Save at an explicit level

Use `fibbrain_remember` with the returned identity and one of:

| Scope and level | Visibility |
| --- | --- |
| `scope=personal`, `context_level=user` | Same human across all workspaces/projects |
| `scope=personal`, `context_level=workspace` | Same human and workspace |
| `scope=personal`, `context_level=project` | Same human, workspace, and project |
| `scope=session`, no personal level | Existing session/task boundary |
| `scope=knowledge` | Shared claim created through promotion from evidence |

Use category `profile` or `preference` for durable user background. Other
categories hold project facts, decisions, and errors. Legacy unlabelled records
retain their existing exact-match visibility; they are never silently broadened.
Application names belong in `metadata.host`. Do not store secrets or raw logs.
Revise or retire changed facts, and confirm/refute using actual named evidence.

## Verification

`tests/test_cross_app_mcp.py` launches independent real stdio MCP server
processes from different directories with distinct client/session names. It
checks shared persistence, user profile across projects, provenance, isolation,
and read-only background. This is a protocol integration test. A live Codex,
Claude Code, or AGY model invocation is a separate application check and may
depend on local authentication, model availability, and approval configuration.

Antigravity headless mode soft-denies tools that require interactive approval.
For unattended context reads, an explicitly approved rule may grant only
`mcp(fibmind/fibbrain_context)` in `permissions.allow` under
`~/.gemini/antigravity-cli/settings.json`. This does not grant memory writes or
other tools. Existing deny/ask rules take precedence. See the official
[headless permissions guide](https://antigravity.google/docs/cli/headless/).
