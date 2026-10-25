# Cross-application user context

Status: implemented. Protocol checks and native Codex context read verified.
Native application limitations are recorded below separately.

The product goal is a shared memory manager for Codex, Claude Code,
Antigravity (AGY), and other harnesses. On invocation it supplies relevant
stored memories and knowledge as background about the human and their work.
The calling harness keeps control of its own conversation and execution.

## Acceptance

1. Independent MCP clients use one absolute store path and the same human
   owner; the originating application is provenance, not an owner.
2. Explicit user-wide profile/preferences follow that owner across projects.
   Workspace context stays in its workspace; project memories stay in their
   project; task scratch keeps the existing session/task boundary. Legacy
   unlabelled memories retain their current visibility.
3. A read-only `fibbrain_context` entry point combines applicable profile
   and task-relevant facts/knowledge. It does not create plans, snapshots,
   episodes, or command executions. Internal planning records and stale,
   refuted, pending, conflicted, or unsafe memories are not background.
4. The rendered background has a token budget, evidence/provenance labels,
   and only reports memories actually rendered. No relevant facts is a valid
   result; a weak word overlap must not manufacture task knowledge.
5. SQLite filters visible lexical candidates before truncation, so other
   projects or stale memories cannot crowd relevant memories out.
6. Generate usable, owner-pinned Codex, Claude Code, and AGY MCP configuration
   with an absolute interpreter/store and portable context-use instructions.
   Preserve unrelated host configuration and existing memory data.
7. Verify independent real MCP connections with distinct client/session names,
   persistence across server restarts, profiles across projects, shared
   knowledge, owner/project/task isolation, budgets, and read-only operation.
   Distinguish protocol tests from a live application connection check.

The existing cognitive tools remain available to integrations using them.
Broader visibility is explicit (`context_level=user|workspace`) and is retained
through persistence, compression, and event replay.

## Validation, 2026-10-25

- 344 non-daemon tests and 8 daemon tests pass, with 43 subtests. Ruff lint,
  formatting, the retrieval evaluation floor, and diff whitespace checks pass.
- Independent stdio MCP processes share one store, user-level background,
  project facts, and application provenance. Context reads leave the semantic
  revision unchanged; a fresh process reads previously written memories.
- All three installed native CLIs register the same absolute database, human
  owner, and project-private interpreter. Current-project Claude overrides are
  synced. Existing settings and instructions are backed up; unrelated Claude
  settings and effective Codex settings are preserved. Codex's native writer
  normalizes an unrelated empty `args=[]` by omitting it.
- Native Codex calls `fibbrain_context` successfully and reads the confirmed
  SQLite runtime upgrade record. This CLI did not expose its conversation ID;
  omitting it leaves session scratch inaccessible while profile/project reads
  still work. Instructions explicitly forbid fabricating a conversation ID.
- Native Claude Code reports the FibMind server connected and lists the new
  context tool. Its model service returns HTTP 402 (no valid plan/balance), so
  a live model-directed context call is not verified.
- Native AGY attempts the exact context tool but headless mode soft-denies its
  MCP permission. An optional grant for only that read-only tool is pending
  explicit user approval; no blanket permission bypass is used.
- The real database passed `integrity_check`; SQLite remains exactly 3.51.3.
  Before normalization, 48 memories belonged to known tool/host aliases. A
  SQLite online backup was made, those owners were reassigned to `abbila`,
  host provenance was retained, and live versus event-replayed identities
  matched. Unowned legacy records were left unchanged.

Local native check logs and configuration backups are under
`.fibmind/integrations/`; they are deliberately not versioned.
