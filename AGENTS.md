# FibBrain

FibBrain is the cognitive runtime. FibMind is the memory engine underneath.

In this repo pass these identity fields on every `fibbrain_*` call:

- `owner`: `abbila` (the human user, never the host name such as `codex` / `claude-code` / `agy`). Put the host name in `metadata.host`. The MCP server also pins this via `FIBMIND_OWNER`, so an omitted or host-named owner is mapped to it.
- `workspace_id`: `fibmind`
- `project_id`: `fibmind`
- `session_id`: the current conversation session id
- `task_id`: the task name the user gave, otherwise the current git branch (`git branch --show-current`, e.g. `main`, `feature/hybrid-retrieval`). Every AI tool must derive it the same way: the same `task_id` is what lets Codex, AGY, and Claude Code see each other's episode observations and active goal.

For non-trivial tasks:

1. Call `fibbrain_plan` with the objective and the current `workspace_id`,
   `project_id`, and `session_id`. Then call `fibbrain_recall` if you need the
   context pack itself. Use only the concise retrieved context.
2. Call `fibbrain_coordinate` to see which capabilities to involve next, or
   `fibbrain_advise` before a specific risky tool if past evidence might say
   not to use it.
3. After implementation, call `fibbrain_remember` (not raw `fibmind_append`)
   with objective, key decisions, changed files, tests, and unresolved risks.
   The admit gate skips duplicates, dumps, and empty speculation.
4. When evidence appears, call `fibbrain_reflect` with `verdict`
   (`confirmed` / `refuted`) and a `source` naming what checked it. Memories
   start unproven; recorded outcomes are the only thing that affects ranking.

Upkeep:

- `fibmind_revise` when the same memory should be corrected and made active
  again, rather than appending a contradiction. Confidence resets to zero.
- `fibmind_mark_stale` when a memory has been superseded but its provenance
  should remain available. Normal recall excludes stale memories.
- `fibmind_forget` for memories the user wants removed or that were captured by
  mistake. It redacts log history too, so the content does not return.
- `fibmind_promote_knowledge` once the same pattern has appeared in three or more
  memories and the general form is worth reusing. Keep it concrete enough to
  change what you would do next.
