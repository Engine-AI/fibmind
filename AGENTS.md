# FibMind Memory

FibMind is available as an MCP server providing long-term memory tools.

For non-trivial tasks:

1. Query FibMind before planning if the task may depend on previous work
   (`fibmind_context` with the goal, or `fibmind_search` for a specific lookup).
2. Use only concise retrieved context, not full memory dumps.
3. After implementation, append a memory record with `fibmind_append`:
   - objective
   - key decisions
   - changed files
   - tests run
   - unresolved risks
4. When you learn whether a stored memory was right, call
   `fibmind_record_outcome` with `verdict` (`confirmed` / `refuted`) and a
   `source` naming what checked it — a test command, a revert commit, a user
   correction. Memories start unproven, and reading one back never makes it more
   trusted; recorded outcomes are the only thing that affects ranking.

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
