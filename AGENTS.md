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
