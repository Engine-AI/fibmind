# FibMind Memory

FibMind is available as an MCP server providing long-term memory tools.

Before starting a multi-step task, call `fibmind_context` with the user's goal
to recall relevant history. Use only the concise retrieved context — do not dump
full memory into the conversation.

After finishing a task, call `fibmind_append` with:

- task goal
- files changed
- important decisions
- errors encountered
- verification commands

Use `fibmind_search` to look up specific past work by keyword, and
`fibmind_search_from` to expand the association tree around a returned node_id.
