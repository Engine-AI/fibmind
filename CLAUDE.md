# FibMind Memory

FibMind is available as an MCP server providing long-term memory tools.

## Before a task

Call `fibmind_context` with the user's goal to recall relevant history. Use only
the concise retrieved context — do not dump full memory into the conversation.

## After a task

Call `fibmind_append` with:

- task goal
- files changed
- important decisions
- errors encountered
- verification commands

## Whenever evidence appears

Call `fibmind_record_outcome` with `verdict` (`confirmed` / `refuted`) and a
`source` naming what checked it. A stored memory starts unproven, and reading it
back never makes it more trusted — recording outcomes is the only thing that
changes how memories rank.

Evidence to record:

- a fix you recorded made the tests pass → `confirmed`, source = the test command
- a decision was later reverted → `refuted`, source = the revert commit
- the user corrected advice you had stored → `refuted`, source = the correction

In this repo the cheap checks are `pytest` and the demo script; prefer citing a
command that anyone can rerun over a summary of your own judgement.

## Upkeep

- `fibmind_revise` when the same memory should be corrected and made active
  again — prefer this over appending a contradiction. Confidence resets.
- `fibmind_mark_stale` when a memory has been superseded but its provenance
  should remain available. Normal recall excludes stale memories.
- `fibmind_forget` when the user asks for removal, or something was captured by
  mistake. It also redacts log history, so the content does not come back.
- `fibmind_promote_knowledge` when the same pattern has appeared in three or more
  separate memories and the general form is worth reusing. Write it at a level
  that changes what you would do next; if it is too abstract to act on, skip it.

## Lookup

`fibmind_search` for specific past work by keyword; `fibmind_search_from` to
expand the association tree around a returned `node_id`.
