# FibBrain

FibBrain is the cognitive runtime. FibMind is the memory engine underneath.

In this repo pass these identity fields on every `fibbrain_*` call:

- `owner`: the human user if known, else `claude-code`
- `workspace_id`: `fibmind`
- `project_id`: `fibmind`
- `session_id`: the current conversation or task id if you have one

## Before a task

Call `fibbrain_plan` with the user's objective and the current `workspace_id`,
`project_id`, and `session_id`. Call `fibbrain_recall` when you need the context
pack itself. Use only the concise retrieved context — do not dump full memory
into the conversation.

## Before acting

Call `fibbrain_coordinate` to see which capabilities to involve next; when it
returns `procedures`, follow their steps. Call `fibbrain_render(goal)` to get
those procedures as SKILL.md text or tool definitions. Call `fibbrain_advise`
when past evidence might say a specific tool should not run.

## During a task

Call `fibbrain_observe` with the identity fields as things happen: `kind` of
`decision`, `test` (with `payload.command`), `error`, `correction`, `risk`, or
`tool_result` (with `payload.files` for changed paths). With a `session_id` the
episode is persisted and only this session recalls it.

## After a task

Call `fibbrain_review_session(session_id, mode="approve")` first. It distils
the episode into pending memories deterministically; approve the good ones
with `fibbrain_approve_memory` and reject the rest. Running it twice writes
nothing new.

Then call `fibbrain_remember` (not raw `fibmind_append`) for anything the
review could not derive, with:

- task goal
- files changed
- important decisions
- errors encountered
- verification commands

The admit gate skips duplicates, raw dumps, and empty speculation.

When a task turned out to be a repeatable recipe, call
`fibbrain_remember_procedure` with the trigger, ordered steps, tools, and how
to verify. After using a procedure, `fibbrain_reflect` on its node_id with
`confirmed` or `refuted`; that is what promotes or retires it.

## Whenever evidence appears

Call `fibbrain_reflect` with `verdict` (`confirmed` / `refuted`) and a
`source` naming what checked it. A stored memory starts unproven, and reading it
back never makes it more trusted — recording outcomes is the only thing that
changes how memories rank.

Evidence to record:

- a fix you recorded made the tests pass → `confirmed`, source = the test command
- a decision that was later reverted → `refuted`, source = the revert commit
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

`fibbrain_recall` is the default. `fibmind_search` / `fibmind_search_from` remain
available for raw store inspection. `fibbrain_observe` notes an episode event
that is not yet long-term memory; `fibbrain_review_session` decides what is.
