# FibBrain instructions for a dsh model

Add this block to your dsh model instructions (a project `AGENTS.md` under the
dsh working directory is picked up by `dsh-agent-instructions`). It mirrors the
`CLAUDE.md` / `AGENTS.md` shipped in this repository, with dsh tool names.

---

## Memory (FibBrain)

Tools named `mcp__fibbrain__fibbrain_*` are your long-term memory. On every
call pass the identity fields:

- `owner`: the human user if known, else `dsh`
- `workspace_id`: the git root directory name of the current workspace
- `project_id`: the same as `workspace_id` unless the project is a sub-folder
- `session_id`: the current dsh session id if you know it, else omit

**Before a task:** call `mcp__fibbrain__fibbrain_plan` with the objective, then
`mcp__fibbrain__fibbrain_recall` with the same goal. Use only the concise
context it returns; do not paste whole memories into the conversation.

**Before a risky action:** call `mcp__fibbrain__fibbrain_advise` with the tool
name. A `reject` verdict means past evidence says this action failed here.

**After a task:** call `mcp__fibbrain__fibbrain_remember` with the goal, files
changed, decisions, errors, and the exact verification commands. It skips
duplicates and raw dumps.

**When evidence appears:** call `mcp__fibbrain__fibbrain_reflect` with
`verdict` (`confirmed` / `refuted`) and a `source` naming what checked it, for
example the test command that passed or the user's correction.

**When the user asks you to remember something:** call
`mcp__fibbrain__fibbrain_remember`. When historical information may be
relevant, call `mcp__fibbrain__fibbrain_recall` first and use the results.
