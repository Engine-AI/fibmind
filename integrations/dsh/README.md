# @fibmind/dsh-fibbrain — FibBrain as a native dsh plugin

A Cordis plugin (~350 lines of plain ESM JavaScript, no build step) that
mounts FibBrain into [DeepSeek-Harness](https://github.com/deepseek-ai/deepseek-harness)
through the harness's own lifecycle events. The Python FibBrain server stays
the brain; this is the thin shell the plan promised in
[`PLAN-evolve.md`](../../PLAN-evolve.md) (S6).

| dsh event / service      | What the plugin does                                                                        | FibBrain call               |
| ------------------------ | ------------------------------------------------------------------------------------------- | --------------------------- |
| `agent/session-start`    | Injects the session's frozen hot memory as a `<fibbrain-hot>` context message               | `fibbrain_recall(include_hot)` |
| `agent/pre-step`         | For each fresh user prompt, appends a token-budgeted `<fibbrain-recall>` pack to the step  | `fibbrain_recall`           |
| `tools/pre-execute`      | Asks whether the tool call should run; refuted evidence → `deny` (or `ask`)                 | `fibbrain_advise(arguments)` |
| `tools/result`           | Records every tool outcome (command, files, error) into the session episode                | `fibbrain_observe`          |
| `agent/turn-stopping`    | Distils the episode into pending (or active) memories; idempotent                          | `fibbrain_review_session`   |
| `ctx.skills` provider    | Publishes verified procedures as a skill catalog; refreshed after each review              | `fibbrain_render("*")`      |

Everything except `advise` fails **open**: if the Python process is down, the
agent keeps working without memory and the plugin logs a warning. `advise`
failing open means "allow", so its reject is the only way this plugin can
change what the agent does.

## Identity

dsh has no owner / project concept, so the plugin maps:

- `owner` ← `config.owner`, else `$USER`, else `dsh`
- `workspace_id` ← basename of the git root that contains the session cwd
- `project_id` ← `config.projectId`, else `workspace_id`
- `session_id` ← the dsh session id

That is what keeps one project's memories out of another's recall.

## Enable

```sh
cd /path/to/fibmind
uv sync --locked --extra dev            # Python side
npx @deepseek-ai/dsh web --patch "$PWD/integrations/dsh/fibbrain.cordis.yml"
```

Edit the absolute paths in `fibbrain.cordis.yml` if the checkout is not at
`/Users/abbila/PycharmCompany/fibmind`. The plugin is referenced by absolute
path because dsh's loader resolves relative paths against the profile
directory, not the overlay file.

The plugin depends on `@modelcontextprotocol/sdk` (already installed with dsh)
and imports `@deepseek-ai/dsh-llm` / `@deepseek-ai/schemastery` from the dsh
installation. When run from a checkout where they resolve differently, install
them next to the plugin or symlink `node_modules`.

## Verify inside dsh

1. Start a session. The first request should contain a `<fibbrain-hot>` user-role
   context message (Trajectory view). Empty on a fresh store — that is expected.
2. Say `Remember that the validation drink is lapsang-<suffix>` and let the model call
   `mcp__fibbrain__fibbrain_remember` (mcp-client row) — or just work; the
   `tools/result` hook is observing.
3. Open a **new** session, ask anything. Its first request now contains a
   `<fibbrain-recall>` message with the memory, added by the plugin, not by the model.
4. `fibbrain_reflect(verdict="refuted", ...)` a memory that names a tool, then ask the
   model to use that tool with matching arguments: the call is denied with the reason.
5. After a turn ends, `fibbrain_pending_reviews` lists what the review extracted.

## Test without dsh

```sh
cd integrations/dsh
node --test test/plugin.test.js
```

The tests use a Cordis-shaped fake context and a scripted client, so they run
anywhere Node 22 runs. The end-to-end path (`src/client.js` against the real
Python server) is exercised by the repository's Python test suite through the
MCP surface it calls.

## Configuration

| key                    | default                                            | meaning                                                |
| ---------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| `command`, `args`, `cwd`, `env` | `python -m fibmind.mcp_server --store .fibmind/memory.db` | how to start the FibBrain server                |
| `owner`, `projectId`, `workspaceId` | derived                                 | identity overrides                                     |
| `hotOnStart`, `hotBudgetTokens`, `hotTopK` | `true`, 400, 3                  | session-start preamble                                  |
| `recall`, `recallTopK`, `recallBudgetTokens` | `true`, 5, 500                | per-prompt recall                                       |
| `advise`               | `deny`                                             | `deny` / `ask` / `off`                                 |
| `observe`              | `true`                                             | record tool results                                    |
| `reviewMode`           | `approve`                                          | `approve` / `auto` / `off`                             |
| `skills`, `skillsMinMaturity` | `true`, `verified`                          | publish procedures as skills                            |

## Boundaries

- dsh's `ctx.goals` is the current-session goal; FibBrain's `plan` goals are
  cross-session history. The plugin does not sync them.
- Pre-tool **argument rewrite** is not a dsh extension point; the plugin can deny
  or ask, never edit.
- The plugin does not run procedures. `render` produces skill text; the harness's
  skill loader shows it to the model.
