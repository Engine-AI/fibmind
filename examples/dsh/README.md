# FibBrain on DeepSeek-Harness (zero-code)

This folder mounts FibBrain into [DeepSeek-Harness](https://github.com/deepseek-ai/deepseek-harness)
(dsh) through its generic MCP client. No TypeScript, no dsh source changes. It
is the same shape dsh uses for its own third-party memory examples
(`docs/user/guide/mcp-memory.md`).

What you get: every `fibbrain_*` / `fibmind_*` tool callable by the model as
`mcp__fibbrain__<tool>`. What you do not get yet: automatic recall on every
step, tool-call interception, or skills rendered from memory. Those need the
native Cordis plugin (slice S6 in [`PLAN-evolve.md`](../../PLAN-evolve.md)).

## Prerequisites

- Node 22+ (`npx @deepseek-ai/dsh` is the launcher; nothing to install globally)
- This repo synced: `uv sync --locked --extra dev`
- A DeepSeek API key configured in dsh

## Enable for one run

Edit the three absolute paths in [`fibbrain.cordis.yml`](fibbrain.cordis.yml)
if your checkout is not at `/Users/abbila/PycharmCompany/fibmind`, then:

```sh
npx @deepseek-ai/dsh web --patch "$PWD/examples/dsh/fibbrain.cordis.yml"
```

Discovery is asynchronous. Wait until the tool list shows
`mcp__fibbrain__fibbrain_recall` before the first prompt.

## Keep it across runs

Merge the single `insert` entry into your user patch layer:

- one profile: `$DSH_HOME/profiles/<name>/cordis.patch.yml`
- every profile: `$DSH_HOME/cordis.patch.yml`

Do not copy the file over an existing one; those files may hold unrelated
patches.

## Tell the model when to use it

Paste [`instructions.md`](instructions.md) into the workspace `AGENTS.md`
that dsh loads, or into your model instructions.

## Verify (three steps, two sessions)

Follow dsh's own memory validation so results are comparable with other
memory servers:

1. Session A: `Remember that my validation drink is lapsang-<unique suffix>.`
   Confirm the model called `mcp__fibbrain__fibbrain_remember` and the result
   has `"verdict": "write"`.
2. Session B (same Host, new session): `What is my validation drink? Check memory.`
   Confirm it called `mcp__fibbrain__fibbrain_recall` and answered with the value.
3. Still in B: `Use that preference to suggest one drink for the meeting.`

Then look at the store directly:

```sh
sqlite3 .fibmind/memory.db "select title, owner, workspace_id, session_id from nodes where node_type='raw' order by created_at desc limit 5;"
```

## Turn a real session into an evaluation dataset

dsh keeps one append-only JSONL log per session. FibBrain ships an importer
that turns the `fibbrain_*` calls in that log into a P0 dataset candidate:

```sh
.venv/bin/python -m evals.import_dsh_session \
  "<dsh-session-root>/--<cwd>--/<session-id>/session.jsonl.zstd" \
  --output evals/candidates/real_dsh.json
```

The default dsh log is Zstandard-compressed; the importer shells out to the
`zstd` CLI (Homebrew: `brew install zstd`). Set `compression: 'none'` on
`@deepseek-ai/dsh-session-persistence-jsonl` if you prefer plain lines.

The output is a **candidate**: memories come from `fibbrain_remember` calls
that were written, cases from `fibbrain_recall` calls, and `relevant_ids` are
auto-labelled from what recall actually returned. Review and correct the labels
before moving the file into `evals/datasets/`, otherwise the evaluation only
measures agreement with itself.
