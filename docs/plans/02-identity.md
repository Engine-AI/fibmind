# FibMind Identity Model Plan

The previous round, the governance overhaul, is in [`PLAN.md`](01-governance.md) and has shipped. The vision for the whole phase is P1 in [`ROADMAP.md`](../../ROADMAP.md).

This document is the **first slice** of P1: it only adds working-context identity and makes every recall entry point go through one visibility rule. Only once this is done can `context(goal)` become the data foundation for `recall(goal, state)` in the later Brain Protocol.

This round does not split `graph.py`, does not add embeddings / FTS / token budgets, and does not write the DeepSeek-Harness plugin.

## Background

Today a node carries only `scope` + `owner`. Scratch written in one session, a decision from another project, a preference from another workspace — as long as the owner matches, they all land in the same recall.

This blocks two things:

- DeepSeek-Harness Sessions cannot be hooked in safely;
- graph expansion is already a side door: the MCP `search_from` tool does not accept owner, and `context.py` only filters again after expanding.

`ROADMAP.md` P1 also lists `branch`, `commit_sha`, `memory_kind`, and a full directory restructure. Those are not in this slice. This slice ships only the four identity fields and unified visibility.

## Principles

- The four fields are **origin labels**, not four equally strict isolation walls. Personal memories must be recallable across sessions, otherwise long-term memory is pointless.
- "Exact match" keeps the existing owner semantics: `None` sees only `None`. A caller that does not pass `workspace_id` cannot see memories labelled with a workspace. This is deliberate: it keeps the default from leaking.
- Visibility lives only in `FibMind.is_visible()`. Search, graph expansion, auto-linking, and compression must not carry their own copy of the rule.
- The public API only gains optional parameters. When they are omitted, behaviour falls back to today's owner / scope isolation.
- The log is still the source of truth. The new fields travel in `node.to_dict()` inside `CREATE_NODE`; no new `EventOp`.

## Scope

| In | Out |
| --- | --- |
| `workspace_id` / `project_id` / `session_id` / `task_id` | `branch` / `commit_sha` / `source_uri` / `content_hash` / `valid_from` / `memory_kind` |
| Writes, search, graph expansion, compression, and auto-linking share one visibility rule | Splitting into `domain/` `retrieval/` in one go |
| SQLite schema v3 → v4; old databases get `NULL` in the new fields | Changes to Fibonacci layers, new relations |
| Backward-compatible MCP parameters (new fields optional) | The five Brain Protocol functions |
| Unit tests + evals for cross-project / cross-session scratch | Expanding to real project trajectories, model-graded scoring |

## Identity semantics

```text
workspace_id   Workspace / repository root. When set, visible only within the same workspace
project_id     Project. When set, visible only within the same project
session_id     The session in which this memory was written
task_id        The task in which this memory was written (recorded only; does not isolate by default)
```

Combined with the existing `scope`:

```text
scope=session     scratch. Must carry a session_id.
                  Visible only when caller.session_id == node.session_id.
                  Not visible if either side is missing.

scope=personal    Long-term personal observation.
                  owner exact match (including both being None)
                  + workspace exact match
                  + project exact match
                  session_id / task_id record origin only and do not block cross-session recall

scope=knowledge   Still ignores owner
                  but must pass the workspace / project exact match
                  Old knowledge with no workspace stays visible to everyone
```

Knowledge promotion: the new node from `promote_to_knowledge` is written with the workspace / project the caller passes; if none is passed, it takes the workspace / project shared by the supporting nodes. If they do not match, promotion is rejected: an observation from project A must never be promoted into knowledge for project B.

## Visibility

The rule lives only in `FibMind.is_visible()`:

```text
status ∈ allowed set
not folded (unless folded nodes are explicitly requested)
scope filter (if the caller gave one)
scope ≠ knowledge → owner exact match
workspace_id exact match
project_id exact match
scope = session → session_id exact match (both sides must have a value; not visible if either is missing)
```

These paths must all go through that one function:

- `search`
- `search_from` (at every hop: an invisible neighbour is neither returned nor traversed further)
- `build_context`
- `_auto_link_similar`
- `compress_overflow` (nodes from different workspaces / projects must not be folded into one summary; if they differ, skip compression)

`search_from` gains the same `owner` / `scopes` / `statuses` / identity parameters as `search`.

## Data and storage

`MemoryNode` gains four `str | None` fields, defaulting to `None`. `to_dict` / `from_dict` treat a missing field as `None`, so replaying old events does not break.

SQLite: `SCHEMA_VERSION = 4`. `_upgrade` adds four `TEXT` columns to `nodes`, plus indexes:

```text
idx_nodes_workspace (workspace_id, project_id)
idx_nodes_session   (session_id)
```

The JSON store follows `to_dict`; no new format version is needed.

A compressed node inherits its source nodes' identity; if the source nodes' identities differ, compression is skipped.

## API

`append` / `search` / `search_from` / `context` / `promote_to_knowledge` each gain four optional parameters. `MemoryService` passes them through unchanged. Node summaries include the four fields.

MCP: `fibmind_append`, `fibmind_search`, `fibmind_search_from`, `fibmind_context`, and `fibmind_promote_knowledge` gain optional parameters with the same names. No existing tool is removed or renamed.

`AGENTS.md` / `CLAUDE.md` gain one sentence: both recall and writes must pass the current `workspace_id` / `project_id` / `session_id`.

The eval fixtures and runner gain the four optional fields. Existing datasets leave them unset, so their results stay as they are.

## Acceptance

Add `tests/test_identity.py`, covering at least:

1. Same owner, different `project_id`: personal memories are not visible to each other
2. Same owner, different `workspace_id`: personal memories are not visible to each other
3. Same owner + same project, different `session_id`: personal memories **are visible**
4. `scope=session` scratch is not visible from another session
5. Without `workspace_id`, memories labelled with a workspace are not visible
6. knowledge ignores owner but is still isolated by workspace / project
7. Graph expansion cannot bypass this: when a project-A node links to a project-B node, expanding in A's context does not reach B
8. `_auto_link_similar` does not link across projects
9. The four fields survive `rebuild_from_log`
10. A schema v3 file is upgraded to v4 on open; old nodes have `None` in the new fields, and old search behaviour is unchanged

Evals: add `evals/datasets/cross_project.json` (or extend `cross_session.json`):

- Each of two projects writes a decision with a similar title
- When the query carries `project_id=A`, B's id must be in `forbidden_ids`
- A `scope=session` draft must not be hit when queried with a different `session_id`

Verification:

```bash
.venv/bin/pytest -q
.venv/bin/python -m evals.runner
```

The existing 11 cases must not regress; isolation in the new cases must hold.

## Commit breakdown

1. Model + schema v4 + `from_dict` compatibility
2. `is_visible` + every recall entry point + auto-linking / compression
3. Service / MCP / agent instructions
4. Unit tests + eval dataset

Each step only adds optional parameters and can be rolled back on its own.

## After this

```text
This slice: identity model
    ↓
Brain v0 (see 03-brain.md): recall / remember / observe / advise / reflect
    ↓
DSH plugin: pre-step calls recall, turn/end calls remember / reflect
    ↓
P2 hybrid retrieval (new engine, same filtering rules)
```

The module split in `ROADMAP.md` P1 moves only after `is_visible` has stabilized.
