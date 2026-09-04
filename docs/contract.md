# FibBrain Contract

This document names what is stable and what is an implementation detail. The
Python package in `src/fibmind/` is *one* implementation. Anything that keeps
the promises below — in any language — is a conforming FibBrain.

The contract has three parts: the **event log** (how memory is persisted), the
**Brain protocol** (the seven `fibbrain_*` actions), and the **conformance
fixture** that checks both.

## 1. Event log

**Version:** `log_version = 1` (`fibmind.models.LOG_VERSION`).

The log is the source of truth. Nodes, edges, and trees are a cache that
`rebuild_from_log(events)` must reproduce exactly. Two things are deliberately
*not* replayed: `familiarity` and `access_count` describe how a store was used,
not what it knows, and reset to zero on rebuild.

Every event is:

```json
{
  "id": "event_<hex>",
  "seq": 12,
  "op": "create_node",
  "payload": { ... },
  "created_at": "2026-09-03T06:28:37.119077+00:00",
  "log_version": 1
}
```

`seq` is a dense, monotonically increasing integer per store. Replay order is
by `seq`. `op` is one of:

| op | payload | effect on replay |
| --- | --- | --- |
| `create_tree` | `tree_id`, `category`, `title`, `root` (node dict) or `existing_root_node_id` | create tree and its root node |
| `create_node` | `node` (full node dict), `tree_id` | insert node, attach to tree |
| `link` | `edge` (full edge dict) | insert edge |
| `fold` | `into_node_id`, `source_node_ids` | set `folded_into` on each source |
| `revise` | `node_id`, `before`, `after` (title / content / tags / status) | replace text, reset confidence to 0 |
| `observe` | `node_id`, `verdict`, `source`, `confidence`, optional `status` / `status_reason` / `note` | set confidence; a `refuted` verdict sets status `refuted` |
| `set_status` | `node_id`, `status`, `reason` | set status |
| `promote` | `node_id`, `node_type` | change node type |
| `forget` | `node_id`, `removed_edges` | delete node and edges; **earlier payloads for that node are redacted in place** |

Redaction on `forget` is the one time an existing log entry is mutated. A replay
that resurrects forgotten content is non-conforming.

**Stores.** A JSON store carries `log_version` at the top level. A SQLite store
carries it in `meta.log_version` and `schema_version` (currently 4) for its
own table layout. A missing `log_version` is read as version 1. A reader must
refuse a log whose version is greater than what it implements.

**Bumping the version.** Bump only when an existing payload changes meaning.
Adding an optional field to a node or payload does not bump; readers must ignore
unknown fields and default missing ones.

### Node fields that carry meaning

| field | contract |
| --- | --- |
| `scope` | `session` / `personal` / `knowledge`. `knowledge` is only written by `promote_to_knowledge`. |
| `status` | `active` / `pending` / `stale` / `refuted` / `forgotten`. Normal recall sees only `active`. `pending` is a review-written memory awaiting approval. |
| `owner`, `workspace_id`, `project_id`, `session_id`, `task_id` | Identity. Visibility is exact-match per [`PLAN-identity.md`](../PLAN-identity.md); `task_id` is recorded but never filters. |
| `confidence`, `confidence_source` | Moves only through `observe` events with a named source. The only trust signal ranking may use. |
| `familiarity`, `access_count` | Usage counters. Never enter ranking. Not replayed. |
| `folded_into` | Set by `fold`. Folded nodes are excluded from normal recall but keep their text. |

## 2. Brain protocol

The MCP server registers exactly the tools listed under `tools` in
[`tests/fixtures/demo-brain.expected.json`](../tests/fixtures/demo-brain.expected.json).
Tool names and the meaning of their parameters are stable. New optional
parameters may be added; existing ones are not renamed or removed.

Every `fibbrain_*` tool accepts the identity quintet `owner`, `workspace_id`,
`project_id`, `session_id`, `task_id` as optional strings. Omitting a field means
"unlabelled", which sees only unlabelled memories — never "all".

| tool | question | persists |
| --- | --- | --- |
| `fibbrain_plan(objective \| goal_id, identity)` | What am I doing, in what order? | yes: a `goal` node tagged `fibbrain-goal`, JSON document with `kind: fibbrain_goal` |
| `fibbrain_recall(goal, identity, top_k, depth, max_chars)` | What do I already know? | no |
| `fibbrain_coordinate(objective \| goal_id, identity)` | Which capabilities next, and are they safe? | may mark the goal `blocked` |
| `fibbrain_remember(category, title, content, tags, scope, identity)` | Is this worth keeping? | only on `verdict: write` |
| `fibbrain_observe(kind, summary, payload, identity)` | What just happened? | with a `session_id`: yes, as `scope=session` scratch in category `episode`; recall never returns it as a hit |
| `fibbrain_review_session(session_id, mode, close, identity)` | What from this session is worth keeping? | `approve`: as `pending`; `auto`: as `active`; `candidates`: nothing |
| `fibbrain_pending_reviews(identity)` / `fibbrain_approve_memory` / `fibbrain_reject_memory` | Which review memories wait, and do they get in? | status changes only |
| `fibbrain_advise(action, kind, identity)` | Should this action run? | no |
| `fibbrain_reflect(verdict, source, node_id \| goal, identity)` | Did a memory hold up? | yes: an `observe` event |
| `fibbrain_complete_goal(goal_id \| objective, identity)` | Done. | yes: goal status `complete` |

Return shapes are plain JSON objects. Fields present today are stable; new
fields may appear.

**Guarantees a conforming brain must keep:**

- `remember` returns `verdict` ∈ `write` / `skip` and a `reason`. Skipped
  candidates are not persisted. Exact duplicates within the same owner /
  workspace / project are skipped.
- `recall` never returns a node the caller's identity cannot see, including
  through graph expansion.
- `advise` returns `reject` when a `refuted` memory matches the action in the
  caller's identity, otherwise `allow`.
- `plan` with the same objective and identity reuses the existing non-complete
  goal (idempotent).
- Reading a memory back never changes its ranking.
- `review_session` is deterministic and idempotent: the same episode yields
  the same candidates, and a second run writes nothing.
- Episode scratch (`category=episode`) never appears as a `recall` hit, only
  in the pack's `episode` list, and only to its own session.

## 3. Conformance fixture

`data/demo-brain.json` is regenerated by `examples/demo.py`. It is small on
purpose: one identity, three trees (`decision`, `goal`, `episode`), and the
handful of events one plan → coordinate → recall → complete run produces.

`tests/fixtures/demo-brain.expected.json` records what a conforming
implementation must produce from it:

- `shape`: node / edge / tree counts and category names after replay;
- `queries`: fixed search queries with identity and the exact ordered titles
  they must return, evaluated the way `recall` evaluates them (active,
  visible, `episode` excluded) — including three that must return nothing
  (wrong project, no identity, unrelated terms);
- `tools`: the exact MCP tool surface.

`tests/test_contract.py` runs the check for the Python implementation. A port
should load the same two files and assert the same things. If the fixture
changes, regenerate the expected file deliberately and explain the change in
the commit; a silent diff there is a contract break.

## Out of contract

Deliberately *not* promised, so they can change without notice:

- the ranking formula and its weights (`ranking.py`);
- Fibonacci layer capacities and the fold / summarize strategy;
- the deterministic plan template and capability hints (`planning.py`);
- the admission heuristics (`admission.py`);
- SQLite table layout beyond `meta.log_version`.

These are exactly the parts [`PLAN-evolve.md`](../PLAN-evolve.md) expects to
replace or tune.
