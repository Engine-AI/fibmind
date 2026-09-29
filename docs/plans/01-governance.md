# FibMind Governance Overhaul Plan

The next slice (identity model) is in [`02-identity.md`](02-identity.md). This document records the governance overhaul, which has shipped.


## Diagnosis

The current implementation (1873 lines of core) puts all of its engineering effort on the **generation / supply side** (store, retrieval, context assembly),
and none on judgement and governance. Three structural problems:

1. **Writes have no admission criterion** — `service.append()` only checks that strings are non-empty. The MCP surface has 5 tools: three reads, two writes, zero judgement.
2. **importance is an inverted signal** — `_reinforce_node` weights only by access frequency and enters ranking at weight 0.3,
   forming a self-reinforcing loop: `retrieved → importance↑ → rank↑ → retrieved more easily`. 60 recalls max it out,
   with zero correctness signals. The store gets dirtier with use, and the dirty part rises to the top on its own.
3. **The state snapshot is the source of truth** — upsert + `_delete_missing`, no event stream. Compression rewrites the original node
   in place (`layer` → `folded:raw`): it cannot be replayed, merged, or re-distilled. And there is no delete/revise.

Plus two reproducible bugs and one layer that exists only in name:

- `promote_node` sets the node to ROOT, while `rank_nodes` excludes ROOT by default → **marking a node important makes it unsearchable**.
- `append` only creates a root→node TREE_CHILD edge, so the graph is a star and `context(depth=1)` always expands to nothing.
- Replacing Fibonacci 21/34/55/89 with 20/40/80/160 changes no behaviour; the hard-coded `low_watermarks` values always equal
  `previous_fibonacci()`, which makes it a redundant parameter.

## Principles

- **The log is the only source of truth; materialized state is a disposable build artifact.**
- **familiarity must not enter the ranking weights; only confidence (backed by external evidence) may.**
- **The personal layer and the knowledge layer are separable in the data model, and promotion must clear a statistical bar.**
- This round does not do real distillation (that needs an LLM plus statistical criteria), but implements its **gate** in code: the caller supplies the abstracted text,
  and the code enforces the threshold.

## Steps

### A. Event log (removing irreversibility)

- Add a `MemoryEvent` model + `EventOp` enum:
  `create_node / link / fold / revise / forget / observe / promote`.
- `FibMind` keeps `events: list[MemoryEvent]`; every semantic change records an event.
- `SqliteStore` gains an `event_log` table (append-only); `JsonStore` serializes `events`.
- Add `rebuild_from_log(events)` → rebuilds the whole forest from the event stream.
- **Compression no longer rewrites in place**: source nodes keep their `layer` / `node_type` and gain a `folded_into: str | None`
  pointer; quota accounting excludes folded nodes.
- familiarity / access_count are explicitly defined as a **snapshot cache, not replayable** (reset to zero on rebuild) —
  they are exactly the "disposable build artifact".

### B. Split importance into familiarity + confidence

- `MemoryNode.importance` → `familiarity` (derived from access frequency); add `confidence: float`
  (moves only with external evidence) and `confidence_source: str | None`.
- `from_dict` still reads the legacy `importance` key.
- `ranking`: remove familiarity's scoring weight and use `WEIGHT_CONFIDENCE * confidence` instead.
  → The self-reinforcing loop is broken.
- Add `record_outcome(node_id, verdict, source, note)`, verdict ∈
  `confirmed / refuted`, which writes an `observe` event. This is where judgement plugs in.
- `_retention_key` now puts confidence first; refuted nodes are folded first.

### C. revise + forget

- `revise(node_id, ...)`: changing the content resets confidence to zero (old evidence says nothing about new content) and writes an event.
- `forget(node_id, reason)`: truly deletes the node and its edges, and **redacts that node's historical payloads in the log**
  (content replaced with a tombstone), so a replay cannot resurrect the content. This is the deletability that compliance requires.

### D. Scope layers + knowledge promotion gate

- Add `scope: MemoryScope` (`session / personal / knowledge`, default `personal`)
  and `owner: str | None`. SQLite schema v1 → v2 migration.
- `search` / `search_from` / `context` support scope filtering.
- `promote_to_knowledge(content, supporting_node_ids, ...)`: **requires ≥ N supporting instances**
  before a knowledge node may be created; adds a DERIVED_FROM edge back to each instance (a provenance pointer);
  confidence starts at 0. The caller supplies the abstracted text; the code enforces the threshold.
- A refuted knowledge node can be removed with `forget`; its DERIVED_FROM edges let you trace back what it contaminated.

### E. The two bugs

- `promote_node` no longer sets ROOT and no longer auto-expands; `ROOT` is reserved for the structural anchors that `create_tree` creates.
  `expand_node_to_tree` also keeps the node a CONCEPT. → A promoted node is still searchable.
- `append` automatically links each new node to its top-N `SIMILAR_TO` matches under the existing ranking (same scope, above a threshold),
  so `depth=1` expansion actually returns something. Configurable, on by default.

### F. Judgement tools on the MCP surface

Add `fibmind_revise` / `fibmind_forget` / `fibmind_record_outcome` /
`fibmind_promote_knowledge`; `append` / `search` / `context` gain a scope parameter.
`CLAUDE.md` / `AGENTS.md` now require the agent to **record outcome evidence**, not just append.

### G. Removing Fibonacci redundancy

Delete the redundant hard-coded `low_watermarks` values from `DEFAULT_LAYER_POLICY` (keeping the override mechanism);
the docstring states that capacity is a count heuristic, not a measure of value.

### H. Docs / demo / tests

README states honestly what the layering does and does not do; update the demo and `data/demo-memory.json`;
add tests for each of A–E; fix existing tests that refer to `importance`.

## Acceptance

- `pytest` passes in full.
- After 60 pure recalls, confidence is still 0 and ranking is unchanged (the loop is broken).
- `rebuild_from_log()` matches the materialized state in nodes / edges / trees.
- Replaying the log after `forget` does not resurrect the content.
- After `promote_node`, the node is still hit by `search`.
- `context(depth=1)` yields expand nodes on the default append path.
- `promote_to_knowledge` refuses when there are fewer than N supporting instances.
