# FibBrain

FibBrain is a cognitive runtime for agents. **FibMind** is the memory engine
underneath it.

```text
Agent  =  FibBrain (plan / recall / decide / coordinate / remember)
       +  Harness  (run / execute / observe)
       +  Plugins  (tools / MCP / models)
```

Brain v1 implements five capabilities on top of the existing store:

| Action | Question |
| --- | --- |
| `plan` | What am I doing, and in what order? |
| `recall` | What do I already know about this goal? |
| `coordinate` | Which capabilities should the body involve next? |
| `remember` | Is this worth writing into long-term memory? |
| `observe` | What just happened in this episode? |
| `advise` | Should this action run, given past evidence? |
| `reflect` | Did a stored memory hold up? |

It does not own an agent loop or a plugin manager. DeepSeek-Harness (or any
other harness) stays the body. Planning is an explicit persisted goal, not a
rewrite of the harness Goal service. Coordination names capabilities and
advises them; it does not execute plugins.

The memory engine models long-term knowledge as:

- multiple category trees, forming a memory forest
- an append-only event log as the source of truth, with nodes and edges as a
  rebuildable cache
- separate signals for how *familiar* a memory is and how much external evidence
  says it is *correct*
- scopes that keep one owner's personal memories out of everyone else's recall,
  with a statistical bar for promoting an observation into shared knowledge
- Fibonacci-sized layers for raw, compressed, summary, and long-term memory
- graph edges between related nodes, plus local graph search

The graph, retrieval, context, admission, and storage modules use only the
Python standard library. Recall is an inverted index with BM25 plus title,
confidence, and recency signals; set `FIBMIND_EMBEDDING=hashing` (offline) or
`FIBMIND_EMBEDDING=openai` with `FIBMIND_EMBEDDING_URL` / `_MODEL` / `_API_KEY`
to fuse a vector candidate list in with reciprocal rank fusion. A provider
failure degrades to lexical recall instead of failing the call. The agent-facing server uses the MCP Python SDK 2.x. Prefer
the `fibbrain_*` tools; `fibmind_*` remains the direct store API.

> [!WARNING]
> FibMind is currently a local, single-user prototype. Multi-owner isolation and
> complete event-log replay invariants are still being hardened. Do not use it as
> a multi-tenant production service or store secrets in it yet.

## Quick Start

```bash
cd /fibmind
uv sync --locked --extra dev
.venv/bin/python examples/demo.py
.venv/bin/pytest -q
```

The demo writes a sample JSON store to `data/demo-memory.json`. SQLite is the
recommended backend for a persistent agent memory service.

## Retrieval Evaluation

The P0 evaluation suite compares four paths over the same fixed datasets:

- no persisted memory;
- static `AGENTS.md`-style context;
- a bounded Hermes-style hot-memory snapshot;
- FibMind's current keyword search and graph context expansion.

Run the aggregate report:

```bash
.venv/bin/python -m evals.runner
```

Write the full per-case report as JSON:

```bash
.venv/bin/python -m evals.runner \
  --format table \
  --output .fibmind/eval-report.json
```

Use `--baseline fibmind_current` to run one baseline, and repeat `--baseline`
to select several. The bundled datasets cover project-history recall, stale and
conflicting memories, negative queries, owner isolation, and cross-session
recall. Reports include Recall@K, MRR, context answerability, irrelevant/stale
pollution, forbidden-memory hits, context size, and P50/P95 latency.

P0 reports `estimated_tokens`: a deterministic, model-agnostic size estimate
for relative comparisons. It is deliberately not described as provider billing
usage. P3 will replace it with the selected model's tokenizer. Dataset format,
metric definitions, and the next implementation phases are documented in
[`ROADMAP.md`](ROADMAP.md).

## Design Constraints

Three properties the implementation is built around. They are worth stating
because each rules out a shape that looks reasonable:

**The log is the source of truth.** Every semantic change appends a
`MemoryEvent`; `FibMind.rebuild_from_log` reconstructs the whole forest from it.
Node and edge tables are a cache you can throw away. This is what makes a store
replayable when a better summarizer comes along, and mergeable across devices —
text diffs and merges, derived state does not. Recall counters (`familiarity`,
`access_count`) are the exception: they describe usage rather than knowledge and
are not replayed.

**Recall is not evidence.** `familiarity` rises whenever a memory is read back;
it is deliberately absent from relevance scoring. Only `confidence` — which moves
solely through `record_outcome`, and only with a named source — affects ranking.
Scoring recall frequency instead closes a loop (read → ranked higher → read
more) that converges on whatever is familiar rather than whatever is true.

**Nothing gets in unscanned.** Admission rejects candidates carrying API keys,
private keys, connection strings, prompt-injection phrasing, or invisible
Unicode, and returns the finding kind. A memory is replayed into every later
prompt, so a leak or an injected instruction would be permanent.

**Knowledge inherits its evidence.** A shared claim is linked to the
observations it was promoted from. Refuting one supporter scales the claim's
confidence by the share of supporters still standing; refuting all of them
retires the claim from recall. The claim's own confirmed count is untouched:
weakening by inheritance is not a judgement about the claim.

**The brain tunes itself only through a gate.** `fibbrain_tune` derives a
one-step, bounded proposal for the ranking weights from which memories were
later confirmed or refuted, runs the fixed evaluation suite before and after,
and adopts the change only if Recall@K, stale pollution, forbidden hits, and
no-relevant accuracy did not get worse. Familiarity is not a tunable weight.

**Deletion has to reach the log.** `forget` removes the node and its edges *and*
tombstones the content in earlier log entries. Dropping only the row would let
the next replay resurrect it.

## Core Concepts

```text
Node        A memory unit. It may be raw data, compressed data, summary, concept, root, or archive.
Edge        A typed relationship between two nodes.
Tree        A category or topic tree whose root is also a normal node.
Forest      Many trees connected by cross-tree graph edges.
Layer       A capacity band controlled by Fibonacci-sized limits.
Event       One entry in the append-only log. The forest is a function of these.
Scope       session / personal / knowledge — who a memory belongs to.
Status      active / stale / refuted — whether normal recall may return it.
```

### Scopes

```text
session     Scratch state, expected to be forgotten.
personal    An observation about one owner. Never shared: sharing both leaks
            privacy and pollutes other owners' preferences.
knowledge   A claim about the world, distilled from several personal
            observations. The only shareable scope.
```

Keeping these in one bucket is what makes a shared layer unshareable — it fills
with one person's preferences — and a personal layer impersonal. `append` writes
`personal` or `session`; `knowledge` is only reachable through
`promote_to_knowledge`, which requires at least three distinct supporting
observations and links each one as provenance so a claim that later proves wrong
can be traced back.

Normal search and context return only `active` memories. `mark_stale` retires a
superseded memory while preserving its provenance; a `refuted` outcome also
removes a memory from normal recall. Maintenance queries can explicitly request
inactive statuses. `revise` corrects the content and makes the memory active and
unproven again. Personal/session memories require an exact owner match; omitting
an owner no longer exposes memories belonging to named owners.

## Default Layers

```text
raw         21 memory weight
compressed 34 memory weight
summary    55 memory weight
long_term  89 memory weight
```

Layer capacity is measured as memory weight. A raw node has weight `1`; a
compressed or summary node carries the total weight of the source nodes it
covers.

Two honest caveats about this part:

- Capacity answers *how many* memories fit, not *which* are worth keeping. The
  Fibonacci spacing gives growing bands with a stated rule for the gaps; any
  similarly growing sequence would behave the same.
- `_summarize` concatenates truncated excerpts. Its output sits at the same level
  of abstraction as its inputs, merely shorter — it is a placeholder, not
  distillation. Real promotion of specifics into general claims goes through
  `promote_to_knowledge`. Because folding marks sources with a `folded_into`
  pointer instead of overwriting them, the originals survive for a better
  summarizer to redo the work.

## Main Operations

```python
memory = FibMind()
tree_id = memory.create_tree("code", title="Code Tree")
node_id = memory.append("code", title="Login bug", content="401 after token expiry")

memory.link_nodes(node_id, other_node_id, RelationType.RELATED_TO)
memory.promote_node(node_id)
results = memory.search_from(node_id, depth=2)
activated = memory.search_from(node_id, depth=2, reinforce=True)

# Attach outside evidence — the only thing that changes how a memory ranks.
memory.record_outcome(node_id, Verdict.CONFIRMED, source="pytest tests/test_auth.py")

# Correct a memory (confidence resets), retire an old one, or delete it for good.
memory.revise(node_id, content="401 only after refresh-token expiry")
memory.mark_stale(old_node_id, reason="superseded by the refresh-token decision")
memory.forget(stale_node_id, reason="user asked for removal")

# Distil repeated observations into one shareable claim (needs >= 3 supporters).
claim_id = memory.promote_to_knowledge(
    title="Uploads need an explicit retry budget",
    content="Large uploads die at the first transient timeout without a retry budget.",
    supporting_node_ids=[first, second, third],
)

# Nodes and edges are a cache; this rebuilds them from the log alone.
replayed = FibMind.rebuild_from_log(memory.events)
```

`search_from()` is read-only by default. Pass `reinforce=True` when a query
should behave like memory activation — note this raises `familiarity` only, and
familiarity does not affect ranking.

`search()` is the query-based counterpart: it ranks the whole forest by keyword
relevance to a text query and never mutates nodes. Pass `owner=` to exclude other
owners' personal memories, or `scopes=` to restrict to e.g. shared knowledge.

## MCP Server (long-term memory for agents)

FibMind can run as an [MCP](https://modelcontextprotocol.io/) server so tools like
Codex and Claude Code can use it as external long-term memory. The project pins
the MCP Python SDK to `>=2,<3`; MCP 1.x used a different server import path.

```bash
# create or synchronize the virtual environment
uv sync --locked --extra dev

# optional smoke test: run the stdio server directly
.venv/bin/python -m fibmind.mcp_server --store .fibmind/memory.db
```

The direct command waits silently for MCP messages on stdin. That is expected:
it is not an HTTP server and does not print a URL. Normally the MCP host starts
and manages this process for you.

### Tools

Recall and record:

| Tool | Purpose |
| --- | --- |
| `fibbrain_plan` | Create or refresh a persisted goal and its deterministic plan. |
| `fibbrain_coordinate` | Recommend the next capabilities and advise each one. |
| `fibbrain_complete_goal` | Mark a persisted goal complete; it stays recallable. |
| `fibbrain_recall` | Recall a token-budgeted context pack: frozen hot memory first, then direct hits, then related memories, with per-section `budget` usage. |
| `fibbrain_hot` | The session's frozen L0 hot-memory snapshot (stable preferences, conventions, evidenced decisions); `refresh` recomputes it. |
| `fibbrain_tune` | Propose a bounded ranking-weight change from confirmed-vs-refuted evidence; adopt it only if the fixed evaluation suite shows no regression. Every attempt is logged. |
| `fibbrain_revalidation_candidates` | Knowledge and procedures with no confirmation for N days. Output only. |
| `fibbrain_remember` | Admit a candidate, then write it if it is worth keeping. |
| `fibbrain_observe` | Note an episode event; persisted as session scratch when a `session_id` is given. |
| `fibbrain_review_session` | Distil one session's episode into long-term candidates (`candidates` / `approve` / `auto`); idempotent. |
| `fibbrain_pending_reviews` | List review memories waiting for approval. |
| `fibbrain_remember_procedure` | Store a repeatable way of doing a class of task (trigger, steps, tools, verify, inputs). |
| `fibbrain_render` | Render the procedures that fit a goal as `skill` (SKILL.md) or `tool` (JSON-Schema) definitions, filtered by maturity. |
| `fibbrain_approve_memory` / `fibbrain_reject_memory` | Let a pending review memory in, or retire it as stale. |
| `fibbrain_advise` | Allow or reject an action given stored evidence. |
| `fibbrain_reflect` | Record that a memory turned out right or wrong. |
| `fibmind_context` | Direct store context pack (prefer `fibbrain_recall`). |
| `fibmind_append` | Direct write that bypasses admit (prefer `fibbrain_remember`). |
| `fibmind_search` | Rank memories by keyword relevance to a query (read-only). |
| `fibmind_search_from` | Expand the association tree rooted at a node. |
| `fibmind_link` | Create a typed relationship between two memories. |
| `fibmind_explain_recall` | Per-result lexical / vector / fusion / confidence / recency signals, and why other candidates were excluded. |
| `fibmind_status` | Store size, lifecycle counts, lexical index size, embedding health. |

Judgement and upkeep — the half that keeps the store from degrading:

| Tool | Purpose |
| --- | --- |
| `fibmind_record_outcome` | Record that a memory turned out right or wrong, with a source. |
| `fibmind_revise` | Correct a memory and reactivate it with confidence reset. |
| `fibmind_mark_stale` | Retire superseded content while keeping provenance. |
| `fibmind_forget` | Delete a memory permanently, including from log history. |
| `fibmind_promote_knowledge` | Turn several observations into one shared claim. |

`fibbrain_plan` then `fibbrain_recall` before starting a task, and
`fibbrain_remember` after finishing one, are the calls you use most.
`fibbrain_coordinate` names the next capabilities; `fibbrain_advise` can still
be called directly for a specific action. `fibbrain_remember` will skip a
duplicate or a dump. A store that only ever writes still degrades unless
`fibbrain_reflect` records whether those memories held up.

### Host check (this machine)

Claude Code and Codex can both drive FibBrain over stdio MCP. A scripted
feasibility pass already wrote isolated memories into `.fibmind/memory.db`:

| Host | Result |
| --- | --- |
| Claude Code (`claude -p`) | `plan` → `remember` → `recall` → `coordinate` → `complete` |
| Codex (`codex exec`) | same sequence, after per-server MCP auto-approve |

Paste [`examples/host_mcp_prompt.md`](examples/host_mcp_prompt.md) into a **new**
interactive session to repeat it by hand. Restart or open a new task after
changing MCP config; an already-open session will not see the server.

### DeepSeek-Harness

Two ways, both in this repository:

- **Zero-code**: [`examples/dsh/`](examples/dsh/README.md) mounts the MCP server
  through `@deepseek-ai/dsh-mcp-client`; the model calls `mcp__fibbrain__*` tools.
- **Native plugin**: [`integrations/dsh/`](integrations/dsh/README.md) is a
  Cordis plugin that recalls before every prompt, advises before every tool call,
  observes results, reviews the session on turn end, and publishes verified
  procedures as skills — without the model having to remember to call anything.

### Register with Codex

Use absolute paths so Codex can launch the server regardless of its current
working directory:

```bash
codex mcp add fibmind -- \
  /Users/abbila/PycharmCompany/fibmind/.venv/bin/python \
  -m fibmind.mcp_server \
  --store /Users/abbila/PycharmCompany/fibmind/.fibmind/memory.db
```

Verify the saved configuration:

```bash
codex mcp get fibmind
codex mcp list
```

Restart Codex or open a new task after registering the server. MCP configuration
is loaded when a task starts and an existing task may not discover a newly added
server. You do not need to keep a separate terminal running the server; Codex
launches the stdio process itself.

To verify the connection from a new Codex task, ask it to call
`fibbrain_remember` with a small test memory and then retrieve it with
`fibbrain_recall`. This repository's `AGENTS.md` already tells Codex to:

1. call `fibbrain_plan` and `fibbrain_recall` before non-trivial work;
2. call `fibbrain_coordinate` (or `fibbrain_advise`) before acting on a capability
   that past evidence might block;
3. call `fibbrain_remember` after implementation with decisions, changed files,
   tests, and unresolved risks; and
4. call `fibbrain_reflect`, `fibmind_revise`, `fibmind_mark_stale`, or
   `fibmind_forget` when new evidence changes a stored memory.

Remove the registration with `codex mcp remove fibmind`. See the
[official Codex MCP documentation](https://developers.openai.com/codex/mcp/) for
general Codex configuration details.

### Register with Claude Code

```bash
claude mcp add --scope project --transport stdio fibmind -- \
  /Users/abbila/PycharmCompany/fibmind/.venv/bin/python \
  -m fibmind.mcp_server \
  --store /Users/abbila/PycharmCompany/fibmind/.fibmind/memory.db
```

Then tell the agent when to use it via `CLAUDE.md` (Claude Code) or `AGENTS.md`
(Codex) — both are included in this repo.

### Architecture

```text
Codex / Claude Code
        │ fibbrain_* / fibmind_* MCP tools
        ▼
MCP Server                (src/fibmind/mcp_server.py)
        ▼
FibBrain                  (plan / recall / coordinate / remember / observe / advise / reflect)
        ▼
MemoryService             (src/fibmind/service.py — load/lock/persist)
        ▼
FibMind Core              (graph, ranking, context, admission)
        ▼
event log  ──rebuild──▶  nodes / edges / trees
(source of truth)        (cache, JsonStore or SqliteStore)
```

### Storage

`--store` picks the backend by suffix: `.json` for `JsonStore` (atomic write +
fsync, good for demos) or `.db` / `.sqlite` for `SqliteStore` (WAL, `BEGIN
IMMEDIATE` transactions). Both persist the event log alongside the materialized
state.

SQLite schema v4 adds `workspace_id`, `project_id`, `session_id`, and `task_id`
so recall can isolate one working context from another. Schema v3 added
lifecycle `status` and `status_reason`. Schema v2 introduced `scope`, `owner`,
`familiarity`, `confidence`, `confidence_source`, `folded_into`, and the
`event_log` table. Older databases upgrade in place on open: the old
`importance` column becomes `familiarity` (confidence starts at zero, since
nothing external ever backed those numbers), and nodes v1 folded by rewriting
their layer recover a `folded_into` pointer.
`fibmind-migrate` converts a JSON store to SQLite.

```bash
.venv/bin/python -m fibmind.migrate \
  --from-json data/demo-memory.json \
  --to-sqlite data/demo-memory.db
```

The migration refuses to overwrite an existing SQLite target.
