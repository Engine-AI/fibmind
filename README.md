# FibBrain

English | [简体中文](README.zh-CN.md)

**FibMind** is a local memory engine that shares one human's background,
project history, and knowledge across Codex, Claude Code, Antigravity (AGY),
and other MCP harnesses. **FibBrain** provides the context, admission, and
evidence interfaces, plus an optional cognitive workflow. The calling harness
controls its conversation, permissions, and execution.

```text
Agent  =  FibBrain (context / recall / plan / remember / reflect)
       +  Harness  (run / execute / observe)
       +  Plugins  (tools / MCP / models)
```

The background interface and optional workflow sit on top of the same store:

| Action | Question |
| --- | --- |
| `context` | Which user background and relevant facts should this harness know? |
| `plan` | What am I doing, and in what order? |
| `recall` | What do I already know about this goal? |
| `coordinate` | Which capabilities should the body involve next? |
| `remember` | Is this worth writing into long-term memory? |
| `observe` | What just happened in this episode? |
| `advise` | Should this action run, given past evidence? |
| `reflect` | Did a stored memory hold up? |
| `verify` | What did a physical check establish about a memory? |
| `reflex` | Can a verified memory or procedure answer this without an LLM? |

It does not own an agent loop or a plugin manager. DeepSeek-Harness (or any
other harness) stays the body. Planning is an explicit persisted goal, not a
rewrite of the harness Goal service. Coordination names capabilities and
advises them; it does not execute plugins. The full tool profile also exposes
explicit command execution through `fibbrain_verify` and opt-in
`fibbrain_reflex(execute=true)`.

The memory engine models long-term knowledge as:

- multiple category trees, forming a memory forest
- an event log as the source of truth, with nodes and edges as a rebuildable
  cache; forgetting redacts earlier event content
- separate signals for how *familiar* a memory is and how much external evidence
  says it is *correct*
- owner and working-context visibility, with explicit user-wide preferences
  and a minimum-support gate for promoting observations into shared knowledge
- Fibonacci-sized layers for raw, compressed, summary, and long-term memory
- graph edges between related nodes, plus local graph search

The graph, retrieval, context, admission, and storage modules use only the
Python standard library. Recall is an inverted index with BM25 plus title,
confidence, and recency signals. SQLite-backed recall also uses a persistent
FTS5 candidate index, including CJK tokenization and file/symbol anchors.
Set `FIBMIND_EMBEDDING=hashing` (offline feature hashing) or
`FIBMIND_EMBEDDING=openai` with `FIBMIND_EMBEDDING_URL` / `_MODEL` / `_API_KEY`
to fuse a vector candidate list in with reciprocal rank fusion (RRF). The
SQLite vector cache is keyed by provider and content hash. A provider failure
degrades to lexical recall instead of failing the call. The agent-facing
server uses the MCP Python SDK 2.x. Prefer the `fibbrain_*` tools;
`fibmind_*` remains the direct store API.

## Current Status

As of **2026-10-27**, the package version is **0.1.0** (alpha). Implemented
features include:

| Area | Available now |
| --- | --- |
| Cross-application context | Read-only `fibbrain_context`, explicit user/workspace/project levels, shared human identity, configuration generation and native host registration. |
| Retrieval | BM25 inverted index, SQLite FTS5 candidates, optional hashing/OpenAI-compatible embeddings, persistent vector cache, RRF, and recall explanations. |
| Context budgets | Pluggable token counter, section budgets, frozen session hot memory, and rendered-memory accounting. The default counter is an estimate. |
| Memory loop | Admission scanning, exact and token-overlap deduplication, episode observation, idempotent review, and pending approval. |
| Evidence and revision | Named outcomes, conflict marking/resolution, optimistic revision checks, pending revisions of verified memories, and derived-knowledge invalidation. |
| Procedures | Skill/tool rendering, maturity levels, and an optional reflex path with precondition checks and fallback to recall. |
| Persistence | JSON and SQLite, event replay, incremental SQL writes, revision-aware in-process cache, and a shared local daemon. |

The fixed evaluation has **8 datasets and 33 cases**. The current and hashing
hybrid baselines both reach Recall@K / MRR **1.0**, with zero stale pollution
and forbidden-memory hits. The tight-budget baseline still loses recall;
these small fixtures establish regression behavior, not production accuracy
or proven token savings. See [the roadmap](ROADMAP.md) for measurements,
remaining work, and release gates. `fibbrain_assess` is a design proposal and
is not an available tool.

> [!WARNING]
> FibMind is a local, single-user prototype. Isolation, evidence attribution,
> compression, and replay invariants still need hardening. Do not use it as a
> multi-tenant production service or store secrets in it.

## Quick Start

Requires Python **3.11+**. CI tests Python 3.11 and 3.12. SQLite-backed storage
requires a Python SQLite build with FTS5 support.

```bash
cd fibmind
uv sync --locked --extra dev
.venv/bin/python examples/demo.py
.venv/bin/pytest -q
```

The demo writes two sample JSON stores to `.fibmind/demo/` (gitignored). The
tracked copies in `data/` double as the conformance fixture for
`tests/test_contract.py`; regenerate them only on purpose with
`examples/demo.py --fixture`. SQLite is the recommended backend for a
persistent agent memory service.

The local SQLite runtime is pinned to **3.51.3** in `.sqlite-version`. To
reproduce the CPython 3.12.13 environment on macOS or Linux:

```bash
uv sync --locked --extra dev --python 3.12.13
.venv/bin/python scripts/setup_sqlite_runtime.py
.venv/bin/python -c 'import sqlite3; print(sqlite3.sqlite_version)'
```

The installer requires a C compiler, `make`, Perl, and development libraries
for xz, bzip2, and libffi (Command Line Tools and Homebrew `xz` on macOS). It verifies the
official source checksums, builds a private interpreter with SQLite 3.51.3
and the original OpenSSL 3.5.5, and switches `.venv` after an FTS5/import
smoke check. Packages and memory files are preserved. Sources, logs, the
interpreter, and the previous venv configuration live in `.fibmind/runtime/`;
keep that directory while using this venv. Reapply the installer after
recreating `.venv`, and restart existing MCP/daemon processes to use the new
runtime. This does not upgrade other Python environments or the system SQLite
command.

`uv` is the supported way to create the environment (`pip install uv` works if
you do not have it). Development checks, all of which CI runs on every push:

```bash
.venv/bin/ruff check src tests evals examples scripts     # lint
.venv/bin/ruff format --check src tests evals examples scripts
.venv/bin/pytest -q
.venv/bin/python -m evals.gate                    # evaluation suite must not fall below evals/floor.json
# Native DSH plugin checks (also run in CI; Node.js 22)
npm --prefix integrations/dsh install
npm --prefix integrations/dsh test
```

The evaluation floor is a merge gate, not a report. Raise it when the suite
improves; do not lower it to make a build pass.

### Share memory across applications

Generate and preview Codex, Claude Code, and AGY configuration for the same
human owner and existing database:

```bash
.venv/bin/fibmind-config --owner abbila --store .fibmind/memory.db \
  --profile full --output-dir .fibmind/integrations
.venv/bin/fibmind-connect --bundle .fibmind/integrations
.venv/bin/fibmind-connect --bundle .fibmind/integrations --apply
```

The generator defaults to a smaller `memory` tool profile. This repository uses
`full` to preserve its existing AGENTS workflow. Registration backs up existing
host settings and installs a marked context-use instruction block. See
[cross-application memory](docs/cross-application-memory.md) for identity,
visibility levels, and protocol versus live-application verification.

`fibmind-config` defaults to `--profile memory`, while starting the server
directly defaults to `--profile full`. The `memory` profile exposes context,
admission, reflection, retrieval, procedures, and upkeep; it omits planning,
episode review, reflex, and command verification tools.

Before each request, call `fibbrain_context` with the human owner, the absolute
`workspace_path`, and the calling host's actual conversation ID as `session_id`.
Omit the session ID if the host does not expose it, and avoid session-scope
operations in that case. Never invent an ID or reuse another conversation's.
The server derives working labels from the workspace and git branch, respecting
explicit IDs and `.fibmind-project.json`. Reuse the returned `state` on later
calls. Application names belong in `metadata.host`, never in `owner`.

Save standing profile/preferences with `fibbrain_remember` and explicit
`context_level="user"`; use `"workspace"` or `"project"` for narrower facts.
`has_task_context=false` can still include profile information. Evidence labels
describe past checks; recalled text remains historical background.

## Retrieval Evaluation

The evaluation suite compares six paths over the same fixed datasets:

- no persisted memory;
- static `AGENTS.md`-style context;
- a bounded Hermes-style hot-memory snapshot;
- FibMind's current lexical search and graph context expansion;
- the same engine with offline hashing vectors and RRF;
- a 120-token budget with hot memory rendered first.

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
recall. `git_history.json` adds human-labelled cases from actual repository
commits, including a Chinese query and unrelated/project-isolated queries; each
fact records its source commit. Reports include Recall@K, MRR, context
answerability, irrelevant/stale pollution, forbidden-memory hits, context size,
and P50/P95 latency. The budgeted baseline counts only memories actually
rendered into its prompt when computing retrieval metrics. The
`real_coding_benchmark.json` fixtures cover coding scenarios; their name does
not establish that they are independently reviewed live-agent traces.

To compare **model answers** and **provider-reported tokens**, record answers
from the same model for each baseline and pass a reviewed JSON file with
`--answers path/to/answers.json`. Each answer needs its `baseline`, `case_id`,
`answer`, human-reviewed `correct` boolean, and provider `input_tokens` and
`output_tokens`; the file also needs `model` and `judged_by`. The report adds
answer accuracy, coverage, and actual input/output token usage per baseline.
Missing answers are reported as incomplete coverage rather than counted as
correct. The bundled repository-history dataset does not stand in for a live
agent session: `evals.import_dsh_session` can produce candidate datasets from
one, and its auto labels need independent review before using them as a gate.

The suite reports `estimated_tokens`: a deterministic, model-agnostic size
estimate for relative comparisons. The runtime already accepts a custom
`TokenCounter`, but a bundled model tokenizer and calibrated savings target
remain open. The fixed retrieval gate covers `fibmind_current`; it does not
measure native application behavior, automatic write precision, or end-to-end
answer correctness without reviewed answers. See [`ROADMAP.md`](ROADMAP.md)
for the current results and remaining acceptance criteria.

## Design Constraints

Six properties the implementation is built around. They are worth stating
because each rules out a shape that looks reasonable:

**The log is the source of truth.** Every semantic change records a
`MemoryEvent`; `FibMind.rebuild_from_log` reconstructs the whole forest from it.
Node and edge tables are a cache you can throw away. This is what makes a store
replayable when a better summarizer comes along. Cross-device synchronization
and log merging are not implemented. Recall counters (`familiarity`,
`access_count`) are the exception: they describe usage rather than knowledge and
are not replayed.

**Recall is not evidence.** Activation may raise `familiarity`; normal search
is read-only, and familiarity is absent from relevance scoring. Named external
outcomes affect `confidence`. Re-reading a memory or agreement from another
model does not establish correctness. Evidence inheritance during compression
still needs hardening, as tracked in the roadmap.

**Admission scans candidates.** It rejects candidates carrying API keys,
private keys, connection strings, prompt-injection phrasing, or invisible
Unicode, and returns the finding kind. Prefer `fibbrain_remember`: the direct
store write API bypasses admission. Background rendering also scans recalled
content, but these checks are not a guarantee that arbitrary content is safe.

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
Event       One entry in the replayable log; forget redacts historical content.
Scope       session / personal / knowledge — who a memory belongs to.
Status      active / pending / stale / refuted — whether normal recall may return it.
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
can be traced back. The optional `require_cross_provenance` gate additionally
requires multiple sessions/hosts or existing verification evidence.

Personal visibility also has an explicit `context_level`: `user` follows the
same human across projects, `workspace` stays in one workspace, and `project`
stays in one workspace/project pair. Broader visibility is opt-in; legacy
unlabelled records retain exact matching. Session scratch is visible in the
same session or a shared named task within its owner/workspace/project boundary.

Normal search and context return only `active` memories. `mark_stale` retires a
superseded memory while preserving its provenance; a `refuted` outcome also
removes a memory from normal recall. Maintenance queries can explicitly request
inactive statuses. `revise` normally corrects the content and makes the memory
active and unproven again. Pass `expected_version` to reject a stale edit, or
`protect_verified=True` to stage a verified memory's replacement as a pending
`VERSION_OF` candidate until confirmation. Conflict marking and resolution
are explicit operations; general automatic contradiction detection is still
planned. Personal/session memories require an exact owner match; omitting
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
- Folding uses a pluggable `Summarizer`. The default `ExcerptSummarizer` still
  concatenates truncated excerpts. `StructuredSummarizer` extracts directives,
  conditions, exceptions, and evidence; `CallableSummarizer` wraps an external
  model with provenance and a fidelity check. It does not configure or call an
  LLM by default. Folding retains sources through `folded_into` pointers.
  Preserving source evidence and constraints across all compression paths is
  ongoing work; a shorter summary is not a newly verified general claim.

## Main Operations

```python
from fibmind import FibMind, RelationType, Verdict

memory = FibMind()
tree_id = memory.create_tree("code", title="Code Tree")
node_id = memory.append("code", title="Login bug", content="401 after token expiry")
related_id = memory.append("code", title="Token refresh", content="Refresh before retrying login")

memory.link_nodes(node_id, related_id, RelationType.RELATED_TO)
memory.promote_node(node_id)
results = memory.search_from(node_id, depth=2)
activated = memory.search_from(node_id, depth=2, reinforce=True)

# Attach outside evidence — the only thing that changes how a memory ranks.
memory.record_outcome(node_id, Verdict.CONFIRMED, source="pytest tests/test_auth.py")

# Correct a memory (confidence resets), retire an old one, or delete it for good.
memory.revise(node_id, content="401 only after refresh-token expiry")
memory.mark_stale(related_id, reason="superseded by the refresh-token decision")
memory.forget(related_id, reason="user asked for removal")

# Distil repeated observations into one shareable claim (needs >= 3 supporters).
supporting_ids = [
    memory.append("error", title=f"Upload timeout {i}", content="A transient timeout ended an upload")
    for i in range(3)
]
claim_id = memory.promote_to_knowledge(
    title="Uploads need an explicit retry budget",
    content="Large uploads die at the first transient timeout without a retry budget.",
    supporting_node_ids=supporting_ids,
)

# Nodes and edges are a cache; this rebuilds them from the log alone.
replayed = FibMind.rebuild_from_log(memory.events)
```

`search_from()` is read-only by default. Pass `reinforce=True` when a query
should behave like memory activation — note this raises `familiarity` only, and
familiarity does not affect ranking.

`search()` is the query-based counterpart: it ranks indexed candidates for a
text query and never mutates nodes. Pass `owner=` to exclude other
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

The CLI defaults to the `full` tool profile. Add `--owner abbila` to pin the
human identity, or `--profile memory` for background and memory management
without command execution or the optional cognitive workflow.

### Tools

The following tables describe the `full` profile. The smaller `memory` profile
is defined by `MEMORY_TOOLS` in `src/fibmind/mcp_server.py`.

Recall and record:

| Tool | Purpose |
| --- | --- |
| `fibbrain_context` | Read-only user profile plus task-relevant memory/knowledge, with budget and evidence labels. |
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
| `fibbrain_verify` | Execute a check and record confirmed/refuted/inconclusive, with optional target-file and output-pattern checks. |
| `fibbrain_reflex` | Return a verified answer or procedure without an LLM; procedure execution requires `execute=true`. |
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
| `fibmind_revise` | Correct a memory; optionally enforce `expected_version` or stage a pending replacement with `protect_verified`. |
| `fibmind_mark_conflict` | Flag contradictory memories and link them with `CONTRADICTS`. |
| `fibbrain_resolve_conflict` | Record a sourced resolution and refute the losing claim. |
| `fibmind_mark_stale` | Retire superseded content while keeping provenance. |
| `fibmind_forget` | Delete a memory permanently, including from log history. |
| `fibmind_promote_knowledge` | Turn several observations into one shared claim. |

For cross-application background, call `fibbrain_context` before answering and
`fibbrain_remember` when durable information appears. Explicit
`context_level=user` carries personal background across projects; project facts
remain project-specific. The optional cognitive workflow uses `fibbrain_plan`
then `fibbrain_recall` before work.
`fibbrain_coordinate` names the next capabilities; `fibbrain_advise` can still
be called directly for a specific action. `fibbrain_remember` will skip a
duplicate or a dump. A store that only ever writes still degrades unless
`fibbrain_reflect` records whether those memories held up.

### Host checks

`tests/test_cross_app_mcp.py` launches independent stdio MCP servers from
different directories and checks shared persistence, profile visibility,
provenance, isolation, and read-only background. This verifies the protocol
path. Native application checks depend separately on each host's credentials,
model availability, and tool permissions; protocol success does not prove
that all three applications invoked the tool.

See [cross-application memory](docs/cross-application-memory.md) and
[`examples/host_mcp_prompt.md`](examples/host_mcp_prompt.md) for manual checks.
Restart or open a new host session after changing MCP configuration.

### DeepSeek-Harness

Two ways, both in this repository:

- **Zero-code**: [`examples/dsh/`](examples/dsh/README.md) mounts the MCP server
  through `@deepseek-ai/dsh-mcp-client`; the model calls `mcp__fibbrain__*` tools.
- **Native plugin**: [`integrations/dsh/`](integrations/dsh/README.md) is a
  Cordis plugin that recalls before every prompt, advises before every tool call,
  observes results, reviews the session on turn end, and publishes verified
  procedures as skills — without the model having to remember to call anything.

### Register with Codex

Codex registrations are global, so run this from the repository root and let
`$PWD` expand to absolute paths; Codex can then launch the server regardless of
its current working directory:

```bash
codex mcp add fibmind -- \
  "$PWD/.venv/bin/python" \
  -m fibmind.mcp_server \
  --store "$PWD/.fibmind/memory.db" \
  --owner abbila --profile full
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
`fibbrain_context` with `owner="abbila"` and this checkout's absolute
`workspace_path`. An empty new store should return empty context successfully.
This repository's `AGENTS.md` requires context before each request and tells
Codex to:

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

A project-scoped server starts from the repository root, so relative paths
work. The committed [`.mcp.json`](.mcp.json) uses this server/store, with owner
and host provenance supplied through environment variables. A manual equivalent
with an explicit owner is:

```bash
claude mcp add --scope project --transport stdio fibmind -- \
  .venv/bin/python \
  -m fibmind.mcp_server \
  --store .fibmind/memory.db --owner abbila --profile full
```

### Shared Daemon Mode (Multi-Tool Workstation)

When several applications run concurrently, a shared daemon centralizes their
memory requests, cache, and process-local lock. All participating hosts must
connect to that daemon; separate stdio processes still open the store
independently, and can still contend with it.

```bash
# Run the shared memory daemon (binds to 127.0.0.1:8765 by default)
.venv/bin/fibmind-daemon --store "$PWD/.fibmind/memory.db" --owner abbila --port 8765

# Or via python directly
.venv/bin/python -m fibmind.mcp_server --transport sse \
  --store "$PWD/.fibmind/memory.db" --owner abbila --port 8765
```

SSE and `--transport streamable-http` are supported. The daemon binds to
`127.0.0.1` by default and refuses non-loopback binding unless
`--allow-insecure-remote` is supplied. Optional token authentication uses
`FIBMIND_TOKEN` or `--token`; when configured, requests other than `/health`
require a Bearer header or token query parameter. These are local workstation
controls, not a multi-tenant authorization model.

### Automated Verification Loop (`fibmind-verify` / `fibbrain_verify`)

Use `fibmind-verify` or `fibbrain_verify` to run a physical check such as a
test, compiler, or linter. Environment failures, zero collected tests, and
output that misses supplied target files or assertion patterns are classified
as `inconclusive`; otherwise success confirms and failure refutes the memory.
Target matching applies to both success and failure. It uses command-output
heuristics, not coverage instrumentation, so callers still need to choose a
check that actually establishes the claim.

```bash
# Verify a decision against physical test execution
.venv/bin/fibmind-verify --owner abbila \
  --workspace-id fibmind --project-id fibmind --node-id <id> \
  --target-files tests/test_verify.py -- \
  .venv/bin/python -m pytest -v tests/test_verify.py
```

### Reflex and guarded revision

`fibbrain_reflex` first looks for a mature procedure, then a high-confidence
declarative memory. It checks recorded file hashes, conflicts, optional file
existence, and model-specific metadata. `execute=false` is the default; with
`execute=true`, it runs procedure steps and falls back to recall when a step
fails.

This execution path is experimental: it does not yet execute a procedure's
separate `verify` specification, and its MCP file checks use the server's
working directory. These gaps, summary evidence inheritance, and protected
revision race handling are priorities in [the roadmap](ROADMAP.md).

### Architecture

```text
Codex / Claude Code / AGY / DSH / other MCP harnesses
        │ fibbrain_* / fibmind_* MCP tools
        ▼
MCP Server                (src/fibmind/mcp_server.py)
        ▼
FibBrain                  (background / recall / admission / evidence / optional workflow)
        ▼
MemoryService             (src/fibmind/service.py — revision cache / lock / persist)
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

The current SQLite schema is **v7**, including working-context identity,
lifecycle status, memory kind, node revisions, and the persistent FTS5 index.
The optional vector cache is stored alongside it. Older schemas upgrade in
place on open. SQL writes update changed nodes, trees, edges, and events;
the transaction still loads and compares the full forest, so fully lazy
retrieval and an explicit mutation journal remain future work.
`fibmind-migrate` converts a JSON store to SQLite.

```bash
.venv/bin/python -m fibmind.migrate \
  --from-json data/demo-memory.json \
  --to-sqlite data/demo-memory.db
```

The migration refuses to overwrite an existing SQLite target.
