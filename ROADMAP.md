# FibMind Roadmap

## Goals

Further FibMind development focuses on four verifiable goals:

1. Reduce the tokens an agent spends to obtain historical context;
2. Improve context recall accuracy across sessions and tasks;
3. Automatically summarize and save memories worth keeping long term;
4. Distil memories that have been verified repeatedly into traceable, revisable
   knowledge.

Work is ordered by "prove the value first, then expand the capabilities":

```text
P0 Evaluation baseline
  ↓
P1 Architecture and identity model
  ↓
P2 Hybrid retrieval
  ↓
P3 Token budget and two-tier memory
  ↓
Release v0.2 Retrieval MVP; use the evaluation to decide whether to continue
  ↓
P4 Automatic memory loop
  ↓
P5 Summarization, knowledge governance, and security
```

The first three phases produce the first MVP whose value can be verified. P3 is
the decision gate for further investment: if the evaluation cannot show FibMind
keeping or improving accuracy while also reducing tokens, non-core features are
not extended.

## Status

As of 2026-08-03, the first version of the P0 evaluation infrastructure has
landed:

- [x] Fixed JSON dataset format and validation;
- [x] Four baselines: no memory, `AGENTS.md`, Hermes hot memory, and current
  FibMind;
- [x] Recall@K, MRR, context answerability, irrelevant/stale pollution,
  forbidden-memory hits, estimated tokens, and latency metrics;
- [x] Command-line table and full JSON report;
- [x] Four sample datasets: project history, stale memory, conflicting memory,
  and cross-session;
- [ ] Extend the datasets with real project traces;
- [ ] Add model-answer evaluation to replace the current context-answerability
  proxy metric;
- [ ] Calibrate the 30% token-savings threshold once P3 uses the target model's
  tokenizer.

Update 2026-09-24: `git_history.json` has been compiled from the real
repository commit history and labelled by hand, including Chinese and unrelated
queries. `evals.runner --answers` can merge independently judged model answers
and provider token usage. The repository does not yet have complete labelled
real DSH sessions or recorded model answers, so the three open long-term
acceptance items above still have to be completed on real traces.

The first baseline run then drove a P1/P2 governance slice: `active/stale/refuted`
lifecycle states, non-active memories excluded by default, strict owner
matching, and English stop-word filtering. Across 15 fixed scenarios (including
cross-project / session scratch isolation), current FibMind keeps Recall@K at
1.0, while the stale-pollution and forbidden-memory hit rates drop from non-zero
to 0 and accuracy on queries with no relevant memory rises from 0 to 1.0. The
sample set is small; these results serve as regression checks, not claims about
production performance.

## P0: Evaluation Framework

This is the top priority. Without evaluation there is no way to show that
FibMind beats `AGENTS.md`, a fixed hot memory, or other memory implementations.

### New Layout

```text
evals/
├── datasets/
│   ├── project_history.json
│   ├── stale_memory.json
│   ├── conflicting_memory.json
│   └── cross_session.json
├── runner.py
├── metrics.py
└── baselines.py
```

### Scenarios

- Recover a past design decision across sessions;
- Find the cause of a given bug and its verification command;
- Do not return stale content when old and new memories conflict;
- Personal memories of different owners must not leak into each other;
- Do not force a recall when no relevant memory exists;
- Still find rare but important information after long accumulation.

### Metrics

- `Recall@K`;
- `MRR`;
- Final answer accuracy;
- Irrelevant-memory injection rate;
- Stale-memory pollution rate;
- Context input tokens;
- Query latency P50/P95.

### Baselines

1. No memory;
2. `AGENTS.md` only;
3. Hermes-style fixed hot memory;
4. Current FibMind keyword search;
5. Optimized FibMind.

### Acceptance

- The evaluation can be rerun with fixed data and fixed parameters;
- The report shows accuracy, pollution, tokens, and latency together;
- Initial suggested target: with no drop in final answer accuracy, cut
  dynamic-memory context tokens by at least 30%. Calibrate this threshold once
  the first round of baseline data is available.

## P1: Project Structure and Identity Model

The first cut covers only the identity fields and unified visibility; the
implementation plan is in [`docs/plans/02-identity.md`](docs/plans/02-identity.md). Move
modules only after `is_visible` is stable, and do not mix the split into the
same change as the identity rules.

> Status (2026-09-18): step one is done. The visibility rules moved to
> `visibility.py` (`is_visible` / `identity_key` / `promotion_identity`, pure
> functions; `FibMind.is_visible` only delegates); event replay moved to
> `replay.py` (one handler per `EventOp`; `rebuild_from_log` calls
> `apply_event`). `graph.py` went from 1400 → 1260 lines, with the public API
> unchanged. Next candidates, ordered by how clean the boundary is:
> folding/compression (the `compress_overflow` family), forgetting/log redaction
> (the `forget` family), and evidence propagation
> (`record_outcome` / `_propagate_refutation`).

`graph.py` currently handles writes, lifecycle, compression, graph traversal,
permission filtering, knowledge promotion, and event replay all at once. Adding
retrieval and automatic summarization on top would make it hard to maintain, so
the boundaries are split first.

### Target Layout

```text
src/fibmind/
├── domain/
│   ├── models.py
│   ├── events.py
│   └── policies.py
├── ingestion/
│   ├── admission.py
│   ├── dedup.py
│   └── security.py
├── retrieval/
│   ├── lexical.py
│   ├── semantic.py
│   ├── fusion.py
│   └── graph_expand.py
├── governance/
│   ├── evidence.py
│   ├── lifecycle.py
│   └── knowledge.py
├── context/
│   ├── hot_memory.py
│   ├── planner.py
│   └── renderer.py
├── storage/
│   ├── sqlite.py
│   └── repository.py
├── service.py
└── mcp_server.py
```

Split in steps, keeping the existing public API and tests working throughout;
there is no need to move every module at once.

### Data Model Extensions

- `workspace_id`;
- `project_id`;
- `session_id`;
- `task_id`;
- `branch`;
- `commit_sha`;
- `source_uri`;
- `content_hash`;
- `valid_from` / `valid_until`;
- `status`: `active` / `stale` / `refuted` / `forgotten`;
- `memory_kind`: `preference` / `decision` / `error` / `requirement` / `summary` / `knowledge`.

### Isolation Rules

- Without an explicit owner, do not read all personal memories by default;
- Project, workspace, and owner filtering must be applied uniformly at the
  retrieval entry point;
- Graph expansion must not bypass the visibility rules that apply to direct
  retrieval.

### Acceptance

- Retrieval, governance, context, and storage can be replaced independently;
- `graph.py` no longer carries every business responsibility;
- Project, session, branch, and user isolation are fully tested;
- Existing MCP tools stay compatible.

## P2: Hybrid Retrieval

> Status (2026-09): a first version has landed. `retrieval.py` replaces the full
> scan with an inverted index + BM25; an optional `EmbeddingProvider` (offline
> hashing / OpenAI-compatible) is fused in via RRF; `fibmind_explain_recall`
> reports each result's signals and the reasons candidates were excluded. P95 is
> about 48 ms at 10,000 nodes, and the fixed evaluation is unchanged at Recall@K
> 1.0 / stale pollution 0 / forbidden hits 0. FTS5 tables were not adopted: the
> in-memory inverted index already meets the target, and it avoids maintaining
> two indexes across the SQLite and JSON backends. Persisting the vector cache is
> left for later.
>
> Gate passed (measured 2026-10-09): `fibmind_hybrid` (hashing vectors +
> prefix-4 token features + RRF) now matches `fibmind_current` with
> Recall@K 1.0, MRR 1.0, and accuracy on queries with no relevant memory at
> 1.0000 (up from 0.5000), while irrelevant injection matches keyword baseline
> at 0.2679 (down from 0.3420). The noise floor calibration (min_similarity=0.25)
> and multi-modal RRF normalization successfully solved negative query leakage.

The current full keyword scan suits a prototype, not a long-term memory store
that keeps growing. The target is a pluggable retrieval pipeline of "lexical
recall + semantic recall + metadata filtering + evidence-based ranking".

### P2.1 SQLite FTS5

- Build an FTS5 index over titles, bodies, tags, and categories;
- Rank with BM25;
- Define a repeatable Chinese tokenization strategy;
- Update the index incrementally on write, revise, and forget;
- Support filtering by project, owner, time, kind, and status.

### P2.2 Semantic Embedding

Define a pluggable interface:

```python
class EmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Requirements:

- Support local embeddings;
- Support OpenAI-compatible embeddings;
- Fall back to pure FTS5 when embeddings are disabled;
- Cache vectors by `content_hash`;
- A provider failure must not break basic lexical retrieval.

### P2.3 Fusion Ranking

```text
Query
 ├── BM25 Top 30
 ├── Vector Top 30
 ├── Metadata and visibility filtering
 └── RRF fusion
       ├── confidence weighting
       ├── freshness / validity
       ├── relation boost
       └── rerank → Top K
```

Graph relations are used only to supplement recall, trace sources, and weight
ranking; they are no longer the primary retrieval entry point.

### Explainable Recall

Add `fibmind_explain_recall`, which returns:

- Why each result matched;
- The matched terms;
- Whether a candidate came from BM25, vectors, or graph expansion;
- The confidence and source evidence used;
- Why older memories were excluded.

### Acceptance

- Queries no longer load and traverse every node each time;
- Hybrid retrieval beats the current keyword baseline on the P0 datasets;
- Performance tests exist for 10,000 and 100,000 memories;
- Every recall result can explain its source and ranking signals.

## P3: Token Budget and Two-Tier Memory

> Status (2026-09): a first version has landed. `recall` renders in three
> sections under a token budget (hot / search / expand) and returns
> `budget.used_tokens`, per-section usage, and truncation reasons; the counter is
> pluggable and defaults to a deterministic estimate. L0 hot memory is selected
> by kind (preference / requirement / decision / knowledge) and confidence,
> frozen as a `hot` node for the session, and refreshed for the next session
> after review. The evaluation adds `fibmind_budgeted` (a hard 120-token budget +
> hot memory up front): on the small fixed datasets Recall@K stays 1.0, but MRR
> is 0.84, answerability 0.67, and irrelevant injection 0.44 — the cost of the
> always-on prefix is counted as-is. The 30% savings threshold has to be
> calibrated on real traces (the S1 import); the current datasets are too small
> to tell.

Upgrade the current character-length limit to a model token budget, and combine
Hermes-style stable hot memory with FibMind's on-demand long-term recall.

### L0 Hot Memory

Injected automatically at session start:

- Stable user preferences;
- Fixed project conventions;
- Frequently used environment information;
- Key project constraints.

Rules:

- Default budget of roughly 800–1200 tokens;
- Strict capacity limit;
- A frozen snapshot is used within a session;
- The next snapshot is updated after the session ends;
- When full, merge, replace, or reject writes; it must not grow without bound.

### L1 Dynamic Long-Term Memory

Recalled on demand for the current task:

- Past decisions and their rationale;
- Bugs, causes, and fixes;
- Test results;
- Requirement changes;
- commit/branch context;
- Known risks and open issues.

### Context Planner

```text
Total budget
├── Hot memory budget
├── Direct hit budget
├── Relation expansion budget
├── Raw evidence budget
└── Safety margin
```

Features:

- Count tokens for the target model;
- Choose the next memory by marginal value;
- Merge similar content;
- When over budget, keep high-confidence raw evidence first;
- Return actual usage, the budget, and truncation reasons.

### Acceptance

- Results never exceed the specified token count;
- The hot-memory snapshot stays stable within a session;
- Output `used_tokens`, `budget_tokens`, and the token cost of each memory;
- Verify token savings in the P0 evaluation instead of only comparing character
  length.

## P4: Automatic Memory Loop

> Status (2026-09): the first half has landed (S2): `observe` persists to a
> session-scoped episode; `fibbrain_review_session` deterministically extracts
> decisions / errors / corrections / risks / verification / changed files /
> summary / procedures (S3), with three modes (candidates / approve / auto), and
> is idempotent; there is a `pending` status with approve / reject tools. The
> native DSH plugin (`integrations/dsh/`) triggers review automatically on
> `agent/turn-stopping` and observes automatically on `tools/result`. Semantic
> near-duplicate detection and conflicting-version detection (automating
> VERSION_OF) are not done.

Removes the current reliance on the agent calling `fibmind_append` on its own
initiative.

### Write Admission

Add `MemoryAdmissionPolicy`.

Worth saving:

- Stable user preferences;
- Important design decisions and their rationale;
- User corrections;
- Verified causes of errors;
- Information that cannot easily be rediscovered;
- Conventions that stay valuable across sessions.

Skipped by default:

- Transient conversation;
- Large raw logs;
- Information that can be rediscovered quickly from the code;
- Exact duplicates;
- Speculation with no conclusion.

### Deduplication and Conflict Handling

- Content-hash deduplication;
- Semantic near-duplicate detection;
- Detecting old and new versions of the same fact;
- On conflict, create `VERSION_OF` or trigger a revise;
- Two contradictory facts must never both stay active.

### Session Review

New MCP tools:

- `fibmind_commit_session`;
- `fibmind_review_pending`;
- `fibmind_approve_memory`;
- `fibmind_reject_memory`.

Extracted after a session ends:

- objective;
- key decisions;
- changed files;
- tests;
- unresolved risks;
- user corrections;
- knowledge promotion candidates.

Three modes are supported:

1. Write automatically;
2. Approve before writing;
3. Generate candidates only, without saving.

### Acceptance

- Automatic writes are idempotent;
- Rerunning Session Review produces no duplicate memories;
- Wrongly admitted memories can be rejected or revoked;
- Automatic write accuracy is part of the P0 evaluation.

## P5: Summarization, Knowledge Governance, Security, and Observability

> Status (2026-09): a first version has landed (S7). `Summarizer` / `Planner` are
> pluggable (`distill.py`); the default is still deterministic, and the LLM
> version records model / prompt_version / sources and can be regenerated from
> the log. When supporting nodes are refuted, derived knowledge loses confidence
> in proportion to its surviving support, and becomes `stale` once all support is
> gone. Writes are scanned for secrets / injection / invisible Unicode
> (`safety.py`). `fibbrain_tune` uses confirmed vs refuted evidence to propose one
> bounded weight adjustment, adopts it only if it passes the P0 evaluation gate,
> and logs every attempt. `fibbrain_revalidation_candidates` lists knowledge and
> procedures that have gone unconfirmed for a long time (output only).
> `fibmind_status` reports kinds / pending / invalidated knowledge / outcome
> counts / weights / tuning history. Not done: backup, restore, and export; a
> database encryption interface; automatic adjustment of the admit threshold
> (pressure is only reported, never tightened automatically).

### Real Summarization

Replace the current excerpt concatenation with a pluggable Summarizer. A summary
must:

- Keep facts and the rationale behind decisions;
- Keep sources such as files, commits, and tests;
- Distinguish facts, speculation, and open questions;
- Not summarize conflicting memories into a definite conclusion;
- Be regenerable from the raw events;
- Record the model, prompt version, and input sources used.

### Knowledge Lifecycle

- Promotion requires multiple independent pieces of evidence;
- Evidence can be withdrawn;
- When a supporting memory is refuted, the confidence of derived knowledge is
  lowered automatically;
- When a new version appears, the old knowledge is marked stale;
- It can explain which memories a piece of knowledge was derived from;
- Knowledge that has gone unconfirmed for a long time is periodically
  revalidated.

### Security

Checked before writing:

- API keys, tokens, passwords, and private keys;
- Prompt injection;
- Invisible Unicode;
- Oversized content and malicious metadata;
- owner/project boundary violations.

Also provided:

- Write approval;
- Deletion audit;
- Backup, restore, and export;
- Redaction of sensitive fields;
- An extension interface for local database encryption.

### Observability

- Query count, hit rate, and empty-recall rate;
- Tokens injected per context;
- How many times each memory is used, confirmed, and refuted;
- Recall latency;
- Acceptance and revocation rates of automatic writes;
- Hit contribution of hot memory versus long-term memory.

Add `fibmind_status` and a diagnostics command showing store size, index state,
memories awaiting approval, invalidated knowledge, and recent evaluation
results.

## Deferred

Not prioritized until P0–P3 are complete:

- More Fibonacci layers;
- More graph relation types;
- A complex visual editor;
- A remote distributed graph database;
- Multi-tenant SaaS;
- Agents generating large numbers of relation edges automatically;
- Fibonacci parameter tuning.

None of these directly shows more accurate recall or lower token use.

## Release Milestones

### v0.2 — Retrieval MVP

Scope: P0–P3.

Release conditions:

- Hybrid retrieval beats the current keyword baseline;
- The token budget can be strictly enforced;
- Hot memory and dynamic long-term memory work together;
- owner/project isolation tests pass;
- Reproducible reports on effectiveness, tokens, and latency exist.

### v0.3 — Automatic Memory

Scope: P4.

Release conditions:

- Session Review is idempotent;
- Automatic admission, deduplication, and conflict handling work;
- Approval mode is supported;
- Automatic write quality has a quantitative report.

### v1.0 — Governed Memory

Scope: P5.

Release conditions:

- Summaries are traceable and regenerable;
- Derived knowledge supports evidence-invalidation propagation;
- Security scanning, backup/restore, and diagnostics are complete;
- Benefits are demonstrated in long-term evaluation on real projects.
