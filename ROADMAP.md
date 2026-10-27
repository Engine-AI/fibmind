# FibBrain / FibMind Roadmap

Last reviewed: **2026-10-27** against the current `main` checkout.
The package version in `pyproject.toml` is **0.1.0** (alpha); the milestones
below are future acceptance gates, not released versions or delivery dates.

FibMind is the local memory engine. FibBrain supplies shared user context,
admission, evidence, and an optional cognitive workflow. The harness owns the
conversation, permissions, and execution.

## Goals

1. Share the same human's applicable background across Codex, Claude Code,
   Antigravity (AGY), and other MCP harnesses.
2. Retrieve relevant history across sessions while preserving owner,
   workspace, project, and session/task boundaries.
3. Reduce context cost without losing answer accuracy or critical constraints.
4. Capture durable decisions, errors, corrections, and procedures with a
   measurable admission/review loop.
5. Keep memories traceable, revisable, and grounded in external evidence.

Evaluation remains the gate for further expansion. A feature being implemented
does not mean its quality or release criteria have been met.

## Status

| Area | Implemented | Remaining acceptance work |
| --- | --- | --- |
| P0: evaluation | Eight datasets, six baselines, retrieval/pollution/token/latency reports, reviewed-answer import, and a CI floor. | Independently reviewed real-session traces, recorded model answers, automatic-write quality, and persistent-store performance. |
| P1: identity and structure | Shared visibility policy, event replay module, node-ID mutation guards, working-context inference, and explicit context levels. | Compression/replay boundary hardening and further separation of graph responsibilities. |
| P2: retrieval | BM25, persistent SQLite FTS5 candidates, CJK and file/symbol anchors, optional vectors, RRF, persistent vector cache, and explanations. | Candidate-pruning/fallback robustness, real semantic-provider evaluation, and large-store benchmarks. |
| P3: context budgets | Pluggable counter, section budgets, frozen L0 snapshots, and rendered-memory accounting. | Model tokenizer integration and savings without an accuracy regression. |
| P4: automatic memory | Admission, exact and token-overlap deduplication, episode observation, idempotent review, approval, and native DSH hooks. | Semantic deduplication, automatic version/conflict detection, and quantified admission quality. |
| P5: governance | Summarizer/planner interfaces, structured distillation, outcome propagation, scanning, tuning gate, and diagnostics. | Faithful compression with sound evidence inheritance, backup/restore/export, encryption interface, and broader observability. |
| P6: reflex and metacognition | Optional reflex, file/model preconditions, protected revisions, explicit conflicts, and command verification. | Procedure verification and workspace binding; knowledge-gap assessment is design only. |
| P7: cross-application context | Read-only background, user/workspace/project levels, shared owner, configuration bundles, native registration, and independent MCP-client tests. | Repeatable acceptance inside each native application and broader background-selection regression coverage. |

Current storage is SQLite schema **v7** or JSON. SQLite has incremental SQL
writes and FTS5 indexing; transactions still load and compare the full forest.
The revision cache avoids repeated warm loads, but fully lazy storage access is
not implemented. SSE and streamable HTTP provide a shared local daemon;
separate stdio processes still access the store independently.

## Current Priorities

Complete these before extending autonomous behavior:

1. **Make evidence attribution reliable.** Reflex must run a procedure's
   declared `verify` check, distinguish `inconclusive` from success, and reject
   empty execution as evidence. Bind command and precondition checks to the
   caller's workspace. Preserve file/command anchors from episode review.
   Enforce the recorded base revision when a pending replacement is confirmed.
2. **Preserve meaning and trust through compression.** Retain conditions,
   exceptions, source anchors, memory kind, and applicability metadata. A
   summary currently inherits the maximum source confidence; replace that
   shortcut with traceable evidence semantics that cannot make unchecked
   content appear independently verified.
3. **Harden background and retrieval selection.** Keep standing user
   preferences available alongside project facts; test budget truncation,
   model applicability, conflicts, and mixed visibility levels. Verify that
   bounded FTS candidates retain the best results under the final ranker and
   that index failures do not silently remove lexical recall.
4. **Prove quality on real work and growing stores.** Add reviewed traces,
   provider-token/model-answer measurements, and cold/warm SQLite benchmarks
   before claiming semantic gains, token savings, or production scalability.
5. **Add knowledge-gap assessment after its prerequisites and evaluation.**
   Keep `fibbrain_assess` as a planned interface until relevance, trust,
   routing, and false-known behavior have measurable acceptance criteria.

## P0: Evaluation Framework

### Available now

- [x] Fixed JSON datasets and validation.
- [x] Six baselines: `no_memory`, `agents_md`, `hermes_hot`,
  `fibmind_current`, `fibmind_hybrid`, and `fibmind_budgeted`.
- [x] Recall@K, MRR, context answerability, no-relevant accuracy,
  irrelevant/stale pollution, forbidden hits, estimated tokens, and P50/P95.
- [x] Table/JSON reports and a non-regression gate in `evals/floor.json`.
- [x] Hand-labelled repository-history cases, including Chinese and negative
  queries, with source commits.
- [x] `evals.import_dsh_session` for candidate session datasets and
  `evals.runner --answers` for reviewed model answers and provider token usage.

The eight datasets are `project_history`, `stale_memory`,
`conflicting_memory`, `cross_project`, `cross_session`, `git_history`,
`session_review`, and `real_coding_benchmark`. Together they contain
**33 cases**. Coding-scenario fixtures and auto-labelled imports do not replace
independently reviewed live-session evidence.

### Reproducible snapshot

Run from the repository root:

```bash
.venv/bin/python -m evals.runner --format table --output .fibmind/eval-report.json
.venv/bin/python -m evals.gate
```

Results checked on 2026-10-27 with default parameters (`top_k=5`,
`max_context_chars=5200`):

| Baseline | Recall@K | MRR | Context answerability | No-relevant accuracy | Mean estimated tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| No memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.00 |
| Static AGENTS.md | 0.0345 | 0.0345 | 0.0345 | 0.7500 | 11.67 |
| Hermes-style hot memory | 0.3793 | 0.3276 | 0.3793 | 0.7500 | 49.88 |
| FibMind current | 1.0000 | 1.0000 | 0.8966 | 1.0000 | 95.70 |
| FibMind hybrid | 1.0000 | 1.0000 | 0.8966 | 1.0000 | 95.70 |
| FibMind budgeted | 0.9310 | 0.8218 | 0.6207 | 0.7500 | 69.85 |

Current and hybrid both have stale pollution **0**, forbidden-case rate **0**,
and irrelevant injection **0.2596**. The budgeted path has irrelevant injection
**0.3990**. Hybrid uses offline feature hashing in this suite; matching the
lexical baseline does not demonstrate a semantic retrieval improvement.

The CI floor currently gates **`fibmind_current` only**: Recall@K ≥ 1.0,
MRR ≥ 0.98, no-relevant accuracy ≥ 1.0, stale pollution = 0, and forbidden-case
rate = 0. Do not lower the floor to make a change pass.

Context answerability is a fixture-term proxy, not model-answer accuracy.
Estimated tokens are not provider billing. Baselines use the in-memory engine,
so their latency does not include persistent-store startup, full-forest loads,
or native-host overhead.

### Remaining acceptance

- [ ] Import complete real agent sessions, retain provenance, and independently
  review relevant, irrelevant, stale, conflicting, and isolated cases.
- [ ] Commit reviewed same-model answers and actual input/output token usage
  for comparable baselines, with explicit coverage.
- [ ] Measure automatic write precision, correction/revocation behavior, and
  long-term recall after repeated review/compression.
- [ ] Add persistent SQLite and native-host evaluation paths.
- [ ] Calibrate the proposed 30% context-token saving target without a drop in
  answer accuracy; the current tight-budget result does not meet that goal.

## P1: Project Structure and Identity Model

### Available now

- [x] `visibility.py` centralizes scope, owner, lifecycle, and working-context
  rules; direct recall and graph expansion share them.
- [x] `replay.py` reconstructs semantic state from events.
- [x] `workspace_id`, `project_id`, `session_id`, and `task_id` are stored
  fields; `revision` supports optional optimistic concurrency checks.
- [x] Agent-facing node-ID mutations and graph roots are visibility-checked.
- [x] `working_context.py` derives stable labels from the canonical workspace,
  git branch, explicit overrides, and `.fibmind-project.json`.
- [x] Broader personal visibility requires explicit `context_level=user` or
  `workspace`; legacy records keep exact matching.

The same human owner is used across hosts. Application names are provenance in
`metadata.host`, never separate memory owners. Session scratch follows the
existing same-session or shared-task rule inside the working-context boundary.

### Remaining acceptance

- [ ] Extend isolation/replay tests through compression, protected replacement,
  conflict resolution, knowledge propagation, and deletion.
- [ ] Separate folding, evidence propagation, and log redaction from
  `graph.py` while preserving public APIs and the conformance fixture.
- [ ] Keep branch, commit, and source-file provenance consistent across manual
  writes, session review, and distillation. These are not all dedicated fields
  today.
- [ ] Preserve read-only behavior for background, normal search, and warm
  recall; keep cache updates distinct from evidence.

The proposed directory split is an architectural direction, not today's
layout. Extract one responsibility at a time rather than combining a package
reorganization with visibility changes.
See [the identity plan](docs/plans/02-identity.md).

## P2: Hybrid Retrieval

### Available now

- [x] In-memory inverted index and BM25 with title, confidence, and recency
  signals.
- [x] SQLite FTS5 is integrated into normal recall as a candidate filter, with
  incremental insert/update/delete and older-schema backfill.
- [x] Shared tokenization supports CJK queries and code/file/symbol anchors.
- [x] Optional offline hashing or OpenAI-compatible embeddings, RRF fusion,
  and lexical operation when the embedding provider fails.
- [x] SQLite vector cache keyed by provider and content hash.
- [x] `fibmind_explain_recall` reports ranking signals and exclusion reasons.

The current flow is:

```text
Query
 ├── visibility/lifecycle/category eligibility
 ├── lexical candidates → optional SQLite FTS5 pruning → BM25
 ├── optional vector candidates
 └── RRF / title / confidence / recency / anchor signals
       └── Top K → optional graph expansion → budgeted rendering
```

### Remaining acceptance

- [ ] Test candidate limits against the final ranking, including many common
  terms, strong title/anchor matches, and more than one visibility domain.
- [ ] Make FTS index failures distinguishable from genuine empty matches and
  verify a useful lexical fallback.
- [ ] Evaluate a real semantic provider on reviewed cross-language/paraphrase
  cases; feature hashing is not a semantic model.
- [ ] Benchmark 10,000 and 100,000 memories, including cold/warm SQLite reads,
  writes, vector-cache reuse, and concurrent clients.
- [ ] Move filtering and candidate loading toward storage-level lazy access
  while preserving the single visibility policy.

## P3: Token Budget and Two-Tier Memory

### Available now

- [x] `TokenCounter`, deterministic `EstimatingCounter`, and a callable
  adapter for external tokenizers.
- [x] Recall sections for hot memory, direct search, and graph expansion,
  bounded by the configured counter.
- [x] Budget reports include usage, section costs, truncation reasons, and
  rendered node IDs.
- [x] Session L0 snapshots are frozen until refresh; warm reuse does not write
  a new snapshot.
- [x] `fibbrain_context` separately renders profile and task background under
  a budget without creating goals, episodes, or hot snapshots.

The budget applies to rendered `text`, not the complete MCP JSON response.
The current default is a deterministic estimate; its strict limit is not a
guarantee about a model's tokenizer or billed input.

### Remaining acceptance

- [ ] Provide a supported model-tokenizer integration and identify the counter
  in every report.
- [ ] Calibrate hot/direct/expanded allocation on real traces; prevent a
  standing prefix from displacing more useful task evidence.
- [ ] Keep conditions, exceptions, and source evidence usable after truncation.
- [ ] Show token savings with no loss in independently judged answer accuracy.
- [ ] Measure the new profile/task background path separately from the older
  hot/search/expand baseline.

## P4: Automatic Memory Loop

### Available now

- [x] `fibbrain_remember` combines admission and writing.
- [x] Admission skips raw dumps, empty speculation, exact duplicates, and
  high token-overlap duplicates with a negation guard.
- [x] `fibbrain_observe` persists session episodes.
- [x] `fibbrain_review_session` extracts decisions, errors, corrections,
  risks, verification, changed files, summaries, and procedures.
- [x] Review modes are `candidates`, `approve`, and `auto`; repeated review
  is idempotent. Pending memories have approve/reject tools.
- [x] The native DSH plugin observes tool results, reviews on turn end, and
  publishes mature procedures as skills.
- [x] Explicit conflict marking/resolution and pending `VERSION_OF`
  replacements are available.

Token overlap is a lexical near-duplicate heuristic, not semantic
deduplication. Conflict handling is explicit; automatic discovery of
contradictory versions has not landed. Review artifacts retain source node
IDs, but still need to carry their source files/commands into usable metadata.

### Remaining acceptance

- [ ] Copy verification and file anchors from episode events to review outputs.
- [ ] Add semantic near-duplicate detection and conservative version/conflict
  suggestions without merging opposite claims.
- [ ] Measure admission precision, review completeness, repeat-review
  idempotence, and rejection/revocation on reviewed real sessions.
- [ ] Ensure pending replacements cannot supersede a base that changed after
  they were proposed.
- [ ] Preserve caller identity and host provenance across every automatic path.

## P5: Summarization, Knowledge Governance, Security, and Observability

### Available now

- [x] Pluggable `Summarizer` / `Planner` interfaces.
- [x] Default excerpt folding, optional structured distillation, and a callable
  summarizer with model/prompt/source provenance and a fidelity check.
- [x] Knowledge promotion requires at least three distinct active supporters;
  an optional cross-provenance gate requires multiple sessions/hosts or
  verification evidence.
- [x] Refuted support reduces derived-knowledge confidence and can retire the
  derived claim.
- [x] Admission/background scanning for secrets, injection patterns, and
  invisible Unicode.
- [x] `fibbrain_tune` proposes bounded ranking-weight changes behind the
  fixed evaluation gate and records attempts.
- [x] `fibbrain_revalidation_candidates` lists old unconfirmed knowledge and
  procedures; it does not revalidate them automatically.
- [x] `fibmind_status` reports store/lifecycle/index/embedding state and
  governance counts.

The default summarizer still concatenates excerpts. A callable adapter is not
a bundled LLM service. Named outcomes and provenance make evidence auditable;
they do not establish that the supplied check is relevant to every claim.

### Remaining acceptance

- [ ] Preserve directives, exceptions, anchors, applicability, and memory kind
  in every compression path, including fallback and regeneration.
- [ ] Replace maximum-source-confidence inheritance with evidence that remains
  valid under mixed support, correction, refutation, and deletion.
- [ ] Verify replay equivalence after all lifecycle operations and redaction of
  forgotten content from source events and derived summaries.
- [ ] Provide tested backup, restore, export, and disaster-recovery commands.
- [ ] Define an optional database-encryption interface.
- [ ] Add query hit/empty rates, injected tokens, latency, admission/revocation
  rates, and hot-versus-dynamic contribution to diagnostics.
- [ ] Keep automatic threshold tuning gated; admission-pressure adjustment and
  autonomous revalidation remain unimplemented.

## P6: Reflex and Knowledge-Gap Assessment

### Reflex available now

- [x] `fibbrain_verify` / `fibmind-verify` execute commands and record
  `confirmed`, `refuted`, or `inconclusive`.
- [x] Target-file matching applies to both successful and failed checks;
  environment failures, zero tests, and missing output assertions are handled.
- [x] `fibbrain_reflex` can return a high-confidence answer or mature
  procedure, with opt-in step execution and fallback on failure.
- [x] Preconditions check conflicts, recorded file hashes, optional file
  existence, and model-generation metadata.
- [x] `fibmind_revise(protect_verified=True)` stages a pending replacement;
  `expected_version` rejects stale edits.

### Remaining reflex acceptance

- [ ] Execute `Procedure.verify` and apply verification diagnosis before
  confirming procedure success, including zero steps and inconclusive output.
- [ ] Pass an explicit caller workspace through MCP to file checks and command
  execution rather than relying on the server's current directory.
- [ ] Handle missing commands, timeouts, and other execution failures through a
  consistent circuit breaker.
- [ ] Require measured relevance as well as confidence before bypassing the
  model, and test false direct answers.
- [ ] Verify protected replacement behavior under concurrent edits and replay.

### Metacognition: design only

[The metacognition plan](docs/plans/06-metacognition.md) proposes
`fibbrain_assess`, evidence-based `known` / `uncertain` / `contradicted` /
`unknown` classifications, and routing to memory, code, verification, the
user, or external search. The MCP tool and assessment evaluation are not
implemented.

- [ ] Recheck G0 prerequisites against today's code: recall hits already expose
  confidence/status; review anchors and caller workspace binding remain open.
- [ ] Implement deterministic classification and a pluggable aspect-decomposition
  interface that cannot grant trust.
- [ ] Add a separate gap dataset and gate false-known rate before integrating
  assessment into planning, coordination, or recall.
- [ ] Persist unresolved questions and measure whether later evidence fills
  gaps or refutes a previously trusted assessment.

## P7: Cross-Application User Context

### Available now

- [x] `fibbrain_context` returns bounded background, evidence labels,
  rendered-memory provenance, `has_task_context`, and reusable `state`.
- [x] Standing user preferences can cross projects; workspace/project facts
  remain at their explicit level. Application names are host provenance.
- [x] Context excludes inactive, pending, conflicted, unsafe, and internal
  planning records without writing goals, snapshots, episodes, or evidence.
- [x] `fibmind-config` generates Codex/Claude Code/AGY configuration and
  instructions; `fibmind-connect` previews registration and backs up settings
  before applying.
- [x] The generator defaults to `memory`; direct server startup defaults to
  `full`. This repository uses `full` for its AGENTS workflow.
- [x] Independent stdio MCP-client tests cover shared persistence, visibility,
  provenance, and read-only context.

### Remaining acceptance

- [ ] Make native Codex, Claude Code, and AGY checks repeatable and record their
  results separately from protocol integration tests.
- [ ] Cover profile selection under many high-confidence project preferences,
  compression, changing applicability, and very small budgets.
- [ ] Exercise existing-label overrides, canonical paths, worktrees, missing
  native conversation IDs, and partial registration recovery.
- [ ] Document and verify daemon connections independently from the generated
  stdio registrations.

Use the host's actual conversation ID; omit it and avoid session-scope actions
when it is unavailable. Do not invent IDs or bypass host approval settings.
See [the integration guide](docs/cross-application-memory.md) and
[the cross-application plan](docs/plans/07-cross-app-memory.md).

## Release Milestones

| Milestone | Required evidence before release |
| --- | --- |
| v0.2 — Retrieval MVP | P0–P3 gates on reviewed traces; owner/project isolation and replay checks; reliable candidate fallback; model-token budgets; demonstrated retrieval benefit without an accuracy regression. |
| v0.3 — Automatic Memory | Quantified admission/review quality; idempotent capture; approval/revocation; conservative conflict/version handling; reliable verification and protected revisions. |
| v1.0 — Governed Memory | Traceable and regenerable summaries; sound evidence invalidation; backup/restore/export and diagnostics; repeatable native-host acceptance; benefits on long-running real projects. |

Metacognition, if exposed in a release, additionally needs its own false-known
and routing gates. Implemented slices can be useful before a milestone passes;
their presence alone does not advance the version.

## Deferred

Until correctness and real-workload value are demonstrated:

- More Fibonacci layers or parameter tuning.
- More relation types or automatic generation of large edge sets.
- A complex visual editor.
- Remote distributed graph storage, cross-device synchronization, or
  multi-tenant SaaS.
- Autonomous background consolidation or execution beyond the measured gates.

Detailed design background remains in
[the plan index](docs/plans/README.md),
[the coding-agent memory specification](docs/plans/05-coding-agent-memory-spec.md),
[metacognition](docs/plans/06-metacognition.md), and
[cross-application memory](docs/plans/07-cross-app-memory.md).
Some plans describe earlier snapshots; the implementation and current
acceptance checklist above determine what is available today.
