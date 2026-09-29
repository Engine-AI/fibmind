# FibBrain Evolution Plan

The previous slice, Brain v1, is in [`03-brain.md`](03-brain.md) and has landed (S0 committed). The vision for the whole phase is in [`ROADMAP.md`](../../ROADMAP.md).

This document answers one question: the target is a fib-brain that "has memory, has knowledge, renders tools / skills per task, evolves on its own, and mounts on DeepSeek-Harness (DSH)". In what order do we build it?

No rewrite in TypeScript. The Python core plus the MCP protocol is the brain; the DSH side gets only a thin shell.

## Current state (2026-09)

In place: the event log as source of truth, separate familiarity / confidence, unified visibility over scope + status + five-level identity, the admit gate, seven actions (plan / recall / coordinate / advise / reflect / remember / observe), verification on two hosts (Claude Code and Codex), and the P0 evaluation (15 cases, R@K 1.0, 0 forbidden hits, 0 stale pollution).

Gaps, ordered by how much they block:

| # | Gap | Goal it blocks |
| --- | --- | --- |
| 1 | No procedural memory; nothing can be rendered as a skill / tool | Per-task rendering |
| 2 | `observe` episodes live in an in-process deque and are lost on restart; no session review | Input for self-evolution |
| 3 | Retrieval is a full keyword scan | Recall quality and latency as the store grows |
| 4 | Character truncation, no token budget, no L0 hot memory | Per-step injection in DSH |
| 5 | reflect results only reorder single memories; nothing flows back into planning / gates / weights | Self-evolution |
| 6 | `_summarize` concatenates; plan is a keyword-matching heuristic | Having knowledge |
| 7 | stdio MCP only, no native DSH plugin | Automatic recall on every step, action interception |

## Principles

- **Wire up real trajectories first, then optimize algorithms.** ROADMAP's P0 still has no real project data; zero-code DSH integration is the cheapest data source.
- **Freeze the Brain interface; keep the implementation swappable.** The names and parameters of the seven `fibbrain_*` actions do not change; the retrieval engine, summarizer, and planner are all swapped behind the interface.
- **Every slice must pass the evaluation gate.** `pytest` all green and no `evals.runner` regression is the merge condition, especially for automatic tuning.
- **Evolution = outcome statistics changing the next behavior.** A feature with no `reflect` data flowing back is not evolution, just a feature.

## Slices

```text
S0 Close-out and contract  →  S1 Zero-code DSH integration
                                    ↓
S2 Persistent episode + Session Review (first half of P4)
                                    ↓
S3 Procedural memory + skill / tool rendering
                                    ↓
S4 Hybrid retrieval (P2)   →  S5 Token budget + L0 hot memory (P3)
                                    ↓
S6 Native DSH Cordis plugin
                                    ↓
S7 Evolution loop + pluggable LLM distillation / planning
```

S4 can run in parallel with S2 / S3; it does not change the Brain interface.

---

### S0 Close-out and contract

**Do**

- Commit the current working tree (Brain v1's `brain.py` / `planning.py` / `admission.py` / tests / docs).
- Add `log_version` to the event log format, written into `MemoryEvent.to_dict()` and SQLite `meta`.
- Write a `docs/contract.md` for the MCP tool schema: parameters, return fields, and stability promises for the seven `fibbrain_*` tools.
- Pin `data/demo-brain.json` as a conformance fixture: after any implementation runs `rebuild_from_log` on it, its retrieval results for a fixed query set must match entry for entry. This is the acceptance bar for any future port to another language.

**Don't**: split `graph.py` into a directory structure.

**Acceptance**: `tests/test_contract.py` passes; `git status` is clean.

---

### S1 Zero-code DSH integration

**Do**

- Add `examples/dsh/fibbrain.cordis.yml`: a single `@deepseek-ai/dsh-mcp-client` stdio row pointing at `fibmind.mcp_server` in this repository's venv.
- Add `examples/dsh/README.md`: installing dsh, enabling it with `--patch`, and the three-step check from dsh's official memory guide (session A writes, session B recalls, session B uses it).
- Copy the calling conventions from `CLAUDE.md` / `AGENTS.md` into a dsh model instruction snippet.
- Export session logs (JSONL) from dsh runs as P0 dataset candidates: add `evals/import_dsh_session.py`, which extracts `fibbrain_*` calls and the final answer from `tool/call` / `tool/result`.

**Don't**: write any TS code; rename MCP tools.

**Acceptance**

- The three-step check passes on this machine, and `.fibmind/memory.db` contains nodes whose `session_id` is the dsh session id.
- At least one real dsh trajectory lands in `evals/datasets/real_dsh.json`, and `evals.runner` can run it.

**Note**: dsh bridges only Tools; resources / prompts will not appear. The model decides when to call, which is what S6 has to fix.

---

### S2 Persistent episode + Session Review

Corresponds to the first half of ROADMAP P4.

**Do**

- Persist `observe`: write nodes with `scope=session`, `category=episode`, reusing the existing session visibility (visible only to the same session). The in-process deque becomes a cache.
- Add `fibbrain_review_session(session_id, mode)`:
  - **deterministic extraction** from that session's episode nodes and goal nodes: objective, changed files (from `files` in the `observe` payload), tests (observes with `kind=test`), errors, user corrections, unresolved risks;
  - every candidate goes through `admit`;
  - `mode` ∈ `auto` / `approve` / `candidates`. `approve` mode writes nodes with `status=pending`; add `fibbrain_approve` / `fibbrain_reject`.
- Idempotent: skip a candidate whose fingerprint already exists; reviewing the same session again produces no duplicate nodes.
- When the session ends, mark its episode nodes `stale`, keeping provenance.

**Don't**: use a model for extraction (that comes in S7).

**Acceptance**

- `observe` → process restart → `recall` in the same session still sees the episode.
- Running `review_session` twice on the same session leaves the node count unchanged.
- In `approve` mode, unapproved candidates do not appear in `recall`.
- New P0 case: memories extracted by review are Recall@K hits for queries in the next session.

---

### S3 Procedural memory + rendering

This is entirely missing today, and it is the only path to "render tools / skills per task".

**Data model**

- `MemoryNode` gains `memory_kind` (already listed in ROADMAP P1): `preference / decision / error / requirement / summary / knowledge / procedure`, inferred from the category by default, `None` for old nodes. SQLite v4 → v5.
- A `procedure` node's content is a structured document (the same approach as goals):

```json
{
  "kind": "fibbrain_procedure",
  "trigger": "run fibmind regression checks",
  "when": ["pytest", "eval", "regression"],
  "steps": [".venv/bin/pytest -q", ".venv/bin/python -m evals.runner"],
  "tools": ["shell"],
  "verify": "142 passed and fibmind_current does not regress",
  "outcomes": {"confirmed": 0, "refuted": 0}
}
```

**Actions**

- `fibbrain_render(goal, state, format)`: search `procedure` nodes by goal (through the same `is_visible`) and render them as:
  - `skill`: SKILL.md-style markdown (name, description, steps, verification);
  - `tool`: a JSON Schema tool definition (name / description / parameters, with steps as the execute hint).
- `coordinate` changes to: look for a matching procedure first; if one exists, build the capability list from its `steps` / `tools`, and only otherwise fall back to today's six fixed CAPABILITIES.
- When `reflect` targets a procedure node, it also updates the `outcomes` counts.
- Generate procedure candidates automatically from S2's review: a "goal → a series of tool calls → verification passed" sequence within one session is extracted into a single `procedure`, with initial confidence 0.

**Evolution rules (deterministic)**

- `confirmed ≥ 3` from ≥ 2 distinct sessions → eligible for `promote_to_knowledge`, becoming a skill shared within the workspace.
- `refuted ≥ 2` and `confirmed = 0` → automatic `mark_stale`.
- When a new procedure with the same trigger appears, the old one gets a `VERSION_OF` edge.

**Don't**: execute procedures on the Python side. Execution is always the Harness's job.

**Acceptance**

- The output of `render(format=skill)` can be dropped straight into `.claude/skills/<name>/SKILL.md` and is listed by Claude Code.
- The output of `render(format=tool)` passes JSON Schema validation.
- Procedures from another project are never rendered (reusing the identity tests).
- After three confirmations, the source of `coordinate`'s capability list becomes that procedure.

---

### S4 Hybrid retrieval (P2)

Carried out per ROADMAP P2; this section covers only the coupling points with this plan.

- SQLite FTS5 + BM25; Chinese text keeps the existing bigram tokenization strategy; writes / revisions / forgets update the index incrementally.
- An `EmbeddingProvider` interface, off by default; when off, pure FTS5. A provider failure falls back to FTS5 and must never make `recall` error out.
- RRF fusion, weighted by `confidence`; graph expansion only supplements recall.
- Add `fibmind_explain_recall`.
- Move `advise` from title keyword matching to hybrid retrieval, and match on the action arguments too (e.g. two `rm` calls with different paths are judged separately).

**Decision gate**: on the P0 datasets (including the real trajectories imported in S1), the new engine's Recall@K, stale, and forbidden must be no worse than the current keyword baseline; P95 < 50 ms at 10,000 nodes. If it falls short, it does not merge.

---

### S5 Token budget + L0 hot memory (P3)

- Upgrade `max_chars` to `budget_tokens`, with a pluggable tokenizer (the default is still the estimator; calibrate once the DeepSeek / Claude tokenizer is wired in).
- L0 hot-memory snapshot: generated at session start, frozen within the session, updated by S2's review when the session ends. Budget 800–1200 tokens; when full, merge or reject.
- The Context Planner returns `used_tokens` / `budget_tokens` / per-memory cost / truncation reasons.

**Acceptance**: `recall` never exceeds the budget; hot memory is stable within a session; the evaluation compares tokens rather than characters, and the 30% savings threshold is calibrated on real trajectories.

---

### S6 Native DSH Cordis plugin

A thin shell, TypeScript, roughly 300–500 lines, in `integrations/dsh/`. Internally it calls the Python side through dsh's bundled mcp-client and does not manage its own subprocess.

| dsh event / service | FibBrain action |
| --- | --- |
| `agent/session-start` | Inject L0 hot memory; `plan` |
| `agent/pre-step` (waterfall) | `recall`; append the pack to this step's message batch |
| `tools/pre-execute` (waterfall) | `advise`; `reject` → `deny`, with the reason passed to the model |
| `tools/result` | `observe` |
| `agent/turn-stopping` | `review_session(mode=approve)` + `reflect` |
| `ctx.skills.registerProvider()` | Serve `render(format=skill)` results as the skill directory |
| `ctx.tools.register()` | Register `render(format=tool)` results on demand |

**Identity mapping**: dsh has only an anonymous install id. The plugin maps `cwd` from the session meta → `workspace_id` (git root) and `project_id` (configurable), and `sessionId` → `session_id`; fork lineage goes into metadata.

**Split with dsh goals**: dsh `ctx.goals` owns the current goal of this session; FibBrain goals own cross-session history and Avoid / Reuse evidence. The plugin writes the dsh goal id into the FibBrain goal's metadata; there is no two-way sync.

**Acceptance**

- With no model instruction at all, the request for a new session's first step already contains the recall pack (visible in the dsh Trajectory view).
- An action that was `reflect(refuted)` is stopped by `pre-execute` in dsh, with a reason.
- S3 carry-through test: a procedure confirmed three times shows up in the `skill` directory of the next dsh session.

---

### S7 Evolution loop + pluggable LLM

The first six slices get data flowing; this one makes data change behavior.

**Feedback points (all deterministic, all behind the evaluation gate)**

- `ranking` weights: from `reflect` outcomes, measure how each signal (coverage / title / confidence / recency) correlates with confirmed, and allow adjustments within configured bounds; every adjustment runs `evals.runner` and is rolled back on regression.
- `admit` thresholds: candidate categories with a high rate of `reject` / `mark_stale` get a stricter threshold automatically.
- `plan` heuristics: S3's procedure success rates replace the `_HINTS` keyword table as the preferred source.
- Knowledge invalidation propagation: a supporting node is refuted → derived knowledge's confidence is scaled down proportionally; all supporters invalid → the knowledge goes `stale`.
- A periodic revalidation job for knowledge that has gone unconfirmed for a long time (output is candidates; nothing changes automatically).

**Pluggable LLM**

- A `Summarizer` interface replaces `_summarize`; a `Planner` interface wraps `synthesize_plan`. The default implementations stay the current deterministic versions; the LLM versions record the model, prompt version, and input sources, and can be regenerated from events.
- Distillation output still has to pass the three-supporter gate of `promote_to_knowledge`.

**Observability**: `fibmind_status`: store size, index state, pending approvals, invalidated knowledge count, recall hits / empty recalls, auto-write acceptance rate, latest evaluation result.

**Safety**: scan for keys / tokens / private keys, prompt-injection patterns, and invisible Unicode before writing. This is a prerequisite for taking real DSH traffic, and can in practice be pulled ahead of S6.

**Acceptance**

- A deliberately injected wrong weight is corrected automatically after N reflects, and the evaluation gate never lets a regression through along the way.
- After a supporting node is refuted, `recall` no longer returns the knowledge derived from it.
- Summary nodes produced by replaying the log with an LLM Summarizer sit in the same layer as the deterministic version, and `folded_into` provenance is unchanged.

---

## Milestones

| Version | Scope | Release criteria |
| --- | --- | --- |
| v0.2 Brain on DSH | S0–S3 | dsh usable with zero code; session review idempotent; renders at least one skill verified by a real task |
| v0.3 Retrieval MVP | S4–S5 | Hybrid retrieval beats the keyword baseline; token budget strictly enforced; savings calibrated on real trajectories |
| v0.4 Native Brain | S6 | Native dsh plugin works end to end for per-step recall, interception, and skill rendering |
| v1.0 Evolving Brain | S7 | At least one automatic tuning is let through by the evaluation gate and kept; knowledge invalidation propagation can be demonstrated |

## Execution log (2026-09)

| Slice | Commit | Result |
| --- | --- | --- |
| S0 | 75755e3 | `log_version`, `docs/contract.md`, conformance fixture |
| S1 | 943a30f | Zero-code dsh overlay, session log importer; real dsh not run on this machine (no API key) |
| S2 | 7f46511 | Persistent episode, `review_session` idempotent across all three modes, pending approval |
| S3 | a93be4d | `memory_kind` / procedure / `render`, coordinate prefers procedures |
| S4 | 2b6c03d | Inverted index + BM25 + optional vectors + RRF; P95 48 ms at 10,000 nodes; explain |
| S5 | 732f1c7 | Token-budgeted sectioned rendering, frozen L0 hot memory |
| S6 | 5fbfa22 | Native Cordis plugin (JS, 11 unit tests, real Python end to end) |
| S7 | this change | Safety scan, knowledge invalidation propagation, pluggable distillation / planning, gated tuning, revalidation candidates |

Both v1.0 acceptance criteria have tests: after a wrong weight is injected, `tune()` corrects it step by step
through the real evaluation gate (`test_a_wrong_weight_is_corrected_by_evidence_with_the_real_gate`); once all
supporters are refuted, the derived knowledge drops out of recall (`test_all_supporters_refuted_retires_the_claim_from_recall`).

## Next

- Calibrate the token savings threshold and the tuning step size on real dsh sessions (via the S1 importer) instead of the small fixed datasets.
- Persist the vector cache; consider FTS5 only if the in-memory index stops being enough.
- Automatic tightening of the admit threshold: today it only reports `admission_pressure`; whether to let it change behavior depends on the real rejection rate.
- Automatic detection of semantic near-duplicates and conflicting versions (`VERSION_OF`).
