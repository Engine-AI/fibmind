# FibBrain Metacognition Plan: Knowing What It Does Not Know

Status: **draft**, not started. Builds on [`03-brain.md`](03-brain.md) (the
Brain protocol), [`04-evolve.md`](04-evolve.md) (episodes, procedures, tuning
gate), and the System 1 reflex in `src/fibmind/reflex.py`.

This slice answers one question before any task starts:

> For this goal, what do I know and trust, what do I know but should not yet
> trust, what do I know to be wrong, and what do I not know at all — and for
> each gap, where should I go to close it?

Today `recall` answers a different question — "what is related to this goal?"
— and cannot say "nothing I have is good enough". Every later direction (an
entity model, the LLM as an advisor whose answers are not evidence, background
consolidation) needs this distinction first, so it goes first.

## Diagnosis (measured 2026-10-19)

**1. recall has no "I don't know".** Two recalls made while preparing this plan,
against the real store (`.fibmind/memory.db`, 101 nodes), returned 3–4 hits
each. None of them was about the goal. Leaving out an old goal node, which
`assess` will exclude, their term coverage was 0.07–0.08 (1–2 of 14 / 25 query
terms; 0.03–0.04 IDF-weighted). The pack rendered them as ordinary context with
nothing marking them as weak.

The fixed evaluation does not see this. Its four no-relevant-memory cases all
have zero term overlap, so `no_relevant_accuracy = 1.0` only covers queries
that share no word with the store at all. On the 29 positive cases, the
relevant hit's IDF-weighted coverage ranges from 0.17 to 1.0.

**2. plan never emits "Reuse".** `plan()` passes recall hits to the planner
(`brain.py:486`), and `synthesize_plan` reads `confidence` / `status` from each
hit (`planning.py:153`). `ContextHit.to_dict()` (`context.py:52`) carries
neither. Refuted hits come from a second `search` call that does carry status,
so "Avoid" works while "Reuse" cannot fire. Reproduced on a temporary store: a
memory confirmed three times (confidence 0.75) produced no Reuse step.

**3. Few memories can be checked.** Of the 48 content memories in the real store
(excluding roots, episodes, goals, and hot snapshots), 37 have never been
confirmed, 10 sit at 0.25, and 1 is at 0.75. **None** carries `files`,
`command`, or `file_hashes` metadata. `review_session` writes only
`source_node_ids` (`brain.py:1033`); the files and commands recorded on the
source episode events are not copied over. `extract_node_anchors` and
`check_node_preconditions` therefore have nothing to work with on reviewed
memories.

**4. MCP precondition checks resolve against the server's cwd.**
`fibbrain_reflex` does not pass `cwd`, so `check_node_preconditions` hashes
files relative to wherever the server process started. In shared daemon mode
this is not the host's workspace.

Items 2–4 are bugs in their own right and are fixed first (G0) because the
assessment depends on them.

## Principles

- **Evidence decides trust; nothing else does.** The classification of a
  memory uses only the existing evidence signals: `status`, `confidence`,
  outcome counts, last-confirmation time, conflict flag, and preconditions.
  Familiarity has no role, as everywhere else.
- **Relevance and trust are separate axes.** A highly relevant unchecked memory
  is "uncertain". A trusted memory that only shares the word "recall" with the
  goal is not "known" — it is not about the goal at all.
- **An LLM may decompose a goal, never vouch for an answer.** The pluggable
  assessor (below) may use a model to split a goal into sub-questions. Whether
  a sub-question is answered is still decided by evidence. This is the same
  rule as "being recalled is not being confirmed", extended to the model.
- **Recommend, don't execute.** Like `coordinate`, `assess` names the next
  source to consult; the harness acts.
- **Prefer the cheapest objective source.** Order of preference when closing a
  gap: trusted memory → the code / a recorded check → the user (ground truth
  for intent) → external search → the model's own knowledge (unverified).
- **Evolution needs a returning signal.** Every assessment is logged, and the
  session review measures whether it was right. Without that, this is a report,
  not metacognition.

## Scope

| In | Out |
| --- | --- |
| `fibbrain_assess`: per-goal classification + gap routing | An entity / world model (next slice, direction 1) |
| Deterministic assessor behind a `GapAssessor` protocol | Shipping an LLM assessor (protocol only) |
| `plan` turns the assessment into Reuse / Verify / Investigate / Avoid steps | Executing any route automatically |
| G0 fixes: hit fields, review metadata, workspace root | Changing recall ranking or the retrieval floor |
| Gap evaluation set + false-known metric | Auto-tuning the thresholds (reported only) |
| Open questions persisted and escalated to the user | Background consolidation (direction 5) |

## Model

### Four states

| State | Meaning | Condition (deterministic v1) |
| --- | --- | --- |
| `known` | relevant and trustworthy | relevant, `active`, confidence ≥ `TRUST_FLOOR`, not conflicted, preconditions hold, confirmed within `FRESHNESS_DAYS` |
| `uncertain` | relevant, not yet trustworthy | relevant, `active`, but any `known` condition fails; the failing conditions are listed as `reasons` |
| `contradicted` | known to be wrong | relevant, `refuted` |
| `unknown` | nothing usable | a goal aspect that no relevant memory covers |

"Relevant" is per hit: IDF-weighted term coverage ≥ `RELEVANCE_FLOOR`.

```text
wcov(hit) = Σ idf(t) for t in hit.matched_terms  /  Σ idf(t) for t in goal terms
```

IDF comes from the existing `LexicalIndex.idf`. A goal term the store has never
seen gets the highest IDF, which is the intended effect: a word the brain has
never encountered is the strongest gap signal. When a semantic embedding
provider is enabled, a hit is also relevant if its vector similarity clears the
provider's `min_similarity`. The offline hashing provider is not semantic and
does not count.

Hits below the floor are returned under `weak` so the host can see what was
excluded and why. They never count as knowing.

### Uncertainty reasons

Each comes from a signal that already exists:

| reason | source |
| --- | --- |
| `unchecked` | confidence == 0 |
| `low_confidence` | 0 < confidence < `TRUST_FLOOR` |
| `confirmation_stale` | `last_confirmation` older than `FRESHNESS_DAYS` (same 90-day default as `revalidation_candidates`) |
| `conflicted` | `metadata.conflicted` |
| `file_hash_mismatch: <path>` / `missing_target_file: <path>` | `check_node_preconditions` |
| `model_heuristic_mismatch` | `grounding == model_heuristic` recorded for another `model_id` |

### Unknown aspects

v1 approximates "aspects of the goal" with goal terms: the terms that no
relevant (`known` / `uncertain` / `contradicted`) hit matched. They are grouped
back into surface units so CJK bigrams do not flood the list: a word run
reports as itself, and a CJK run reports as the whole run when fewer than half
of its bigrams are covered. Units are sorted by IDF and capped at 5.

This is a crude proxy, and the reason the assessor is pluggable. An LLM
assessor can decompose "fix the concurrent refresh race in auth" into
sub-questions ("how is refresh triggered?", "what locks exist?") far better
than term coverage can. Per the principles, the decomposition is all it
provides.

### Goal verdict

```text
unknown   no relevant hit at all
known     IDF-weighted share of goal terms covered by `known` hits ≥ KNOWN_COVERAGE,
          and no `conflicted` / `contradicted` hit on the same terms
partial   everything else
```

### Starting thresholds

| constant | value | why |
| --- | --- | --- |
| `RELEVANCE_FLOOR` | 0.12 | below every positive case today (min 0.17), above both real irrelevant recalls (0.03–0.04) |
| `TRUST_FLOOR` | 0.5 | two net confirmations at `CONFIDENCE_STEP = 0.25`; one lucky pass is not enough |
| `KNOWN_COVERAGE` | 0.6 | most of the goal, weighted by how distinctive each term is |
| `FRESHNESS_DAYS` | 90 | matches `revalidation_candidates` |

All four are calibrated in G3 on the gap set. The current datasets are small,
so these are starting points, not claims. Later they can join the bounded,
gate-checked settings that `fibbrain_tune` may move; in this slice they are
module constants and calibration outcomes are reported, not applied.

**Expected effect on today's store:** with `TRUST_FLOOR = 0.5`, almost every
relevant memory classifies as `uncertain`, because almost nothing has been
confirmed twice. That is an accurate picture rather than a defect in the
assessment, but a list where everything is uncertain is noise. The mitigations
are part of the design: only uncertain items with a check available get an
active route (`verify` / `code`); the rest are marked `caution` and capped; and
G0 makes reviewed memories carry their files and commands, so more of them
become checkable.

## Routing

Rules are evaluated top to bottom; the first match wins. Like `_HINTS` in
`planning.py`, the table is deliberately small and extensible.

| situation | route | target | why |
| --- | --- | --- | --- |
| `contradicted` | `avoid` | the refuted memory | already known to be wrong |
| uncertain, has a recorded command | `verify` | `fibbrain_verify(command, node_id)` | objective and cheap |
| uncertain, file drift or has files | `code` | re-read the files | the code is ground truth |
| uncertain, `conflicted` | `resolve` | check, then `fibbrain_resolve_conflict` | two claims cannot both stay active |
| uncertain, kind `preference` / `requirement`, no handle | `ask` | the user | only the user can confirm intent |
| uncertain, no handle | `caution` | use, flagged | nothing cheaper can check it |
| unknown, unit looks like a path / symbol | `code` | read it | reuses `is_lexical_shortcut_query` per unit |
| unknown, matches an open question (G4) | `ask` | the user | asked before, still unresolved |
| unknown, goal asks about convention / preference (`should`, `prefer`, `convention`, `应该`, `偏好`, `规范`) | `ask` | the user | intent is not in the code |
| unknown, goal is time-sensitive (`latest`, `version`, `release`, `deprecated`, `最新`, `版本`, a year) | `search` | web search | the model's knowledge is dated |
| unknown, otherwise | `model` | the host's own knowledge | conclusions are written as unproven |

`route = model` is where direction 4 (the LLM as advisor) starts: whatever the
host concludes from its own knowledge is remembered with
`metadata.grounding = "model"` and confidence 0 until something outside the
model confirms it.

## API

### `FibBrain.assess`

```python
def assess(
    self,
    goal: str,
    state: BrainState | None = None,
    top_k: int = 8,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]: ...
```

It reuses the same retrieval as recall (same visibility, same excluded
categories: episode / hot / goal), adds the refuted search that `plan` already
runs, and reads outcome counts and last confirmations from the store. It does
not render a token-budgeted pack. Use `recall` for that.

### Return shape

```json
{
  "goal": "fix login 401 after token refresh",
  "verdict": "partial",
  "coverage": {"known": 0.41, "any": 0.78},
  "known": [
    {"node_id": "node_…", "title": "Refresh tokens before retrying login",
     "confidence": 0.75, "relevance": 0.64,
     "evidence": "confirmed 3x in 2 sessions via pytest tests/test_auth.py; last 2026-10-10"}
  ],
  "uncertain": [
    {"node_id": "node_…", "title": "401 only when refresh token expired",
     "confidence": 0.0, "relevance": 0.52, "reasons": ["unchecked"],
     "route": "verify", "target": "pytest tests/test_auth.py -k refresh"}
  ],
  "contradicted": [
    {"node_id": "node_…", "title": "Retry login on 401 without refreshing",
     "route": "avoid", "reason": "refuted by pytest"}
  ],
  "unknown": [
    {"aspect": "并发刷新", "terms": ["并发", "刷新"], "route": "code",
     "why": "no relevant memory; goal names src/auth/session.py"}
  ],
  "weak": [
    {"node_id": "node_…", "title": "…", "relevance": 0.04, "matched_terms": ["recall"]}
  ],
  "next": [
    {"route": "avoid", "target": "node_…"},
    {"route": "verify", "target": "pytest tests/test_auth.py -k refresh"},
    {"route": "code", "target": "src/auth/session.py"}
  ],
  "assessor": "deterministic-v1",
  "thresholds": {"relevance_floor": 0.12, "trust_floor": 0.5, "known_coverage": 0.6, "freshness_days": 90},
  "state": {"owner": "…", "workspace_id": "…", "project_id": "…", "session_id": "…", "task_id": "…"}
}
```

With a `session_id`, the call also records an `assess` episode event whose
payload lists the classified node ids and unknown units. G4 reads it back.

### `GapAssessor` protocol

Sits next to `Summarizer` and `Planner` in `distill.py`:

```python
class GapAssessor(Protocol):
    name: str

    def aspects(self, goal: str, query_terms: set[str]) -> list[Aspect]: ...
```

Only aspect extraction is pluggable. Classification and routing stay in
`assessment.py` as fixed, evidence-only code, so swapping in an LLM can change
*which questions are asked* but never *which answers are trusted*.

### Relation to reflex

`reflex` fast path A (direct answer) is the `known` case decided ad hoc: the
first visible hit with confidence ≥ 0.75. Once `assess` exists, reflex should
take its fast path only when `assess(goal).verdict == "known"`. This is listed
as a follow-up rather than done here, so reflex behavior does not change in
this slice.

## Slices

```text
G0 prerequisite fixes  →  G1 assess core  →  G3 gap evaluation (calibrate)
                                              ↓
                                G2 plan / coordinate / recall integration
                                              ↓
                                G4 open questions + calibration loop
```

G3 comes before G2 on purpose ("prove the value first"): `plan` should not
start depending on thresholds that have not been calibrated.

### G0 Prerequisite fixes

**Do**

- `ContextHit.to_dict()` adds `confidence`, `status`, `memory_kind`. This is
  additive, which the contract allows. Regression test: a memory confirmed
  twice produces a `Reuse:` plan step.
- `review_session` copies anchors into the metadata of the memories it writes:
  `files` from the source episode events' `payload.files`, and `commands` from
  their `payload.command`. `ReviewCandidate` gains a `metadata` field;
  `extract_candidates` fills it from the digest it already builds.
- `fibbrain_reflex` (and `fibbrain_assess`) accept an optional
  `workspace_root`, falling back to the server cwd, and pass it as `cwd` to the
  precondition checks.

**Acceptance**: the new tests pass; a reviewed memory from a session that
observed `payload.files` has non-empty `extract_node_anchors`.

### G1 assess core

**Do**

- New `src/fibmind/assessment.py`: `Assessment` dataclasses, `classify()`,
  `route()`, the thresholds, and `DeterministicAssessor`. Pure functions over
  (goal, candidates, refuted hits, outcome lookup, precondition checker), in
  the same style as `visibility.py`.
- `FibBrain.assess()`; MCP tool `fibbrain_assess`; `docs/contract.md` row and
  guarantees; `tests/fixtures/demo-brain.expected.json` tool list.
- `tests/test_assessment.py`, one test per row of the state and routing
  tables.

**Contract guarantees added**

- `assess` never returns a node the caller cannot see (same rule as recall).
- Same store, goal, identity, and file state → same output, apart from
  timestamps.
- Classification uses only status, confidence, outcomes, last confirmation,
  conflict, and preconditions. Familiarity cannot change any field.
- A `GapAssessor` can change `unknown[].aspect` and `unknown[].terms`, but
  cannot change any memory's state.

### G3 Gap evaluation

**Do**

- `evals/gaps/knowledge_gaps.json`. It lives in its own directory because
  `evals.runner` loads every file under `evals/datasets/`, and this slice must
  not move the retrieval floor. Memory fixtures add `confirmations`,
  `refutations`, `conflicted`, and `files` / `file_hashes`. Cases add
  `expect_verdict`, `expect_known_ids`, `expect_uncertain_ids`,
  `expect_contradicted_ids`, and `expect_routes`.
- Required cases:
  1. **Partial-overlap negative**: the goal shares 1–2 generic terms with a
     trusted memory about something else, as in the two real recalls above.
     Must be `unknown`.
  2. Relevant but unchecked → `uncertain / verify` (has a command) or
     `uncertain / caution` (has none).
  3. Confirmed twice, fresh → `known`.
  4. Confirmed, then the file changed → `uncertain / code`.
  5. Refuted → `contradicted / avoid`.
  6. Conflicted pair → both `uncertain / resolve`.
  7. Half the goal covered → `partial`, with the uncovered unit listed.
  8. A Chinese goal with mixed CJK and identifier runs.
  9. A preference question with no preference memory → `unknown / ask`.
- `evals/gaps_runner.py` reporting verdict accuracy, **false-known rate** (said
  `known`, or put a node in `known`, when it should not have), false-unknown
  rate, and route accuracy.
- `evals/floor.json` gains a `gaps` section: `false_known_rate` max 0.0.

**Acceptance**

- `false_known_rate = 0`, verdict accuracy ≥ 0.9 on the gap set; thresholds
  adjusted and recorded here if they had to move.
- **Real-trace check**: on a copy of `.fibmind/memory.db`, the two goals used
  to prepare this plan return `verdict: unknown` with an empty `known` list.
  Today, recall presents their hits as context.
- `evals.runner` retrieval floor unchanged.

False-known is the primary metric because it is the harmful error: a brain
that thinks it knows does not go and look. False-unknown only costs an extra
check.

### G2 Integration

**Do**

- `plan()` calls `assess()` and passes the assessment to the planner.
  `synthesize_plan` emits, before the work step: `Reuse: <known>` (max 3),
  `Verify: <target>` for `verify` / `code` routes, `Resolve: <conflict>`,
  `Avoid: <contradicted>` (existing behavior), `Investigate: <unit> via
  <route>` for unknowns, and `Ask: <question>` for `ask`. The `Planner`
  protocol gains an optional `assessment` argument, so existing planners keep
  working.
- `CAPABILITIES` adds `ask` ("Ask the user"), and `coordinate` maps routes to
  capabilities: verify→test, code→code / review, search→search, ask→ask.
- `recall(..., include_gaps=False)`: when true, appends one `Gaps:` line
  (verdict + top unknown units) inside the token budget, as its own budget
  section. Opt-in, because the DSH plugin and other hosts that only call
  `recall` should not change behavior until G3's numbers are in. The default
  can flip later on evidence.

**Acceptance**: `plan` tests for each step type; `coordinate` reports `ask`
when the assessment routes to the user; recall's budget guarantee still holds
with `include_gaps=True`.

### G4 Open questions and the calibration loop

**Do**

- At `review_session`, an unknown unit from this session's `assess` events
  that no memory written in the session covers becomes a `pending` memory in
  category `question`. A category is used instead of a new `MemoryKind` value,
  so older readers that parse `memory_kind` do not break. Approve / reject work
  as for any review memory.
- Routing rule "matches an open question → ask" is enabled. Asking twice
  without an answer escalates to the user instead of guessing a third time.
- Calibration counts, derived from the episode:
  - `overconfident`: a node assessed `known` and refuted later in the same
    session;
  - `verified_uncertain`: assessed `uncertain` and later confirmed;
  - `gap_filled`: an unknown unit later covered by a written memory.
- `fibmind_status` reports these counts and
  `P(refuted | assessed known)`. Nothing is tuned automatically in this slice.
  If overconfidence shows up in real traces, raising `TRUST_FLOOR` becomes a
  `fibbrain_tune` proposal that has to pass the gate.

**Acceptance**: review is still idempotent with questions included;
calibration counts are reproducible from the log alone.

## Files

| file | change |
| --- | --- |
| `src/fibmind/assessment.py` | new: states, routing, thresholds, `DeterministicAssessor` |
| `src/fibmind/distill.py` | `GapAssessor` protocol |
| `src/fibmind/brain.py` | `assess()`; `plan` / `coordinate` consume it; review copies anchors; open questions |
| `src/fibmind/context.py` | hit fields (G0); optional gaps section (G2) |
| `src/fibmind/review.py` | `ReviewCandidate.metadata`; anchors from the digest |
| `src/fibmind/planning.py` | assessment-aware `synthesize_plan`; `ask` capability |
| `src/fibmind/reflex.py`, `mcp_server.py` | `workspace_root`; `fibbrain_assess` tool |
| `docs/contract.md`, `tests/fixtures/demo-brain.expected.json` | new tool and guarantees |
| `evals/gaps/`, `evals/gaps_runner.py`, `evals/floor.json` | gap evaluation |
| `tests/test_assessment.py`, `tests/test_brain.py`, `tests/test_contract.py` | tests |

## Risks

- **Everything is uncertain.** Covered above. The real fix is more
  verification, which G0's anchors and the `verify` routes push toward.
- **Term coverage is a weak proxy for aspects.** Long goals full of generic
  words dilute coverage. IDF weighting helps, and the pluggable assessor is
  the long-term answer.
- **Thresholds from a small set.** Same caveat as the ROADMAP: regression
  checks, not production claims, until real traces are labelled.
- **Hosts that only call recall get nothing** until `include_gaps` is enabled.
  The DSH plugin currently calls `fibbrain_recall`, not `plan`.
- **Cost.** Preconditions hash files only for hits that carry `file_hashes`,
  bounded by `top_k`. Everything else reuses retrieval that recall already
  runs.

## Not in this slice

- An entity / world model per module, user, or project. When it lands, it
  becomes the preferred `known` source, and `assess` is how it gets used.
- Calling an LLM or search from inside FibBrain.
- Changing recall ranking, the retrieval floor, or `reflex` behavior.
- Automatic threshold tuning.
