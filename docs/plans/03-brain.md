# FibBrain Plan

The identity model is in [`02-identity.md`](02-identity.md) and has landed. This document is the Brain protocol: a Cognitive Runtime grown on top of the FibMind store.

No rewrite of the agent loop, no plugin manager, no hookup to DeepSeek-Harness Cordis.

## Positioning

```text
    FibBrain  =  Memory + Planning + Decision + Coordination + Reflection
    FibMind   =  event log, identity isolation, retrieval, governance
    Harness   =  Loop / Session / Tool / Event
```

| Action | Meaning | Persisted? |
| --- | --- | --- |
| `plan` | Write the objective as a Goal and give deterministic steps | Yes, `category=goal` |
| `observe` | What happened in this turn | No, in-process episode |
| `recall` | Recall by Goal + State | No |
| `coordinate` | Which capabilities to involve next, and `advise` on each suggested action | The Goal may be marked blocked |
| `admit` / `remember` | Is this worth writing into long-term memory? | Only on `write` |
| `advise` | Should this action run? | No |
| `reflect` | Did a memory turn out right? | Yes, through `record_outcome` |
| `complete_goal` | The goal is done | Yes, the node stays recallable |

## Write gate

`remember` runs `admit` and then `append` in one transaction. It skips:

- candidates meant to be written as `knowledge` (those must go through `promote_to_knowledge`)
- session scratch without a `session_id`
- raw text that is too long or looks like a log / stack trace
- very short speculation with no conclusion
- exact duplicates within the same owner / workspace / project

`fibmind_append` can still bypass the gate; it is kept for debugging and evaluation. Agents use `fibbrain_remember` by default. Goals are written by `plan` and do not go through admit.

## Planning

`plan` writes the objective as a personal `goal` memory (tag `fibbrain-goal`). Within the same identity, the same objective reuses the existing non-complete Goal. The steps are deterministic and call no model:

1. Recall related history
2. Related memories that were refuted → `Avoid: …`
3. Related memories that were confirmed → `Reuse: …`
4. Work capabilities inferred from the objective (code / test / docs / search / review)
5. Verify
6. Reflect and remember

## Advise and coordinate

`advise` searches for the action name within the current identity. If a matching **refuted** memory exists it returns `reject`; otherwise `allow`, with related active memories attached as notes. It does no rewriting and does not build its own Harness Goal service.

`coordinate` runs `plan` first, then lists capabilities in the order Memory → work capabilities → Memory, and runs `advise` on each suggested action. Any reject marks the Goal `blocked`. Coordination only recommends; it does not execute.

## Acceptance

- After `remember` writes, `recall` with the same State sees it
- Duplicates / logs / short speculation are skipped and not persisted
- `observe` shows up only in the episode, never in long-term memory
- `recall` / `plan` do not leak data across projects
- For an action that was refuted, `advise` returns reject and `coordinate` goes blocked
- `plan` is idempotent for the same objective + same identity
- `reflect` can resolve the target memory by `node_id` or by goal
- The existing `fibmind_*` MCP tools remain; new `fibbrain_*` tools cover the five capabilities
- `pytest` all green, no regression on the P0 evaluation
- `examples/mcp_smoke.py` runs plan → coordinate → remember → recall → advise → reflect → complete end to end

## Next

```text
Brain v1 (this slice: Memory + Planning + Decision + Coordination + Reflection)
    ↓
DSH Cordis plugin: pre-step → recall, turn/end → remember / reflect
    ↓
P2 hybrid retrieval (swap the engine, not the Brain interface)
```
