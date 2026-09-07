"""FibBrain MCP server: cognitive tools plus the FibMind memory store.

The ``fibbrain_*`` tools are the Brain protocol (plan / recall / coordinate /
remember / observe / advise / reflect). The older ``fibmind_*`` tools stay as
a direct store API.

Run it directly::

    python -m fibmind.mcp_server --store .fibmind/memory.json

Or register it with Claude Code::

    claude mcp add --scope project --transport stdio fibmind -- \\
        /path/to/.venv/bin/python -m fibmind.mcp_server --store .fibmind/memory.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from fibmind.brain import BrainState, FibBrain
from fibmind.embedding import provider_from_env
from fibmind.service import MemoryService

DEFAULT_STORE = ".fibmind/memory.db"


def _state(
    owner: str | None,
    workspace_id: str | None,
    project_id: str | None,
    session_id: str | None,
    task_id: str | None,
) -> BrainState:
    return BrainState(
        owner=owner,
        workspace_id=workspace_id,
        project_id=project_id,
        session_id=session_id,
        task_id=task_id,
    )


def build_server(service: MemoryService, brain: FibBrain | None = None) -> MCPServer:
    """Create an MCP server whose tools delegate to ``service`` and ``brain``."""
    brain = brain or FibBrain(service)
    mcp = MCPServer("fibmind")

    @mcp.tool()
    def fibmind_append(
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        scope: str = "personal",
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Save one memory (task result, error, requirement, plan, or code summary).

        Use this after finishing a task to record what happened. ``category``
        groups related memories into a tree (e.g. "code", "requirement",
        "decision"). ``scope`` is "personal" for observations about one owner or
        "session" for scratch notes; shared knowledge is created through
        fibmind_promote_knowledge instead, which requires supporting evidence.

        Pass the current ``workspace_id``, ``project_id``, and ``session_id`` so
        later recall can keep one working context out of another. Session-scoped
        scratch requires ``session_id``.

        A stored memory starts with zero confidence. When you later learn whether
        it was right, call fibmind_record_outcome — memories that are never
        checked stay unranked no matter how often they are read back.
        """
        return service.append(
            category,
            title,
            content,
            tags=tags,
            metadata=metadata,
            scope=scope,
            owner=owner,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
        )

    @mcp.tool()
    def fibmind_record_outcome(
        node_id: str,
        verdict: str,
        source: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Record that a memory turned out to be right or wrong.

        ``verdict`` is "confirmed" or "refuted"; ``source`` must name what
        checked it — a test command, a failing build, a user correction. This is
        the only thing that moves a memory's confidence, and confidence is the
        only trust signal that affects ranking.

        Call it whenever evidence appears: a recorded fix that made the tests
        pass, a decision that was later reverted, advice the user rejected.
        """
        return service.record_outcome(node_id, verdict, source, note=note)

    @mcp.tool()
    def fibmind_revise(
        node_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Correct a memory whose content is wrong or out of date.

        Prefer this over appending a contradicting memory. Confidence resets to
        zero, since evidence backing the old text says nothing about the new.
        """
        return service.revise(node_id, title=title, content=content, tags=tags)

    @mcp.tool()
    def fibmind_mark_stale(node_id: str, reason: str) -> dict[str, Any]:
        """Retire an outdated memory without deleting its provenance.

        Normal search and context exclude stale memories. Use ``fibmind_revise``
        instead when the same memory should be corrected and made active again.
        """
        return service.mark_stale(node_id, reason)

    @mcp.tool()
    def fibmind_forget(node_id: str, reason: str) -> dict[str, Any]:
        """Delete a memory permanently, including from the event log.

        Use for memories the user asks to remove, content stored by mistake, or
        anything that should not have been captured. The deletion also redacts
        the text from log history, so it does not return on a replay.
        """
        return service.forget(node_id, reason)

    @mcp.tool()
    def fibmind_promote_knowledge(
        title: str,
        content: str,
        supporting_node_ids: list[str],
        category: str = "knowledge",
        tags: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Turn several observations into one shared, reusable claim.

        Use when the same pattern has appeared repeatedly and the general form is
        worth keeping — three separate memories about one failure mode becoming
        a single statement about its cause.

        Requires at least three distinct supporting memories: whether something
        generalizes is a claim about a population, not a judgement call. Each
        supporter is linked as provenance, so a claim that later proves wrong can
        be traced back. Write ``content`` at a level that changes what you would
        do next; if it is too abstract to act on, it is not worth promoting.
        """
        return service.promote_knowledge(
            title,
            content,
            supporting_node_ids,
            category=category,
            tags=tags,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
        )

    @mcp.tool()
    def fibmind_link(
        from_node_id: str,
        to_node_id: str,
        relation_type: str,
        weight: float = 1.0,
        bidirectional: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a typed relationship between two existing memories.

        Use node IDs returned by append, search, or context. ``relation_type``
        must be one of the FibMind relation values. Set ``bidirectional`` when
        the relationship itself should be stored as bidirectional.
        """
        return service.link(
            from_node_id,
            to_node_id,
            relation_type,
            weight=weight,
            bidirectional=bidirectional,
            metadata=metadata,
        )

    @mcp.tool()
    def fibmind_search(
        query: str,
        top_k: int = 5,
        categories: list[str] | None = None,
        min_score: float = 0.0,
        scopes: list[str] | None = None,
        owner: str | None = None,
        statuses: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Search memories by keyword relevance to ``query``.

        Read-only. Returns the top ranked memories with their node_ids so you
        can expand or link them. Optionally restrict to ``categories`` or
        ``scopes``; pass ``owner`` to keep other owners' personal memories out.
        Pass the current workspace / project / session so labelled memories
        from another context stay out.
        """
        return service.search(
            query,
            top_k=top_k,
            categories=categories,
            min_score=min_score,
            scopes=scopes,
            owner=owner,
            statuses=statuses,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
        )

    @mcp.tool()
    def fibmind_search_from(
        node_id: str,
        depth: int = 2,
        relation_types: list[str] | None = None,
        reinforce: bool = False,
        direction: str = "both",
        scopes: list[str] | None = None,
        owner: str | None = None,
        statuses: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Expand the association tree rooted at ``node_id`` up to ``depth`` hops.

        Use a node_id returned by fibmind_search or fibmind_context to explore
        related memories. ``reinforce`` marks traversed memories as familiar;
        note that this does not make them more trusted — only
        fibmind_record_outcome does that. Invisible neighbours are not
        traversed, so a graph edge cannot leak another project or session.
        """
        return service.search_from(
            node_id,
            depth=depth,
            relation_types=relation_types,
            reinforce=reinforce,
            direction=direction,
            scopes=scopes,
            owner=owner,
            statuses=statuses,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
        )

    @mcp.tool()
    def fibmind_context(
        goal: str,
        top_k: int = 5,
        depth: int = 1,
        max_chars: int = 2000,
        reinforce: bool = False,
        scopes: list[str] | None = None,
        owner: str | None = None,
        statuses: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        budget_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build a compact memory pack to load into context before a task.

        Call this at the START of a multi-step task with the user's goal. It
        searches for relevant history and expands related memories, returning a
        char-bounded ``text`` block ready to drop into your working context,
        plus the structured ``hits``. Pass the current workspace / project /
        session so recall stays inside this working context.
        """
        return service.context(
            goal,
            top_k=top_k,
            depth=depth,
            max_chars=max_chars,
            reinforce=reinforce,
            scopes=scopes,
            owner=owner,
            statuses=statuses,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            budget_tokens=budget_tokens,
        )

    @mcp.tool()
    def fibmind_explain_recall(
        query: str,
        top_k: int = 5,
        categories: list[str] | None = None,
        scopes: list[str] | None = None,
        owner: str | None = None,
        statuses: list[str] | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Explain a recall: per-result lexical / vector / fusion / confidence /
        recency signals, matched terms, and why other candidates were excluded
        (status, visibility, folded, category). Read-only; use it when a recall
        looks wrong before revising or marking memories stale.
        """
        return service.explain_recall(
            query,
            top_k=top_k,
            categories=categories,
            scopes=scopes,
            owner=owner,
            statuses=statuses,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            exclude_categories=["episode"],
        )

    @mcp.tool()
    def fibmind_status() -> dict[str, Any]:
        """Store size, lifecycle counts, lexical index size, and embedding health."""
        return service.status()

    @mcp.tool()
    def fibbrain_recall(
        goal: str,
        top_k: int = 5,
        depth: int = 1,
        max_chars: int = 2000,
        budget_tokens: int | None = None,
        include_hot: bool = True,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Recall what the brain already knows about a goal, inside a token budget.

        Call this at the start of a task. Pass the current workspace / project /
        session so recall stays inside this working context. The pack renders
        the session's frozen hot memory first, then direct hits, then related
        memories, and reports ``budget`` usage per section. ``budget_tokens``
        defaults to 500. Prefer this over fibmind_context — it also returns the
        episode log.
        """
        return brain.recall(
            goal,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
            top_k=top_k,
            depth=depth,
            max_chars=max_chars,
            budget_tokens=budget_tokens,
            include_hot=include_hot,
        )

    @mcp.tool()
    def fibbrain_hot(
        refresh: bool = False,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """The frozen hot-memory snapshot (stable preferences, conventions,
        evidenced decisions) for this session. Computed once per session and
        reused by every fibbrain_recall; ``refresh`` recomputes it.
        """
        return brain.hot_snapshot(
            _state(owner, workspace_id, project_id, session_id, task_id),
            refresh=refresh,
        )

    @mcp.tool()
    def fibbrain_remember(
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        scope: str = "personal",
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Admit a candidate, then write it if it is worth keeping.

        Use this after a task instead of raw fibmind_append. Duplicates, raw
        dumps, and empty speculation are skipped and returned with a reason.
        """
        return brain.remember(
            category,
            title,
            content,
            tags=tags,
            metadata=metadata,
            scope=scope,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
        )

    @mcp.tool()
    def fibbrain_observe(
        kind: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Note an episode event (tool result, decision, test, error, correction).

        With a ``session_id`` the event is persisted as session-scoped scratch
        that only this session can recall, and ``fibbrain_review_session`` can
        later distil it. Use ``kind`` values ``decision``, ``test``, ``error``,
        ``correction``, ``risk``, or ``tool_result``; put changed paths under
        ``payload.files`` and the command under ``payload.command``. Observation
        is not long-term memory: call fibbrain_remember for that.
        """
        return brain.observe(
            kind,
            summary,
            payload=payload,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
        )

    @mcp.tool()
    def fibbrain_remember_procedure(
        title: str,
        trigger: str,
        steps: list[str],
        when: list[str] | None = None,
        tools: list[str] | None = None,
        verify: str = "",
        inputs: dict[str, str] | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Store a repeatable way of doing a class of task (procedural memory).

        ``trigger`` says what task it solves, ``when`` lists cue words, ``steps``
        are the ordered actions, ``tools`` the tools they need, ``verify`` how
        to tell it worked, ``inputs`` names any parameters. Later,
        fibbrain_render turns it into a skill or tool and fibbrain_reflect on
        its node_id records whether it worked; two refutations with no
        confirmation retire it automatically.
        """
        return brain.remember_procedure(
            title,
            trigger,
            steps,
            when=when,
            tools=tools,
            verify=verify,
            inputs=inputs,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
        )

    @mcp.tool()
    def fibbrain_render(
        goal: str,
        format: str = "skill",
        top_k: int = 3,
        min_maturity: str = "candidate",
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Render the procedures that fit ``goal`` as skills or tool definitions.

        ``format`` is ``skill`` (SKILL.md text you can save under a skills
        directory) or ``tool`` (a JSON-Schema tool definition). ``min_maturity``
        is ``candidate``, ``verified`` (confirmed once), or ``established``
        (confirmed three times across two sessions). Nothing is executed.
        """
        return brain.render(
            goal,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
            format=format,
            top_k=top_k,
            min_maturity=min_maturity,
        )

    @mcp.tool()
    def fibbrain_tune(apply: bool = True) -> dict[str, Any]:
        """Let the brain adjust its own ranking weights from evidence — but only
        if the fixed evaluation suite proves nothing regressed.

        Returns the evidence profile (confirmed vs refuted memories), the
        bounded proposal, the gate scores before and after, and the decision.
        Every attempt is appended to the log. ``apply=False`` previews.
        """
        return brain.tune(apply=apply)

    @mcp.tool()
    def fibbrain_revalidation_candidates(
        older_than_days: int = 90,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Knowledge and procedures with no confirmation for ``older_than_days``.
        Nothing is changed; re-check them and fibbrain_reflect the result.
        """
        return brain.revalidation_candidates(
            _state(owner, workspace_id, project_id, session_id, task_id),
            older_than_days=older_than_days,
        )

    @mcp.tool()
    def fibbrain_review_session(
        session_id: str,
        mode: str = "approve",
        close: bool = False,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Distil one session's observations into long-term memory candidates.

        Deterministic: decisions, errors, corrections, risks, a verification
        record, the files touched, and one summary. ``mode`` is ``candidates``
        (preview only), ``approve`` (write as pending until
        fibbrain_approve_memory), or ``auto`` (write as active). Running it
        twice writes nothing new. ``close`` retires the reviewed episode.
        """
        return brain.review_session(
            _state(owner, workspace_id, project_id, session_id, task_id),
            mode=mode,
            close=close,
        )

    @mcp.tool()
    def fibbrain_pending_reviews(
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List review memories waiting for approval in this working context."""
        return brain.pending_reviews(_state(owner, workspace_id, project_id, session_id, task_id))

    @mcp.tool()
    def fibbrain_approve_memory(node_id: str, reason: str = "approved by reviewer") -> dict[str, Any]:
        """Let a pending review memory into normal recall."""
        return brain.approve_memory(node_id, reason)

    @mcp.tool()
    def fibbrain_reject_memory(node_id: str, reason: str = "rejected by reviewer") -> dict[str, Any]:
        """Retire a pending review memory; it stays inspectable as stale."""
        return brain.reject_memory(node_id, reason)

    @mcp.tool()
    def fibbrain_advise(
        action: str,
        kind: str = "tool",
        arguments: dict[str, Any] | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Ask whether an action should run, given stored evidence.

        Rejects when a refuted memory matches the action — and, if you pass the
        tool ``arguments``, also shares a term with them, so ``rm build/`` being
        refuted does not block ``rm tmp/``. Otherwise allows and returns related
        memories as notes.
        """
        return brain.advise(
            action,
            kind=kind,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
            top_k=top_k,
            arguments=arguments,
        )

    @mcp.tool()
    def fibbrain_reflect(
        verdict: str,
        source: str,
        node_id: str | None = None,
        note: str | None = None,
        goal: str | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Record that a memory turned out right or wrong.

        Prefer this over fibmind_record_outcome. Pass ``node_id`` or a ``goal``
        that recalls the memory. ``source`` must name what checked it.
        """
        return brain.reflect(
            verdict,
            source,
            node_id=node_id,
            note=note,
            goal=goal,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
        )

    @mcp.tool()
    def fibbrain_plan(
        objective: str | None = None,
        goal_id: str | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or refresh a persisted goal and its plan.

        Call this at the start of a non-trivial task. The plan is deterministic:
        recall first, then avoid/reuse steps from stored evidence, then the work
        implied by the objective, then verify, then reflect and remember.
        Pass ``goal_id`` to refresh an existing goal, or ``objective`` to create
        one. The same objective in the same working context is reused.
        """
        return brain.plan(
            objective=objective,
            goal_id=goal_id,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
        )

    @mcp.tool()
    def fibbrain_coordinate(
        objective: str | None = None,
        goal_id: str | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Recommend which capabilities to involve next, with an advise on each.

        This is not a plugin manager and does not execute anything. Use it after
        ``fibbrain_plan`` (or pass ``objective`` to plan first). A rejected
        advise marks that capability blocked given stored evidence.
        """
        return brain.coordinate(
            objective=objective,
            goal_id=goal_id,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
        )

    @mcp.tool()
    def fibbrain_complete_goal(
        goal_id: str | None = None,
        objective: str | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Mark a persisted goal complete. It stays recallable as history."""
        return brain.complete_goal(
            goal_id=goal_id,
            objective=objective,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
        )

    return mcp


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fibmind.mcp_server", description="FibBrain MCP server")
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help=f"Path to a SQLite or JSON memory store (default: {DEFAULT_STORE})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # Semantic recall is opt-in through FIBMIND_EMBEDDING=hashing|openai (plus
    # FIBMIND_EMBEDDING_URL / _MODEL / _API_KEY for openai). Unset = lexical.
    service = MemoryService(Path(args.store), embedding_provider=provider_from_env())
    server = build_server(service)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
