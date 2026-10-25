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
import functools
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from fibmind.brain import BrainState, FibBrain
from fibmind.embedding import provider_from_env
from fibmind.migrate import DEFAULT_HOST_OWNERS
from fibmind.service import MemoryService
from fibmind.working_context import working_context

DEFAULT_STORE = ".fibmind/memory.db"
MEMORY_TOOLS = frozenset(
    {
        "fibbrain_context",
        "fibbrain_remember",
        "fibbrain_reflect",
        "fibbrain_remember_procedure",
        "fibbrain_render",
        "fibmind_search",
        "fibmind_search_from",
        "fibmind_explain_recall",
        "fibmind_status",
        "fibmind_revise",
        "fibmind_mark_stale",
        "fibmind_forget",
        "fibmind_promote_knowledge",
        "fibmind_link",
        "fibmind_mark_conflict",
        "fibbrain_resolve_conflict",
    }
)


@dataclass(frozen=True, slots=True)
class IdentityDefaults:
    """Identity the server fills in when a tool call leaves it out.

    Cross-tool sharing depends on Codex, Claude Code, and Antigravity all
    writing under the *same* owner. Models routinely omit ``owner`` or pass
    their own host name, so the server, which is configured once per
    machine, is the reliable place to pin it. With ``owner`` set, any caller
    owner listed in ``owner_aliases`` (the known AI host names by default) is
    mapped onto it too.
    """

    owner: str | None = None
    workspace_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    application: str | None = None
    owner_aliases: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_HOST_OWNERS))

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **overrides: str | None) -> IdentityDefaults:
        env = os.environ if environ is None else environ

        def pick(name: str) -> str | None:
            value = overrides.get(name)
            if value is None:
                value = env.get(f"FIBMIND_{name.upper()}")
            value = value.strip() if value else None
            return value or None

        extra = {alias.strip() for alias in env.get("FIBMIND_OWNER_ALIASES", "").split(",") if alias.strip()}
        return cls(
            owner=pick("owner"),
            workspace_id=pick("workspace_id"),
            project_id=pick("project_id"),
            task_id=pick("task_id"),
            application=pick("application"),
            owner_aliases=frozenset(DEFAULT_HOST_OWNERS) | frozenset(extra),
        )

    @property
    def active(self) -> bool:
        return any((self.owner, self.workspace_id, self.project_id, self.task_id, self.application))

    def apply(self, arguments: dict[str, Any], accepted: set[str]) -> dict[str, Any]:
        resolved = dict(arguments)
        if self.application and "metadata" in accepted:
            metadata = dict(resolved.get("metadata") or {})
            metadata.setdefault("host", self.application)
            resolved["metadata"] = metadata
        if "owner" in accepted and self.owner is not None:
            owner = resolved.get("owner")
            owner = str(owner).strip() if owner is not None else ""
            if not owner or (owner in self.owner_aliases and owner != self.owner):
                resolved["owner"] = self.owner
                if owner and "metadata" in accepted:
                    metadata = dict(resolved.get("metadata") or {})
                    metadata.setdefault("host", owner)
                    resolved["metadata"] = metadata
        for name in ("workspace_id", "project_id", "task_id"):
            default = getattr(self, name)
            if name in accepted and default is not None and not resolved.get(name):
                resolved[name] = default
        return resolved


def _state(
    owner: str | None,
    workspace_id: str | None,
    project_id: str | None,
    session_id: str | None,
    task_id: str | None,
    workspace_path: str | None = None,
) -> BrainState:
    if workspace_path:
        resolved = working_context(workspace_path)
        workspace_id = workspace_id or resolved["workspace_id"]
        project_id = project_id or resolved["project_id"]
        task_id = task_id or resolved["task_id"]
    return BrainState(
        owner=owner,
        workspace_id=workspace_id,
        project_id=project_id,
        session_id=session_id,
        task_id=task_id,
    )


def _tool_registrar(
    mcp: MCPServer, defaults: IdentityDefaults, enabled_tools: frozenset[str] | None = None
) -> Callable[[], Callable[[Callable], Callable]]:
    """``mcp.tool()`` that first fills identity arguments from ``defaults``.

    ``functools.wraps`` keeps the original signature and docstring, which is
    what MCP introspects to build each tool's schema.
    """

    def tool() -> Callable[[Callable], Callable]:
        def decorate(fn: Callable) -> Callable:
            if enabled_tools is not None and fn.__name__ not in enabled_tools:
                return fn
            annotations = (
                ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
                if fn.__name__ == "fibbrain_context"
                else None
            )
            if not defaults.active:
                return mcp.tool(annotations=annotations)(fn)
            accepted = set(inspect.signature(fn).parameters)

            @functools.wraps(fn)
            def wrapper(**kwargs: Any) -> Any:
                return fn(**defaults.apply(kwargs, accepted))

            return mcp.tool(annotations=annotations)(wrapper)

        return decorate

    return tool


def build_server(
    service: MemoryService,
    brain: FibBrain | None = None,
    defaults: IdentityDefaults | None = None,
    *,
    profile: str = "full",
) -> MCPServer:
    """Create an MCP server whose tools delegate to ``service`` and ``brain``."""
    if profile not in {"memory", "full"}:
        raise ValueError("profile must be memory or full")
    brain = brain or FibBrain(service)
    mcp = MCPServer("fibmind")
    tool = _tool_registrar(mcp, defaults or IdentityDefaults(), MEMORY_TOOLS if profile == "memory" else None)

    @tool()
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
        scratch requires ``session_id`` or ``task_id``; the same ``task_id``
        lets another AI tool's session recall it.

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

    @tool()
    def fibmind_record_outcome(
        node_id: str,
        verdict: str,
        source: str,
        note: str | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Record that a memory turned out to be right or wrong.

        ``verdict`` is "confirmed" or "refuted"; ``source`` must name what
        checked it — a test command, a failing build, a user correction. This is
        the only thing that moves a memory's confidence, and confidence is the
        only trust signal that affects ranking.

        Call it whenever evidence appears: a recorded fix that made the tests
        pass, a decision that was later reverted, advice the user rejected.
        """
        return service.record_outcome(
            node_id,
            verdict,
            source,
            note=note,
            session_id=session_id,
            identity=_state(owner, workspace_id, project_id, session_id, task_id).identity(),
        )

    @tool()
    def fibmind_revise(
        node_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        expected_version: int | None = None,
        protect_verified: bool = False,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Correct a memory whose content is wrong or out of date.

        Prefer this over appending a contradicting memory. Confidence resets to
        zero, since evidence backing the old text says nothing about the new.
        Pass ``expected_version`` for optimistic concurrency control (OCC).
        Set ``protect_verified=True`` to stage revisions of verified/knowledge
        memories as pending shadow candidates until verified.
        """
        return service.revise(
            node_id,
            title=title,
            content=content,
            tags=tags,
            expected_version=expected_version,
            protect_verified=protect_verified,
            identity=_state(owner, workspace_id, project_id, session_id, task_id).identity(),
        )

    @tool()
    def fibmind_mark_conflict(
        node_id_a: str,
        node_id_b: str,
        reason: str | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Mark two memories as in dialectical contradiction with bidirectional CONTRADICTS edges.

        Use when two agents or observations assert conflicting claims about the codebase or behavior.
        """
        return service.mark_conflict(
            node_id_a,
            node_id_b,
            reason=reason,
            identity=_state(owner, workspace_id, project_id, session_id, task_id).identity(),
        )

    @tool()
    def fibmind_mark_stale(
        node_id: str,
        reason: str,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Retire an outdated memory without deleting its provenance.

        Normal search and context exclude stale memories. Use ``fibmind_revise``
        instead when the same memory should be corrected and made active again.
        """
        return service.mark_stale(
            node_id, reason, identity=_state(owner, workspace_id, project_id, session_id, task_id).identity()
        )

    @tool()
    def fibmind_forget(
        node_id: str,
        reason: str,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete a memory permanently, including from the event log.

        Use for memories the user asks to remove, content stored by mistake, or
        anything that should not have been captured. The deletion also redacts
        the text from log history, so it does not return on a replay.
        """
        return service.forget(
            node_id, reason, identity=_state(owner, workspace_id, project_id, session_id, task_id).identity()
        )

    @tool()
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
        owner: str | None = None,
        require_cross_provenance: bool = False,
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
            require_cross_provenance=require_cross_provenance,
            identity=_state(owner, workspace_id, project_id, session_id, task_id).identity(),
        )

    @tool()
    def fibmind_link(
        from_node_id: str,
        to_node_id: str,
        relation_type: str,
        weight: float = 1.0,
        bidirectional: bool = False,
        metadata: dict[str, Any] | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
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
            identity=_state(owner, workspace_id, project_id, session_id, task_id).identity(),
        )

    @tool()
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
        task_id: str | None = None,
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
            task_id=task_id,
        )

    @tool()
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
        task_id: str | None = None,
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
            task_id=task_id,
        )

    @tool()
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
        task_id: str | None = None,
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
            task_id=task_id,
            budget_tokens=budget_tokens,
        )

    @tool()
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
        task_id: str | None = None,
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
            task_id=task_id,
            exclude_categories=["episode"],
        )

    @tool()
    def fibmind_status() -> dict[str, Any]:
        """Store size, lifecycle counts, lexical index size, and embedding health."""
        return service.status()

    @tool()
    def fibbrain_context(
        goal: str = "",
        top_k: int = 5,
        profile_top_k: int = 3,
        budget_tokens: int = 500,
        min_relevance: float = 0.12,
        model_id: str | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        workspace_path: str | None = None,
    ) -> dict[str, Any]:
        """Read relevant human background for Codex, Claude Code, AGY, or any harness.

        Call before answering a user request. Combines applicable user profile,
        project memories, and shared knowledge. With no goal, returns profile
        only. Does not write plans, snapshots, or episodes, and executes nothing.
        ``budget_tokens`` bounds rendered ``text``; ``memories`` contains compact
        provenance for rendered entries only. Historical evidence is not fresh
        verification. Current user instructions take precedence over memories.
        Supply absolute ``workspace_path`` to derive the same workspace,
        project, and git-branch task labels in every local harness. Explicit
        IDs or server defaults take precedence. Reuse returned ``state`` when
        remembering or checking a memory later.
        """
        return brain.context(
            goal,
            state=_state(owner, workspace_id, project_id, session_id, task_id, workspace_path),
            top_k=top_k,
            profile_top_k=profile_top_k,
            budget_tokens=budget_tokens,
            min_relevance=min_relevance,
            model_id=model_id,
        )

    @tool()
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

    @tool()
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

    @tool()
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
        context_level: str | None = None,
        workspace_path: str | None = None,
    ) -> dict[str, Any]:
        """Admit a candidate, then write it if it is worth keeping.

        Use this after a task instead of raw fibmind_append. Duplicates, raw
        dumps, and empty speculation are skipped and returned with a reason.
        For personal memories, explicitly set ``context_level`` to "user" for
        cross-project profile/preferences, "workspace" for workspace-wide
        context, or "project" for the current project. Omitting it preserves
        existing exact-context behavior. The human owner stays private.
        """
        return brain.remember(
            category,
            title,
            content,
            tags=tags,
            metadata=metadata,
            scope=scope,
            state=_state(owner, workspace_id, project_id, session_id, task_id, workspace_path),
            context_level=context_level,
        )

    @tool()
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

        With a ``session_id`` or ``task_id`` the event is persisted as
        session-scoped scratch that this session — or any session (in any AI
        tool) passing the same ``task_id`` — can recall, and ``fibbrain_review_session`` can
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

    @tool()
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

    @tool()
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

    @tool()
    def fibbrain_tune(apply: bool = True) -> dict[str, Any]:
        """Let the brain adjust its own ranking weights from evidence — but only
        if the fixed evaluation suite proves nothing regressed.

        Returns the evidence profile (confirmed vs refuted memories), the
        bounded proposal, the gate scores before and after, and the decision.
        Every attempt is appended to the log. ``apply=False`` previews.
        """
        return brain.tune(apply=apply)

    @tool()
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

    @tool()
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

    @tool()
    def fibbrain_pending_reviews(
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List review memories waiting for approval in this working context."""
        return brain.pending_reviews(_state(owner, workspace_id, project_id, session_id, task_id))

    @tool()
    def fibbrain_approve_memory(
        node_id: str,
        reason: str = "approved by reviewer",
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Let a pending review memory into normal recall."""
        return brain.approve_memory(node_id, reason, _state(owner, workspace_id, project_id, session_id, task_id))

    @tool()
    def fibbrain_reject_memory(
        node_id: str,
        reason: str = "rejected by reviewer",
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Retire a pending review memory; it stays inspectable as stale."""
        return brain.reject_memory(node_id, reason, _state(owner, workspace_id, project_id, session_id, task_id))

    @tool()
    def fibbrain_advise(
        action: str,
        kind: str = "tool",
        arguments: dict[str, Any] | None = None,
        planned_files: list[str] | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Ask whether an action should run, given stored evidence.

        Rejects when a refuted memory matches the action — and, if you pass the
        tool ``arguments`` or ``planned_files``, also shares a term or file path
        with them, so ``rm build/`` being refuted does not block ``rm tmp/``.
        Otherwise allows and returns related memories as notes.
        """
        return brain.advise(
            action,
            kind=kind,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
            top_k=top_k,
            arguments=arguments,
            planned_files=planned_files,
        )

    @tool()
    def fibbrain_reflex(
        goal: str,
        execute: bool = False,
        inputs: dict[str, str] | None = None,
        min_confidence: float = 0.75,
        min_maturity: str = "verified",
        model_id: str | None = None,
        require_files_exist: bool = False,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """System 1 fast reflex: answer from high-confidence verified memory or run a verified procedure without LLM.

        Returns ``bypass_llm=True`` when a verified declarative memory or procedure
        satisfies environmental preconditions (file hashes, target files, non-conflicted,
        model-generation compatible). When ``execute=True`` and a verified procedure
        matches, runs its steps and automatically trips a circuit breaker (`refuted`)
        if execution fails, falling back to ``recall`` context for System 2 LLM reasoning.
        """
        return brain.reflex(
            goal,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
            execute=execute,
            inputs=inputs,
            min_confidence=min_confidence,
            min_maturity=min_maturity,
            model_id=model_id,
            require_files_exist=require_files_exist,
        )

    @tool()
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

    @tool()
    def fibbrain_verify(
        command: str,
        node_id: str | None = None,
        goal: str | None = None,
        source: str | None = None,
        target_files: list[str] | None = None,
        assertion_patterns: list[str] | None = None,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a verification command and automatically reflect its outcome.

        Runs the command in the workspace (e.g. 'pytest -q', 'cargo check', 'npm test').
        If the command exits with 0 and touches target_files, records verdict='confirmed'.
        If the command exits non-zero on target_files, records verdict='refuted'.
        Environment or unrelated failures record verdict='inconclusive' without penalty.
        """
        from fibmind.verify import run_command_and_reflect

        return run_command_and_reflect(
            command,
            node_id=node_id,
            goal=goal,
            source=source,
            target_files=target_files,
            assertion_patterns=assertion_patterns,
            store_path=service._store.path,
            owner=owner,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
        )

    @tool()
    def fibbrain_resolve_conflict(
        winning_node_id: str,
        losing_node_id: str,
        reason: str,
        source: str,
        owner: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Arbitrate a conflict: refute losing node with evidence and clear conflict on winner.

        Requires an objective evidence source (e.g. 'pytest', 'compiler', 'user_feedback')
        to refute the losing memory.
        """
        return brain.resolve_conflict(
            winning_node_id,
            losing_node_id,
            reason=reason,
            source=source,
            state=_state(owner, workspace_id, project_id, session_id, task_id),
        )

    @tool()
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

    @tool()
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

    @tool()
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
        default=os.environ.get("FIBMIND_STORE") or DEFAULT_STORE,
        help=f"Path to a SQLite or JSON memory store (env FIBMIND_STORE, default: {DEFAULT_STORE})",
    )
    parser.add_argument("--owner", help="Default owner when a call omits it or passes an AI host name (FIBMIND_OWNER)")
    parser.add_argument("--workspace-id", help="Default workspace_id (FIBMIND_WORKSPACE_ID)")
    parser.add_argument("--project-id", help="Default project_id (FIBMIND_PROJECT_ID)")
    parser.add_argument("--task-id", help="Default task_id (FIBMIND_TASK_ID)")
    parser.add_argument(
        "--profile",
        choices=["memory", "full"],
        default=os.environ.get("FIBMIND_TOOL_PROFILE", "full"),
        help="memory: context and memory management only; full: legacy cognitive/command tools too",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=os.environ.get("FIBMIND_TRANSPORT") or "stdio",
        help="Transport protocol: 'stdio' (default), 'sse', or 'streamable-http' (env FIBMIND_TRANSPORT)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("FIBMIND_HOST") or "127.0.0.1",
        help="Host address to bind HTTP/SSE server (env FIBMIND_HOST, default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FIBMIND_PORT") or "8765"),
        help="Port to bind HTTP/SSE server (env FIBMIND_PORT, default: 8765)",
    )
    parser.add_argument(
        "--allow-insecure-remote",
        action="store_true",
        default=os.environ.get("FIBMIND_ALLOW_INSECURE_REMOTE", "").lower() in ("1", "true", "yes"),
        help="Allow binding daemon to non-loopback addresses (insecure, not recommended)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("FIBMIND_TOKEN"),
        help="Bearer token required for daemon HTTP/SSE authentication (env FIBMIND_TOKEN)",
    )
    return parser.parse_args(argv)


def run_server(
    server: MCPServer,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    allow_insecure_remote: bool = False,
    token: str | None = None,
) -> None:
    """Run the MCP server over stdio or as a daemon over SSE / streamable-http."""
    if transport == "stdio":
        server.run(transport="stdio")
        return

    safe_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in safe_hosts and not allow_insecure_remote:
        raise ValueError(
            f"Refusing to bind daemon to non-loopback address '{host}'. "
            "FibMind is a single-user local workstation memory store and does not support multi-tenant exposure. "
            "Pass --allow-insecure-remote to override."
        )

    if transport == "sse":
        import anyio
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        app = server.sse_app(host=host)

        if token:
            expected_token = token

            class TokenAuthMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request: Any, call_next: Any) -> Any:
                    if request.url.path in ("/health",):
                        return await call_next(request)
                    auth = request.headers.get("authorization", "")
                    query_token = request.query_params.get("token", "")
                    passed_token = auth[7:].strip() if auth.startswith("Bearer ") else query_token
                    if not passed_token or passed_token != expected_token:
                        return JSONResponse({"error": "Unauthorized: invalid or missing daemon token"}, status_code=401)
                    return await call_next(request)

            app.add_middleware(TokenAuthMiddleware)

        async def health_endpoint(request: Any) -> JSONResponse:
            return JSONResponse({"status": "ok", "service": "fibmind", "transport": "sse"})

        app.routes.append(Route("/health", health_endpoint))
        app.routes.append(Route("/", health_endpoint))

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        uvicorn_server = uvicorn.Server(config)
        anyio.run(uvicorn_server.serve)
    elif transport == "streamable-http":
        import anyio
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        app = server.streamable_http_app(host=host)

        if token:
            expected_token = token

            class TokenAuthMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request: Any, call_next: Any) -> Any:
                    if request.url.path in ("/health",):
                        return await call_next(request)
                    auth = request.headers.get("authorization", "")
                    query_token = request.query_params.get("token", "")
                    passed_token = auth[7:].strip() if auth.startswith("Bearer ") else query_token
                    if not passed_token or passed_token != expected_token:
                        return JSONResponse({"error": "Unauthorized: invalid or missing daemon token"}, status_code=401)
                    return await call_next(request)

            app.add_middleware(TokenAuthMiddleware)

        async def health_endpoint(request: Any) -> JSONResponse:
            return JSONResponse({"status": "ok", "service": "fibmind", "transport": "streamable-http"})

        app.routes.append(Route("/health", health_endpoint))
        app.routes.append(Route("/", health_endpoint))

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        uvicorn_server = uvicorn.Server(config)
        anyio.run(uvicorn_server.serve)
    else:
        raise ValueError(f"Unknown transport: {transport}. Choose 'stdio', 'sse', or 'streamable-http'")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    defaults = IdentityDefaults.from_env(
        owner=args.owner,
        workspace_id=args.workspace_id,
        project_id=args.project_id,
        task_id=args.task_id,
    )
    store = Path(args.store).expanduser().resolve()
    # stdout carries the MCP protocol; diagnostics go to stderr. Logging the
    # resolved path makes a relative --store that split one brain into two
    # (each host launching from a different cwd) visible at a glance.
    print(
        f"fibmind: store={store} owner={defaults.owner or '-'} "
        f"workspace={defaults.workspace_id or '-'} project={defaults.project_id or '-'} "
        f"transport={args.transport}",
        file=sys.stderr,
    )
    # Semantic recall is opt-in through FIBMIND_EMBEDDING=hashing|openai (plus
    # FIBMIND_EMBEDDING_URL / _MODEL / _API_KEY for openai). Unset = lexical.
    service = MemoryService(store, embedding_provider=provider_from_env())
    server = build_server(service, defaults=defaults, profile=args.profile)
    run_server(
        server,
        transport=args.transport,
        host=args.host,
        port=args.port,
        allow_insecure_remote=args.allow_insecure_remote,
        token=args.token,
    )


def daemon_main(argv: list[str] | None = None) -> None:
    """Entry point for running FibMind as a shared background daemon over SSE."""
    args = sys.argv[1:] if argv is None else list(argv)
    if "--transport" not in args:
        args = ["--transport", "sse"] + args
    main(args)


if __name__ == "__main__":
    main()
