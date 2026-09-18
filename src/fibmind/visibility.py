"""Who may recall which memory.

One rule, applied at every recall entry point (search, graph expansion,
context, hot memory, procedures): a node is visible to a caller only if its
scope, owner, working context, and lifecycle status all agree with the
caller's. Keeping it in one place is what lets the identity tests assert the
rule once instead of once per code path.
"""

from __future__ import annotations

from fibmind.models import MemoryNode, MemoryScope, MemoryStatus, optional_id

ACTIVE_ONLY: frozenset[MemoryStatus] = frozenset({MemoryStatus.ACTIVE})


def is_visible(
    node: MemoryNode,
    *,
    scopes: set[MemoryScope] | None = None,
    owner: str | None = None,
    include_folded: bool = False,
    statuses: set[MemoryStatus] | frozenset[MemoryStatus] | None = None,
    workspace_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
) -> bool:
    """Return whether ``node`` may be recalled by this caller.

    Knowledge is shared by definition, so it stays visible regardless of
    ``owner``; personal memories are only visible to their own owner, and
    session scratch only inside the session that wrote it.
    """
    allowed_statuses = statuses if statuses is not None else ACTIVE_ONLY
    if node.status not in allowed_statuses:
        return False
    if not include_folded and node.folded_into is not None:
        return False
    if scopes is not None and node.scope not in scopes:
        return False
    if node.scope != MemoryScope.KNOWLEDGE and node.owner != owner:
        return False
    if node.scope == MemoryScope.SESSION:
        caller_session = optional_id(session_id)
        if not node.session_id or not caller_session or node.session_id != caller_session:
            return False
    if node.scope == MemoryScope.KNOWLEDGE:
        # Knowledge written without a workspace / project is global; knowledge
        # written with one stays inside it.
        if node.workspace_id is not None and node.workspace_id != optional_id(workspace_id):
            return False
        if node.project_id is not None and node.project_id != optional_id(project_id):
            return False
        return True
    if node.workspace_id != optional_id(workspace_id):
        return False
    if node.project_id != optional_id(project_id):
        return False
    return True


def identity_key(node: MemoryNode) -> tuple:
    """The identity a fold may not cross: nodes with different keys are never
    summarized into one node."""
    return (node.scope, node.owner, node.workspace_id, node.project_id)


def promotion_identity(
    supporters: list[MemoryNode],
    workspace_id: str | None,
    project_id: str | None,
) -> tuple[str | None, str | None]:
    """The working context a promoted claim inherits from its supporters.

    All supporters must share one workspace and one project; a caller may
    restate that context but not override it.
    """
    workspaces = {supporter.workspace_id for supporter in supporters}
    projects = {supporter.project_id for supporter in supporters}
    if len(workspaces) != 1 or len(projects) != 1:
        raise ValueError("promotion supporters must share one workspace_id and one project_id")
    inferred_workspace = next(iter(workspaces))
    inferred_project = next(iter(projects))
    requested_workspace = optional_id(workspace_id)
    requested_project = optional_id(project_id)
    if requested_workspace is not None and requested_workspace != inferred_workspace:
        raise ValueError("promotion workspace_id does not match the supporting memories")
    if requested_project is not None and requested_project != inferred_project:
        raise ValueError("promotion project_id does not match the supporting memories")
    return inferred_workspace, inferred_project
