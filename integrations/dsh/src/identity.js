/**
 * Map a dsh session onto FibBrain's identity quintet.
 *
 * dsh has no owner / project concept of its own (its identity package is an
 * anonymous install id), so:
 *
 *   owner        config.owner, else $USER, else "dsh"
 *   workspace_id basename of the git root containing the session cwd
 *   project_id   config.projectId, else the workspace_id
 *   session_id   the dsh session id (shared agent/session identity)
 *   task_id      never set here
 */

import { existsSync } from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'

/**
 * @param {string | undefined} cwd
 * @returns {string | undefined}
 */
export function gitRoot(cwd) {
  if (!cwd) return undefined
  let current = resolve(cwd)
  for (;;) {
    if (existsSync(join(current, '.git'))) return current
    const parent = dirname(current)
    if (parent === current) return undefined
    current = parent
  }
}

/**
 * @param {{ owner?: string, projectId?: string, workspaceId?: string }} config
 * @param {{ cwd?: string, sessionId?: string }} session
 */
export function identityFor(config, session) {
  const root = config.workspaceId ? undefined : gitRoot(session.cwd)
  const workspace = config.workspaceId ?? (root ? basename(root) : session.cwd ? basename(resolve(session.cwd)) : undefined)
  return {
    owner: config.owner ?? process.env.USER ?? 'dsh',
    workspace_id: workspace,
    project_id: config.projectId ?? workspace,
    session_id: session.sessionId,
  }
}

/** The cwd and id of an agent's session, tolerant of shape differences. */
export function sessionOf(agent) {
  const session = agent?.session
  const meta = session?.meta ?? session?.header ?? {}
  return {
    cwd: typeof meta.cwd === 'string' ? meta.cwd : undefined,
    sessionId: String(agent?.id ?? session?.id ?? ''),
  }
}
