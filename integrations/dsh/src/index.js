/**
 * FibBrain for DeepSeek-Harness — the native Cordis plugin.
 *
 * What the zero-code MCP row cannot do, this does through dsh's lifecycle:
 *
 *   agent/session-start   inject the session's hot memory as context
 *   agent/pre-step        recall for each fresh user prompt, append the pack
 *   tools/pre-execute     advise; refuted evidence denies (or asks)
 *   tools/result          observe every tool outcome into the episode
 *   agent/turn-stopping   review the session into pending / active memories
 *   ctx.skills            procedures rendered as a skill catalog
 *
 * The plugin never executes a procedure and never blocks on its own failures:
 * every hook fails open and logs, except `advise`, whose *reject* is the point.
 */

const { createUserMessage } = await loadLlm()

import { FibBrainClient } from './client.js'
import { identityFor, sessionOf } from './identity.js'

export const name = 'fibbrain'
// Cordis holds the plugin until `tools` exists; `skills` is probed at use.
export const inject = ['tools']

/** @typedef {import('./types').Config} Config */

export const Config = await loadSchema()

const OWN_TOOL_PREFIX = 'mcp__'
const RECALL_TAG = 'fibbrain-recall'
const HOT_TAG = 'fibbrain-hot'

/**
 * @param {any} ctx  Cordis context
 * @param {Config} config
 */
export function apply(ctx, config, clientOverride) {
  const log = ctx.logger ?? console
  const client = clientOverride ?? new FibBrainClient({
    command: config.command,
    args: config.args,
    cwd: config.cwd,
    env: config.env,
    timeoutMs: config.toolCallTimeoutMs,
  })
  ctx.effect(() => () => client.close())

  const identity = agent => identityFor(config, sessionOf(agent))
  const brain = createBrain(client, log)

  // ---------------------------------------------------------------- session-start
  ctx.on('agent/session-start', ({ agent, source }) => {
    if (!config.hotOnStart) return
    void (async () => {
      const id = identity(agent)
      const pack = await brain.recall(config.hotGoal, id, {
        top_k: config.hotTopK,
        budget_tokens: config.hotBudgetTokens,
        include_hot: true,
      })
      if (!pack || !pack.text) return
      agent.inject(contextMessage(`<${HOT_TAG} source="${source}">\n${pack.text}\n</${HOT_TAG}>`))
    })()
  })

  // ---------------------------------------------------------------- pre-step
  ctx.on('agent/pre-step', async (payload, next) => {
    const decision = await next()
    if (!config.recall || decision.kind !== 'enter') return decision
    const prompt = freshUserText(decision.messages)
    if (!prompt) return decision
    const pack = await brain.recall(prompt, identity(payload.agent), {
      top_k: config.recallTopK,
      budget_tokens: config.recallBudgetTokens,
      include_hot: false,
    })
    if (!pack || !pack.text) return decision
    const used = pack.budget?.used_tokens ?? '?'
    return {
      ...decision,
      messages: [
        ...decision.messages,
        contextMessage(
          `<${RECALL_TAG} goal=${JSON.stringify(truncate(prompt, 80))} tokens="${used}">\n${pack.text}\n</${RECALL_TAG}>`,
        ),
      ],
    }
  })

  // ---------------------------------------------------------------- pre-execute
  ctx.on('tools/pre-execute', async (exec, next) => {
    if (config.advise === 'off' || isOwnTool(exec.name)) return next()
    const args = isRecord(exec.arguments) ? exec.arguments : undefined
    const decision = await brain.advise(exec.name, args, identity(exec.agent))
    if (decision && decision.verdict === 'reject') {
      const reason = `FibBrain: ${decision.reason}. ` + evidenceLine(decision.memories)
      return config.advise === 'ask' ? { kind: 'ask', reason } : { kind: 'deny', reason }
    }
    return next()
  })

  // ---------------------------------------------------------------- result
  ctx.on('tools/result', (exec, result) => {
    if (!config.observe || isOwnTool(exec.name)) return
    const args = isRecord(exec.arguments) ? exec.arguments : {}
    const summary = truncate(`${exec.name}: ${textOf(result.content) || (result.isError ? 'error' : 'ok')}`, 400)
    const payload = { tool: exec.name, isError: Boolean(result.isError) }
    const command = pick(args, ['command', 'cmd'])
    if (command) payload.command = String(command)
    const files = filesOf(args)
    if (files.length) payload.files = files
    if (result.isError && result.error) payload.error = String(result.error.code ?? result.error.name ?? '')
    void brain.observe(result.isError ? 'error' : 'tool_result', summary, payload, identity(exec.agent))
  })

  // ---------------------------------------------------------------- turn-stopping
  ctx.on('agent/turn-stopping', async ({ agent }) => {
    if (config.reviewMode === 'off') return
    const id = identity(agent)
    if (!id.session_id) return
    const report = await brain.review(id, config.reviewMode)
    if (report && Array.isArray(report.written) && report.written.length) {
      log.info?.(`fibbrain: review wrote ${report.written.length} ${config.reviewMode} memories for ${id.session_id}`)
      invalidateSkills()
    }
  })

  // ---------------------------------------------------------------- skills
  let invalidateSkills = () => {}
  const skills = ctx.get?.('skills') ?? ctx.skills
  if (config.skills && skills && typeof skills.registerProvider === 'function') {
    skills.registerProvider(control => {
      invalidateSkills = () => control.invalidate()
      return {
        name: 'fibbrain',
        async list(options) {
          const id = identityFor(config, { cwd: options?.cwd, sessionId: undefined })
          const rendered = await brain.render('*', id, config.skillsMinMaturity)
          const items = rendered?.items ?? []
          return items.map(item => ({
            name: item.name,
            description: `${item.title} (FibBrain procedure, ${item.maturity})`,
            invocation: { modelInvocable: true, userInvocable: true },
            source: 'runtime',
            provider: 'fibbrain',
            rank: 50,
            locator: { node_id: item.node_id, text: item.text },
          }))
        },
        async get(candidate) {
          const locator = candidate?.locator
          if (!locator || typeof locator.text !== 'string') return undefined
          return {
            name: candidate.name,
            description: candidate.description,
            invocation: candidate.invocation,
            source: candidate.source,
            provider: 'fibbrain',
            content: stripFrontmatter(locator.text),
            metadata: { fibbrain_node_id: locator.node_id },
          }
        },
      }
    })
  }
}

// ------------------------------------------------------------------ brain facade

function createBrain(client, log) {
  const safe = async (label, fn) => {
    try {
      return await fn()
    } catch (error) {
      log.warn?.(`fibbrain: ${label} failed open: ${String(error?.message ?? error)}`)
      return undefined
    }
  }
  return {
    recall: (goal, id, opts) =>
      safe('recall', () => client.call('fibbrain_recall', { goal, ...id, ...opts })),
    advise: (action, args, id) =>
      safe('advise', () => client.call('fibbrain_advise', { action, kind: 'tool', arguments: args, ...id })),
    observe: (kind, summary, payload, id) =>
      safe('observe', () => client.call('fibbrain_observe', { kind, summary, payload, ...id })),
    review: (id, mode) =>
      safe('review', () => client.call('fibbrain_review_session', { mode, ...id })),
    render: (goal, id, min_maturity) =>
      safe('render', () => client.call('fibbrain_render', { goal, format: 'skill', top_k: 50, min_maturity, ...id })),
    client,
  }
}

// ------------------------------------------------------------------ helpers

function contextMessage(text) {
  // `form: 'recall'` is dsh's own vocabulary for memory brought back into a
  // request; the session log and UI treat it as context, not as a prompt.
  return createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: name, form: 'recall' },
  })
}

function freshUserText(messages) {
  const parts = []
  for (const message of messages ?? []) {
    if (message?.source?.kind !== 'user') continue
    for (const block of message.content ?? []) {
      if (block?.type === 'text' && block.text?.trim()) parts.push(block.text.trim())
    }
  }
  return parts.join('\n').slice(0, 2000)
}

function isOwnTool(toolName) {
  return typeof toolName === 'string' && toolName.startsWith(OWN_TOOL_PREFIX) && /__fib(brain|mind)_/.test(toolName)
}

function evidenceLine(memories) {
  const titles = (memories ?? [])
    .filter(memory => memory?.status === 'refuted')
    .map(memory => memory.title)
    .slice(0, 3)
  return titles.length ? `Refuted evidence: ${titles.join('; ')}.` : ''
}

function textOf(content) {
  if (!Array.isArray(content)) return ''
  return content
    .filter(block => block?.type === 'text' && typeof block.text === 'string')
    .map(block => block.text)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function filesOf(args) {
  const out = []
  for (const key of ['files', 'paths', 'path', 'file_path', 'file', 'filename']) {
    const value = args[key]
    if (typeof value === 'string' && value.trim()) out.push(value)
    else if (Array.isArray(value)) out.push(...value.filter(item => typeof item === 'string'))
  }
  return [...new Set(out)]
}

function pick(record, keys) {
  for (const key of keys) if (record[key] !== undefined && record[key] !== null) return record[key]
  return undefined
}

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function truncate(text, limit) {
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`
}

function stripFrontmatter(text) {
  return (text.startsWith('---') ? text.replace(/^---[\s\S]*?\n---\n/, '') : text).replace(/^\s+/, '')
}

/**
 * Schemastery is dsh's schema library. It is a peer dependency here so the
 * plugin can be unit-tested without dsh; when absent, a plain validator with
 * the same defaults stands in.
 */
async function loadSchema() {
  const defaults = {
    command: 'python',
    args: ['-m', 'fibmind.mcp_server', '--store', '.fibmind/memory.db'],
    cwd: undefined,
    env: {},
    toolCallTimeoutMs: 15_000,
    owner: undefined,
    projectId: undefined,
    workspaceId: undefined,
    hotOnStart: true,
    hotGoal: 'project conventions, preferences, and standing decisions',
    hotTopK: 3,
    hotBudgetTokens: 400,
    recall: true,
    recallTopK: 5,
    recallBudgetTokens: 500,
    advise: 'deny',
    observe: true,
    reviewMode: 'approve',
    skills: true,
    skillsMinMaturity: 'verified',
  }
  try {
    const { default: Schema } = await import('@deepseek-ai/schemastery')
    return Schema.object({
      command: Schema.string().default(defaults.command).description('Python interpreter that runs fibmind.mcp_server'),
      args: Schema.array(String).default(defaults.args),
      cwd: Schema.string(),
      env: Schema.dict(String).default({}),
      toolCallTimeoutMs: Schema.natural().default(defaults.toolCallTimeoutMs),
      owner: Schema.string().description('FibBrain owner; defaults to $USER'),
      projectId: Schema.string(),
      workspaceId: Schema.string(),
      hotOnStart: Schema.boolean().default(true),
      hotGoal: Schema.string().default(defaults.hotGoal),
      hotTopK: Schema.natural().default(defaults.hotTopK),
      hotBudgetTokens: Schema.natural().default(defaults.hotBudgetTokens),
      recall: Schema.boolean().default(true),
      recallTopK: Schema.natural().default(defaults.recallTopK),
      recallBudgetTokens: Schema.natural().default(defaults.recallBudgetTokens),
      advise: Schema.union(['deny', 'ask', 'off']).default('deny'),
      observe: Schema.boolean().default(true),
      reviewMode: Schema.union(['approve', 'auto', 'off']).default('approve'),
      skills: Schema.boolean().default(true),
      skillsMinMaturity: Schema.union(['candidate', 'verified', 'established']).default('verified'),
    })
  } catch {
    const validate = input => ({ ...defaults, ...(input ?? {}) })
    validate.defaults = defaults
    return validate
  }
}

async function loadLlm() {
  try {
    return await import('@deepseek-ai/dsh-llm')
  } catch {
    // Unit tests run without dsh; the shape below matches dsh-llm's factory.
    let counter = 0
    return {
      createUserMessage: input => Object.freeze({ ...input, id: `msg_${++counter}`, role: 'user' }),
    }
  }
}

export { loadSchema as _loadSchema }
