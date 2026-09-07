import assert from 'node:assert/strict'
import { test } from 'node:test'

import { apply, Config, inject, name } from '../src/index.js'
import { gitRoot, identityFor } from '../src/identity.js'

/** A minimal Cordis-shaped context: records listeners, exposes fake services. */
function fakeContext({ skills } = {}) {
  const listeners = new Map()
  const effects = []
  const ctx = {
    logger: { info() {}, warn() {}, error() {} },
    tools: {},
    on(event, listener) {
      listeners.set(event, listener)
      return () => listeners.delete(event)
    },
    effect(run) {
      effects.push(run())
    },
    get(key) {
      return key === 'skills' ? skills : undefined
    },
  }
  return { ctx, listeners, effects }
}

/** Replace the MCP client with a scripted fake and capture calls. */
function withFakeClient(responses) {
  const calls = []
  return {
    calls,
    install(pluginModule) {
      return pluginModule
    },
    client: {
      async call(tool, args) {
        calls.push({ tool, args })
        const handler = responses[tool]
        if (typeof handler === 'function') return handler(args)
        if (handler instanceof Error) throw handler
        return handler ?? {}
      },
      async close() {},
    },
  }
}

const AGENT = {
  id: 'sess-1',
  session: { meta: { cwd: process.cwd() } },
  injected: [],
  inject(message) {
    this.injected.push(message)
  },
}

function config(overrides = {}) {
  const base = typeof Config === 'function' && Config.defaults ? { ...Config.defaults } : Config({})
  return { ...base, ...overrides }
}

test('plugin metadata', () => {
  assert.equal(name, 'fibbrain')
  assert.deepEqual(inject, ['tools'])
})

test('identity maps cwd to the git root and defaults project to workspace', () => {
  const root = gitRoot(process.cwd())
  assert.ok(root, 'tests run inside the repository')
  const id = identityFor({ owner: 'iroha' }, { cwd: process.cwd(), sessionId: 's1' })
  assert.equal(id.owner, 'iroha')
  assert.equal(id.workspace_id, 'fibmind')
  assert.equal(id.project_id, 'fibmind')
  assert.equal(id.session_id, 's1')
  const explicit = identityFor({ projectId: 'gateway', workspaceId: 'mono' }, { cwd: '/nowhere', sessionId: 's2' })
  assert.equal(explicit.workspace_id, 'mono')
  assert.equal(explicit.project_id, 'gateway')
})

test('pre-step appends a recall pack after the downstream decision', async () => {
  const fake = withFakeClient({
    fibbrain_recall: args => ({ text: `- [decision] Retry: cap backoff (goal=${args.goal})`, budget: { used_tokens: 12 } }),
  })
  const { ctx, listeners } = fakeContext()
  apply(ctx, config(), fake.client)
  const prompt = { id: 'm1', role: 'user', content: [{ type: 'text', text: 'fix the upload retry' }], source: { kind: 'user' } }
  const decision = await listeners.get('agent/pre-step')(
    { agent: AGENT, messages: [prompt], turn: 1, step: 1, signal: new AbortController().signal },
    async () => ({ kind: 'enter', messages: [prompt] }),
  )
  assert.equal(decision.kind, 'enter')
  assert.equal(decision.messages.length, 2)
  assert.equal(decision.messages[0], prompt)
  assert.match(decision.messages[1].content[0].text, /<fibbrain-recall goal="fix the upload retry" tokens="12">/)
  assert.equal(decision.messages[1].source.kind, 'plugin')
  assert.equal(decision.messages[1].source.form, 'recall')
  const recall = fake.calls.find(call => call.tool === 'fibbrain_recall')
  assert.equal(recall.args.goal, 'fix the upload retry')
  assert.equal(recall.args.session_id, 'sess-1')
  assert.equal(recall.args.workspace_id, 'fibmind')
  assert.equal(recall.args.include_hot, false)
})

test('pre-step leaves tool continuations and rejections alone', async () => {
  const fake = withFakeClient({})
  const { ctx, listeners } = fakeContext()
  apply(ctx, config(), fake.client)
  const listener = listeners.get('agent/pre-step')
  const empty = await listener({ agent: AGENT, messages: [], turn: 1, step: 2 }, async () => ({ kind: 'enter', messages: [] }))
  assert.deepEqual(empty, { kind: 'enter', messages: [] })
  const rejected = await listener({ agent: AGENT, messages: [] }, async () => ({ kind: 'reject' }))
  assert.deepEqual(rejected, { kind: 'reject' })
  assert.equal(fake.calls.length, 0)
})

test('pre-step fails open when the brain is unreachable', async () => {
  const fake = withFakeClient({ fibbrain_recall: new Error('spawn ENOENT') })
  const { ctx, listeners } = fakeContext()
  apply(ctx, config(), fake.client)
  const prompt = { id: 'm1', role: 'user', content: [{ type: 'text', text: 'hello' }], source: { kind: 'user' } }
  const decision = await listeners.get('agent/pre-step')({ agent: AGENT, messages: [prompt] }, async () => ({ kind: 'enter', messages: [prompt] }))
  assert.deepEqual(decision, { kind: 'enter', messages: [prompt] })
})

test('pre-execute denies on refuted evidence, allows otherwise, skips own tools', async () => {
  const fake = withFakeClient({
    fibbrain_advise: args =>
      args.action === 'rm' && args.arguments?.path === 'build/'
        ? { verdict: 'reject', reason: 'refuted evidence advises against rm with build', memories: [{ status: 'refuted', title: 'rm build deletes wheels' }] }
        : { verdict: 'allow', reason: 'no refuted evidence', memories: [] },
  })
  const { ctx, listeners } = fakeContext()
  apply(ctx, config(), fake.client)
  const listener = listeners.get('tools/pre-execute')
  const next = async () => ({ kind: 'allow' })

  const denied = await listener({ name: 'rm', arguments: { path: 'build/' }, agent: AGENT }, next)
  assert.equal(denied.kind, 'deny')
  assert.match(denied.reason, /FibBrain: refuted evidence/)
  assert.match(denied.reason, /rm build deletes wheels/)

  const allowed = await listener({ name: 'rm', arguments: { path: 'tmp/' }, agent: AGENT }, next)
  assert.deepEqual(allowed, { kind: 'allow' })

  const own = await listener({ name: 'mcp__fibbrain__fibbrain_recall', arguments: {}, agent: AGENT }, next)
  assert.deepEqual(own, { kind: 'allow' })
  assert.equal(fake.calls.filter(call => call.tool === 'fibbrain_advise').length, 2)
})

test('pre-execute can ask instead of deny, or be switched off', async () => {
  const fake = withFakeClient({ fibbrain_advise: { verdict: 'reject', reason: 'r', memories: [] } })
  const { ctx, listeners } = fakeContext()
  apply(ctx, config({ advise: 'ask' }), fake.client)
  const asked = await listeners.get('tools/pre-execute')({ name: 'rm', arguments: {}, agent: AGENT }, async () => ({ kind: 'allow' }))
  assert.equal(asked.kind, 'ask')

  const off = fakeContext()
  apply(off.ctx, config({ advise: 'off' }), fake.client)
  const passed = await off.listeners.get('tools/pre-execute')({ name: 'rm', arguments: {}, agent: AGENT }, async () => ({ kind: 'allow' }))
  assert.deepEqual(passed, { kind: 'allow' })
})

test('tools/result observes with command and files, errors as kind=error', async () => {
  const fake = withFakeClient({ fibbrain_observe: {} })
  const { ctx, listeners } = fakeContext()
  apply(ctx, config(), fake.client)
  const listener = listeners.get('tools/result')
  listener(
    { name: 'bash', arguments: { command: 'pytest -q' }, agent: AGENT },
    { isError: false, content: [{ type: 'text', text: '199 passed' }] },
  )
  listener(
    { name: 'edit_file', arguments: { path: 'src/x.py' }, agent: AGENT },
    { isError: true, error: { code: 'ENOENT', name: 'ToolFailure' }, content: [{ type: 'text', text: 'no such file' }] },
  )
  listener({ name: 'mcp__fibbrain__fibbrain_observe', arguments: {}, agent: AGENT }, { isError: false, content: [] })
  await new Promise(resolve => setTimeout(resolve, 0))
  const observed = fake.calls.filter(call => call.tool === 'fibbrain_observe')
  assert.equal(observed.length, 2)
  assert.equal(observed[0].args.kind, 'tool_result')
  assert.equal(observed[0].args.payload.command, 'pytest -q')
  assert.match(observed[0].args.summary, /^bash: 199 passed/)
  assert.equal(observed[1].args.kind, 'error')
  assert.deepEqual(observed[1].args.payload.files, ['src/x.py'])
  assert.equal(observed[1].args.payload.error, 'ENOENT')
})

test('turn-stopping reviews the session in the configured mode and invalidates skills', async () => {
  let invalidated = 0
  const skills = { registerProvider(create) { create({ invalidate: () => invalidated++, signal: new AbortController().signal }) } }
  const fake = withFakeClient({ fibbrain_review_session: { written: [{ node_id: 'n1' }], skipped: [] } })
  const { ctx, listeners } = fakeContext({ skills })
  apply(ctx, config({ reviewMode: 'auto' }), fake.client)
  await listeners.get('agent/turn-stopping')({ agent: AGENT, turn: 1 })
  const review = fake.calls.find(call => call.tool === 'fibbrain_review_session')
  assert.equal(review.args.mode, 'auto')
  assert.equal(review.args.session_id, 'sess-1')
  assert.equal(invalidated, 1)
})

test('session-start injects the hot preamble once', async () => {
  const fake = withFakeClient({ fibbrain_recall: { text: '- [preference] Editor: PyCharm', budget: {} } })
  const { ctx, listeners } = fakeContext()
  apply(ctx, config(), fake.client)
  const agent = { ...AGENT, injected: [], inject(m) { this.injected.push(m) } }
  listeners.get('agent/session-start')({ agent, source: 'startup' })
  await new Promise(resolve => setTimeout(resolve, 0))
  assert.equal(agent.injected.length, 1)
  assert.match(agent.injected[0].content[0].text, /<fibbrain-hot source="startup">/)
  const recall = fake.calls.find(call => call.tool === 'fibbrain_recall')
  assert.equal(recall.args.include_hot, true)
})

test('skills provider lists rendered procedures and serves their bodies', async () => {
  let provider
  const skills = { registerProvider(create) { provider = create({ invalidate() {}, signal: new AbortController().signal }) } }
  const fake = withFakeClient({
    fibbrain_render: args => ({
      items: [{ name: 'run-the-regression', title: 'Run the regression', maturity: 'verified', node_id: 'n9', text: '---\nname: run-the-regression\ndescription: "x"\n---\n\n# Run the regression\n\n1. pytest\n' }],
      goal: args.goal,
    }),
  })
  const { ctx } = fakeContext({ skills })
  apply(ctx, config(), fake.client)
  const listed = await provider.list({ cwd: process.cwd() })
  assert.equal(listed.length, 1)
  assert.equal(listed[0].name, 'run-the-regression')
  assert.equal(listed[0].provider, 'fibbrain')
  const render = fake.calls.find(call => call.tool === 'fibbrain_render')
  assert.equal(render.args.goal, '*')
  assert.equal(render.args.min_maturity, 'verified')
  assert.equal(render.args.workspace_id, 'fibmind')
  const body = await provider.get(listed[0], {})
  assert.match(body.content, /^# Run the regression/)
  assert.equal(body.metadata.fibbrain_node_id, 'n9')
})
