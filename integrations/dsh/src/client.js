/**
 * A thin MCP stdio client to the Python FibBrain server.
 *
 * The plugin owns one child process for its own calls. The model-facing tools
 * can still come from a separate `@deepseek-ai/dsh-mcp-client` row pointing at
 * the same store; SQLite (WAL, BEGIN IMMEDIATE) makes two writers safe.
 *
 * Every call returns the tool's JSON payload. Failures are surfaced as thrown
 * errors; callers decide whether a hook may fail open.
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'

export class FibBrainClient {
  /**
   * @param {{ command: string, args: string[], cwd?: string, env?: Record<string,string>, timeoutMs?: number }} options
   */
  constructor(options) {
    this.options = options
    this.timeoutMs = options.timeoutMs ?? 15_000
    /** @type {Client | undefined} */
    this.client = undefined
    /** @type {Promise<Client> | undefined} */
    this.connecting = undefined
  }

  async connect() {
    if (this.client) return this.client
    if (!this.connecting) {
      this.connecting = (async () => {
        const transport = new StdioClientTransport({
          command: this.options.command,
          args: this.options.args,
          cwd: this.options.cwd,
          env: { ...scrubbedEnv(), ...(this.options.env ?? {}) },
          stderr: 'pipe',
        })
        const client = new Client({ name: 'dsh-fibbrain', version: '0.1.0' })
        await client.connect(transport)
        this.client = client
        return client
      })()
      this.connecting.catch(() => {
        this.connecting = undefined
      })
    }
    return this.connecting
  }

  /**
   * Call one FibBrain tool and return its JSON payload.
   * @param {string} name
   * @param {Record<string, unknown>} args
   * @returns {Promise<any>}
   */
  async call(name, args) {
    const client = await this.connect()
    const result = await client.callTool({ name, arguments: dropUndefined(args) }, undefined, {
      timeout: this.timeoutMs,
    })
    if (result.isError) {
      throw new Error(`fibbrain ${name} failed: ${textOf(result.content)}`)
    }
    if (result.structuredContent && typeof result.structuredContent === 'object') {
      return unwrapStructured(result.structuredContent)
    }
    const text = textOf(result.content)
    return text ? JSON.parse(text) : {}
  }

  async listTools() {
    const client = await this.connect()
    const listed = await client.listTools()
    return listed.tools.map(tool => tool.name)
  }

  async close() {
    const client = this.client
    this.client = undefined
    this.connecting = undefined
    if (client) await client.close()
  }
}

function unwrapStructured(structured) {
  // The MCP Python SDK wraps non-object returns as {"result": ...}.
  if ('result' in structured && Object.keys(structured).length === 1) return structured.result
  return structured
}

function textOf(content) {
  if (!Array.isArray(content)) return ''
  return content
    .filter(block => block && block.type === 'text' && typeof block.text === 'string')
    .map(block => block.text)
    .join('')
}

function dropUndefined(args) {
  const out = {}
  for (const [key, value] of Object.entries(args ?? {})) {
    if (value !== undefined) out[key] = value
  }
  return out
}

/** Mirror dsh-mcp-client: do not leak credential-looking or DSH_* variables. */
function scrubbedEnv() {
  const out = {}
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue
    if (key.startsWith('DSH_')) continue
    if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)/i.test(key) && !key.startsWith('FIBMIND_')) continue
    out[key] = value
  }
  return out
}
