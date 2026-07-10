# FibMind

FibMind is a Python prototype for a Fibonacci Memory Graph.

It models long-term AI memory as:

- multiple category trees, forming a memory forest
- Fibonacci-sized layers for raw, compressed, summary, and long-term memory
- graph edges between related nodes
- node promotion, tree expansion, tree merge, tree split, and local graph search

This project is intentionally dependency-free for the first prototype.

## Quick Start

```bash
cd /Users/abbila/PycharmCompany/fibmind
python3 examples/demo.py
python3 -m unittest
```

## Core Concepts

```text
Node        A memory unit. It may be raw data, compressed data, summary, concept, root, or archive.
Edge        A typed relationship between two nodes.
Tree        A category or topic tree whose root is also a normal node.
Forest      Many trees connected by cross-tree graph edges.
Layer       A compression level controlled by Fibonacci-sized capacity limits.
```

## Default Layers

```text
raw         21 memory weight
compressed 34 memory weight
summary    55 memory weight
long_term  89 memory weight
```

Layer capacity is measured as memory weight. A raw node has weight `1`; a
compressed or summary node carries the total weight of the source nodes it
covers.

## Main Operations

```python
memory = FibMind()
tree_id = memory.create_tree("code", title="Code Tree")
node_id = memory.append("code", title="Login bug", content="401 after token expiry")

memory.link_nodes(node_id, other_node_id, RelationType.RELATED_TO)
memory.promote_node(node_id)
results = memory.search_from(node_id, depth=2)
activated = memory.search_from(node_id, depth=2, reinforce=True)
```

`search_from()` is read-only by default. Pass `reinforce=True` when a query
should behave like memory activation and increase access/importance signals.

`search()` is the query-based counterpart: it ranks the whole forest by keyword
relevance to a text query and never mutates nodes.

## MCP Server (long-term memory for agents)

FibMind can run as an [MCP](https://code.claude.com/docs/en/mcp) server so tools
like Claude Code and Codex can use it as external long-term memory — no built-in
integration required. The core package stays dependency-free; the server layer
depends on the `mcp` SDK.

```bash
# install the server extra into the project venv
uv pip install -e ".[mcp]"

# run it directly (stdio transport)
python -m fibmind.mcp_server --store .fibmind/memory.json
```

### v1 tools

| Tool | Purpose |
| --- | --- |
| `fibmind_append` | Save one memory (task result, error, requirement, plan, code summary). |
| `fibmind_search` | Rank memories by keyword relevance to a query (read-only). |
| `fibmind_search_from` | Expand the association tree rooted at a node. |
| `fibmind_context` | Build a compact, char-bounded memory pack for a task goal. |

The two you will call most: `fibmind_context` before starting a task (recall),
and `fibmind_append` after finishing one (record).

### Register with Claude Code

```bash
claude mcp add --scope project --transport stdio fibmind -- \
  /Users/abbila/PycharmCompany/fibmind/.venv/bin/python \
  -m fibmind.mcp_server \
  --store .fibmind/memory.json
```

Then tell the agent when to use it via `CLAUDE.md` (Claude Code) or `AGENTS.md`
(Codex) — both are included in this repo.

### Architecture

```text
Codex / Claude Code
        │ MCP tools
        ▼
FibMind MCP Server        (src/fibmind/mcp_server.py — thin FastMCP wiring)
        ▼
MemoryService             (src/fibmind/service.py — load/lock/persist)
        ▼
FibMind Core + JsonStore  (graph, ranking, context, storage)
```
