# Host MCP feasibility prompt

Paste this into a **new** Claude Code or Codex task in this repo. Do not continue an old session — MCP is loaded at start.

```text
You are testing whether FibBrain MCP is reachable from this host.
Do not edit files. Do not run shell commands. Use only fibbrain_* MCP tools.

If no tool named fibbrain_plan / fibbrain_remember / fibbrain_recall exists,
reply exactly: MCP_MISSING and list the tool names you can see.

Otherwise, using owner=<your-host>, workspace_id=fibmind, project_id=fibmind,
session_id=<your-host>-feasibility:

1. fibbrain_plan objective="Host MCP feasibility check"
2. fibbrain_remember category=decision title="<your-host> wrote a host-test memory"
   content="FibBrain MCP is reachable from <your-host>."
3. fibbrain_recall goal="Host MCP feasibility check"
4. fibbrain_coordinate with the goal_id from plan
5. fibbrain_complete_goal

Reply with JSON only:
{"host":"...","goal_id":"...","remember":"write|skip","recall_hit":true,"coordinate":"allow|blocked"}
```

Use `claude-code` or `codex` as `<your-host>`.
