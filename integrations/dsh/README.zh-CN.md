# @fibmind/dsh-fibbrain — 以原生 dsh 插件形式运行的 FibBrain

[English](README.md) | 简体中文

一个 Cordis 插件（约 350 行纯 ESM JavaScript，无需构建），通过 harness 自身的生命周期事件把 FibBrain 挂载进 [DeepSeek-Harness](https://github.com/deepseek-ai/deepseek-harness)。Python 的 FibBrain 服务端仍然是大脑；这里只是 [`docs/plans/04-evolve.md`](../../docs/plans/04-evolve.md)（S6）中计划好的那层薄壳。

| dsh 事件 / 服务          | 插件做什么                                                                                  | FibBrain 调用               |
| ------------------------ | ------------------------------------------------------------------------------------------- | --------------------------- |
| `agent/session-start`    | 把当前 session 冻结的 hot memory 作为 `<fibbrain-hot>` 上下文消息注入                        | `fibbrain_recall(include_hot)` |
| `agent/pre-step`         | 对每个新的用户 prompt，向该 step 追加一个受 token 预算约束的 `<fibbrain-recall>` 包           | `fibbrain_recall`           |
| `tools/pre-execute`      | 询问该工具调用是否应执行；存在驳倒证据 → `deny`（或 `ask`）                                  | `fibbrain_advise(arguments)` |
| `tools/result`           | 把每次工具执行结果（命令、文件、错误）记录进 session episode                                 | `fibbrain_observe`          |
| `agent/turn-stopping`    | 把 episode 提炼为待审（或 active）记忆；幂等                                                 | `fibbrain_review_session`   |
| `ctx.skills` provider    | 把已验证的 procedure 发布为 skill 目录；每次 review 后刷新                                   | `fibbrain_render("*")`      |

除 `advise` 之外，所有环节都 fail **open**（出错即放行）：如果 Python 进程挂了，agent 会在没有记忆的情况下继续工作，插件只记一条警告。`advise` fail open 意味着“允许”，因此它的拒绝是这个插件改变 agent 行为的唯一途径。

## 身份

dsh 没有 owner / project 的概念，所以插件做如下映射：

- `owner` ← `config.owner`，否则 `$USER`，否则 `dsh`
- `workspace_id` ← 包含 session cwd 的 git 根目录的 basename
- `project_id` ← `config.projectId`，否则取 `workspace_id`
- `session_id` ← dsh 的 session id

正是这一点保证了一个项目的记忆不会出现在另一个项目的 recall 里。

## 启用

```sh
cd /path/to/fibmind
uv sync --locked --extra dev            # Python 端
npx @deepseek-ai/dsh web --patch "$PWD/integrations/dsh/fibbrain.cordis.yml"
```

把 `fibbrain.cordis.yml` 中的 `/path/to/fibmind` 替换为你的 checkout 路径。插件的 `name` 和 `cwd` 必须是绝对路径，因为 dsh 的加载器以 profile 目录而不是 overlay 文件为基准解析相对路径；`command` 和存储路径则以 `cwd` 为基准解析。

插件依赖 `@modelcontextprotocol/sdk`（随 dsh 一起已经装好），并从 dsh 的安装目录导入 `@deepseek-ai/dsh-llm` / `@deepseek-ai/schemastery`。如果从一个模块解析结果不同的 checkout 运行，请把它们装在插件旁边，或者给 `node_modules` 建一个符号链接。

## 在 dsh 中验证

1. 启动一个 session。第一个请求里应该有一条 user 角色的 `<fibbrain-hot>` 上下文消息（在 Trajectory 视图中查看）。新存储下它是空的——这是正常的。
2. 输入 `Remember that the validation drink is lapsang-<suffix>`，让模型调用 `mcp__fibbrain__fibbrain_remember`（mcp-client 那一条挂载项）——或者直接干活；`tools/result` hook 一直在观察。
3. 打开一个**新的** session，随便问点什么。它的第一个请求现在会带上一条包含该记忆的 `<fibbrain-recall>` 消息，这是插件加的，不是模型。
4. 对一条提到某个工具的记忆执行 `fibbrain_reflect(verdict="refuted", ...)`，然后让模型用匹配的参数调用那个工具：调用会被拒绝，并附上原因。
5. 一个 turn 结束后，`fibbrain_pending_reviews` 会列出 review 提取出的内容。

## 脱离 dsh 测试

```sh
cd integrations/dsh
node --test test/plugin.test.js
```

测试使用一个 Cordis 形态的伪造上下文和一个脚本化的客户端，所以任何能跑 Node 22 的地方都能运行。端到端路径（`src/client.js` 对接真实的 Python 服务端）由仓库的 Python 测试套件通过它所调用的 MCP 接口覆盖。

## 配置

| 配置项                 | 默认值                                             | 含义                                                   |
| ---------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| `command`, `args`, `cwd`, `env` | `python -m fibmind.mcp_server --store .fibmind/memory.db` | 如何启动 FibBrain 服务端                        |
| `owner`, `projectId`, `workspaceId` | 自动推导                                  | 覆盖身份映射                                           |
| `hotOnStart`, `hotBudgetTokens`, `hotTopK` | `true`, 400, 3                  | session 启动时的前置注入                               |
| `recall`, `recallTopK`, `recallBudgetTokens` | `true`, 5, 500                | 每个 prompt 的 recall                                  |
| `advise`               | `deny`                                             | `deny` / `ask` / `off`                                 |
| `observe`              | `true`                                             | 记录工具结果                                           |
| `reviewMode`           | `approve`                                          | `approve` / `auto` / `off`                             |
| `skills`, `skillsMinMaturity` | `true`, `verified`                          | 把 procedure 发布为 skill                              |

## 边界

- dsh 的 `ctx.goals` 是当前 session 的目标；FibBrain 的 `plan` 目标是跨 session 的历史。插件不在两者之间同步。
- 工具调用前的**参数改写**不是 dsh 的扩展点；插件只能拒绝或询问，不能修改。
- 插件不执行 procedure。`render` 只产出 skill 文本；由 harness 的 skill 加载器把它展示给模型。
