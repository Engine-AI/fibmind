# FibMind 身份模型 Plan

上一轮判据化改造见 [`PLAN.md`](PLAN.md)，已落地。整阶段愿景见 [`ROADMAP.md`](ROADMAP.md) 的 P1。

本文是 P1 的**第一刀**：只补工作上下文身份，并让所有召回入口走同一套可见性。做完之后，`context(goal)` 才能变成后续 Brain Protocol 里 `recall(goal, state)` 的数据基础。

本轮不拆 `graph.py`，不加 embedding / FTS / token 预算，不写 DeepSeek-Harness 插件。

## 背景

当前节点只有 `scope` + `owner`。一次会话写下的 scratch、另一个项目里的决策、另一个工作区的偏好，只要 owner 对得上就会进同一次召回。

这会挡住两件事：

- DeepSeek-Harness 的 Session 无法安全挂钩；
- 图扩展已经是侧门：`search_from` 的 MCP 面不接收 owner，`context.py` 只在展开后再滤一遍。

`ROADMAP.md` P1 还列了 `branch`、`commit_sha`、`memory_kind` 和整棵目录拆分。那些不在本切片。本切片只落地四个身份字段和统一可见性。

## 原则

- 四个字段是**产地标签**，不是四道同等隔离墙。personal 记忆必须能跨 session 召回，否则长期记忆没有意义。
- 「精确匹配」沿用现有 owner 语义：`None` 只看见 `None`。调用方不传 `workspace_id`，就看不到带工作区的记忆。这是故意的，避免默认漏光。
- 可见性只放在 `FibMind.is_visible()`。检索、图扩展、自动连边、压缩禁止再写一份规则。
- 公共 API 只追加可选参数。不传时，行为退化为今天的 owner / scope 隔离。
- 日志仍是事实源。新字段进 `CREATE_NODE` 的 `node.to_dict()`，不新增 `EventOp`。

## 范围

| 做 | 不做 |
| --- | --- |
| `workspace_id` / `project_id` / `session_id` / `task_id` | `branch` / `commit_sha` / `source_uri` / `content_hash` / `valid_from` / `memory_kind` |
| 写入、检索、图扩展、压缩、自动连边共用可见性 | 一次性拆成 `domain/` `retrieval/` |
| SQLite schema v3 → v4，旧库字段为 `NULL` | 改 Fibonacci 层、新 relation |
| MCP 参数向后兼容（新字段可选） | Brain Protocol 五个函数 |
| 单测 + 评测补跨 project / 跨 session scratch | 扩真实项目轨迹、上模型打分 |

## 身份语义

```text
workspace_id   工作区 / 仓库根。有值则只在同一工作区可见
project_id     项目。有值则只在同一项目可见
session_id     这条记忆是在哪次会话写下的
task_id        这条记忆是在哪个任务写下的（只记录，默认不隔离）
```

和现有 `scope` 叠加：

```text
scope=session     scratch。必须带 session_id。
                  只有 caller.session_id == node.session_id 才可见。
                  缺一边则不可见。

scope=personal    长期个人观察。
                  owner 精确匹配（含双方都是 None）
                  + workspace 精确匹配
                  + project 精确匹配
                  session_id / task_id 只作产地，不挡跨会话召回

scope=knowledge   仍不看 owner
                  但要过 workspace / project 精确匹配
                  无 workspace 的旧知识继续对所有人可见
```

知识晋升：`promote_to_knowledge` 的新节点写入调用方传入的 workspace / project；未传则取支撑节点的交集。对不上就拒绝，禁止把 A 项目的观察晋升成 B 项目的知识。

## 可见性

把规则只放在 `FibMind.is_visible()`：

```text
status ∈ 允许集合
未折叠（除非显式要 folded）
scope 过滤（若调用方给了）
scope ≠ knowledge → owner 精确匹配
workspace_id 精确匹配
project_id 精确匹配
scope = session → session_id 精确匹配（双方都要有值；缺一边则不可见）
```

下列路径必须走同一个函数：

- `search`
- `search_from`（每一跳：不可见的邻居既不返回也不继续遍历）
- `build_context`
- `_auto_link_similar`
- `compress_overflow`（不同 workspace / project 的节点不得压进同一条摘要；不一致则跳过压缩）

`search_from` 补上与 `search` 相同的 `owner` / `scopes` / `statuses` / 身份参数。

## 数据与存储

`MemoryNode` 增加四个 `str | None`，默认 `None`。`to_dict` / `from_dict` 缺省当 `None`，旧事件重放不炸。

SQLite：`SCHEMA_VERSION = 4`。`_upgrade` 给 `nodes` 加四列 `TEXT`，并加索引：

```text
idx_nodes_workspace (workspace_id, project_id)
idx_nodes_session   (session_id)
```

JSON store 跟着 `to_dict` 走，无需新格式版本。

压缩节点继承源节点的身份；源节点身份不一致则跳过压缩。

## API

`append` / `search` / `search_from` / `context` / `promote_to_knowledge` 都增加四个可选参数。`MemoryService` 原样下传。节点摘要带上这四个字段。

MCP：`fibmind_append`、`fibmind_search`、`fibmind_search_from`、`fibmind_context`、`fibmind_promote_knowledge` 增加同名可选参数。不删现有工具、不改工具名。

`AGENTS.md` / `CLAUDE.md` 补一句：召回和写入都要带当前 `workspace_id` / `project_id` / `session_id`。

评测夹具和 runner 增加这四个可选字段。旧数据集不填，回归保持现状。

## 验收

新建 `tests/test_identity.py`，至少覆盖：

1. 同一 owner、不同 `project_id`，personal 互不可见
2. 同一 owner、不同 `workspace_id`，personal 互不可见
3. 同一 owner + 同一 project、不同 `session_id`，personal **可见**
4. `scope=session` 的 scratch，换 session 后不可见
5. 不传 `workspace_id`，看不到带 workspace 的记忆
6. knowledge 忽略 owner，但仍按 workspace / project 隔离
7. 图扩展不能绕过：A 项目节点连到 B 项目节点，在 A 的 context 里展开不到 B
8. `_auto_link_similar` 不跨 project 连边
9. `rebuild_from_log` 后四个字段还在
10. schema v3 文件打开后升到 v4，旧节点字段为 `None`，旧检索行为不变

评测：新增 `evals/datasets/cross_project.json`（或扩 `cross_session.json`）：

- 两项目各写一条相似标题的决策
- 查询带 `project_id=A` 时，B 的 id 必须进 `forbidden_ids`
- 一条 `scope=session` 的草稿，用另一个 `session_id` 查询时不得命中

验证：

```bash
.venv/bin/pytest -q
.venv/bin/python -m evals.runner
```

现有 11 案不得回退；新案隔离必须成立。

## 提交切分

1. 模型 + schema v4 + `from_dict` 兼容
2. `is_visible` + 各召回入口 + 自动连边 / 压缩
3. Service / MCP / agent 说明
4. 单测 + 评测数据集

每步只追加可选参数，可独立回滚。

## 做完之后

```text
本刀身份模型
    ↓
Brain v0（见 PLAN-brain.md）：recall / remember / observe / advise / reflect
    ↓
DSH 插件：pre-step 调 recall，turn/end 调 remember / reflect
    ↓
P2 混合检索（换引擎，不换过滤规则）
```

`ROADMAP.md` P1 的模块拆分，等 `is_visible` 稳定后再搬。
