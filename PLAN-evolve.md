# FibBrain 进化 Plan

上一刀 Brain v1 见 [`PLAN-brain.md`](PLAN-brain.md)，代码已落地但**尚未提交**。整阶段愿景见 [`ROADMAP.md`](ROADMAP.md)。

本文回答一个问题：目标是"有记忆、有知识、能按任务渲染 tools / skills、可自动进化、可挂在 DeepSeek-Harness（DSH）上"的 fib-brain，接下来按什么顺序做。

不重写成 TypeScript。Python 内核 + MCP 协议是脑，DSH 侧只写一层薄壳。

## 现状（2026-09）

已有：事件日志事实源、familiarity / confidence 分离、scope + status + 五级身份的统一可见性、admit 门、plan / recall / coordinate / advise / reflect / remember / observe 七个动作、Claude Code 和 Codex 两个宿主验证、P0 评测（15 案，R@K 1.0，越权 0，过期污染 0）。

缺口按阻碍程度排序：

| # | 缺口 | 挡住的目标 |
| --- | --- | --- |
| 1 | 没有程序性记忆，没有东西能渲染成 skill / tool | 按任务渲染 |
| 2 | `observe` 的 episode 是进程内 deque，重启即丢；没有 session review | 自动进化的输入 |
| 3 | 检索是关键词全量扫描 | 库增长后的召回质量与延迟 |
| 4 | 字符截断，无 Token 预算，无 L0 热记忆 | DSH 每步注入 |
| 5 | reflect 结果只改单条排序，不回流到规划 / 门槛 / 权重 | 自动进化 |
| 6 | `_summarize` 是拼接，plan 是正则启发式 | 有知识 |
| 7 | 只有 stdio MCP，没有原生 DSH 插件 | 每步自动召回、动作拦截 |

## 原则

- **先接真实轨迹，再优化算法。** ROADMAP 的 P0 至今没有真实项目数据；DSH 零代码接入是最便宜的数据来源。
- **Brain 接口冻结，实现可换。** 七个 `fibbrain_*` 动作的名字和参数不改；检索引擎、总结器、规划器都在接口后面换。
- **每一刀都要过评测门。** `pytest` 全绿且 `evals.runner` 不回退是合并条件，自动调参尤其如此。
- **进化 = 结果统计改变下一次行为。** 没有 `reflect` 数据回流的功能不算进化，只算功能。

## 切片

```text
S0 收口与契约        →  S1 DSH 零代码接入
                              ↓
S2 持久 episode + Session Review（P4 前半）
                              ↓
S3 程序性记忆 + skill / tool 渲染
                              ↓
S4 混合检索（P2）    →  S5 Token 预算 + L0 热记忆（P3）
                              ↓
S6 DSH 原生 Cordis 插件
                              ↓
S7 进化闭环 + 可插拔 LLM 蒸馏 / 规划
```

S4 可与 S2 / S3 并行，它不改 Brain 接口。

---

### S0 收口与契约

**做**

- 提交当前工作树（Brain v1 的 `brain.py` / `planning.py` / `admission.py` / 测试 / 文档）。1200 行未提交改动不能再拖。
- 给事件日志格式加 `log_version`，写进 `MemoryEvent.to_dict()` 与 SQLite `meta`。
- 给 MCP 工具 schema 写一份 `docs/contract.md`：七个 `fibbrain_*` 的参数、返回字段、稳定性承诺。
- 把 `data/demo-brain.json` 固化为一致性夹具：任何实现从它 `rebuild_from_log` 后，对固定查询集的检索结果必须逐条相同。这是将来任何语言移植的验收标准。

**不做**：拆 `graph.py` 目录结构。

**验收**：`tests/test_contract.py` 通过；`git status` 干净。

---

### S1 DSH 零代码接入

**做**

- 新增 `examples/dsh/fibbrain.cordis.yml`：一条 `@deepseek-ai/dsh-mcp-client` stdio 行，指向本仓库 venv 的 `fibmind.mcp_server`。
- 新增 `examples/dsh/README.md`：安装 dsh、`--patch` 启用、按 dsh 官方记忆指南的三步验证（session A 写，session B 召回，session B 使用）。
- `CLAUDE.md` / `AGENTS.md` 的调用约定复制一份为 dsh 的 model instruction 片段。
- 把 dsh 跑出来的会话日志（JSONL）导出为 P0 数据集候选：新增 `evals/import_dsh_session.py`，从 `tool/call` / `tool/result` 里抽 `fibbrain_*` 调用和最终答案。

**不做**：任何 TS 代码；改 MCP 工具名。

**验收**

- 三步验证在本机走通，`.fibmind/memory.db` 里出现 `session_id` 为 dsh session id 的节点。
- 至少一条真实 dsh 轨迹进入 `evals/datasets/real_dsh.json`，`evals.runner` 可跑。

**注意**：dsh 只桥接 Tools；resources / prompts 不会出现。调用时机由模型决定，这是 S6 要解决的。

---

### S2 持久 episode + Session Review

对应 ROADMAP P4 前半。

**做**

- `observe` 落盘：写成 `scope=session`、`category=episode` 的节点，复用现有 session 可见性（只有同 session 可见）。进程内 deque 退化为缓存。
- 新增 `fibbrain_review_session(session_id, mode)`：
  - 从该 session 的 episode 节点和 goal 节点做**确定性抽取**：objective、changed files（来自 `observe` payload 的 `files`）、tests（`kind=test` 的 observe）、errors、user corrections、unresolved risks；
  - 每条候选走 `admit`；
  - `mode` ∈ `auto` / `approve` / `candidates`。`approve` 模式写入 `status=pending` 的节点，新增 `fibbrain_approve` / `fibbrain_reject`。
- 幂等：候选 fingerprint 已存在则跳过；同一 session 重复 review 不产生重复节点。
- Session 结束后把 episode 节点标 `stale`，保留溯源。

**不做**：用模型做抽取（S7 再接）。

**验收**

- `observe` → 进程重启 → 同 session 的 `recall` 仍能看到 episode。
- 同一 session 连跑两次 `review_session`，节点数不变。
- `approve` 模式下未批准的候选不出现在 `recall`。
- 新增 P0 案：review 抽出的记忆，在下一 session 的查询里 Recall@K 命中。

---

### S3 程序性记忆 + 渲染

这是目前完全空白、也是"按任务渲染 tools / skills"的唯一路径。

**数据模型**

- `MemoryNode` 增加 `memory_kind`（ROADMAP P1 已列）：`preference / decision / error / requirement / summary / knowledge / procedure`，默认按 category 推断，旧节点为 `None`。SQLite v4 → v5。
- `procedure` 节点的 content 是结构化文档（同 goal 的做法）：

```json
{
  "kind": "fibbrain_procedure",
  "trigger": "跑 fibmind 的回归验证",
  "when": ["pytest", "评测", "回归"],
  "steps": [".venv/bin/pytest -q", ".venv/bin/python -m evals.runner"],
  "tools": ["shell"],
  "verify": "142 passed 且 fibmind_current 不回退",
  "outcomes": {"confirmed": 0, "refuted": 0}
}
```

**动作**

- `fibbrain_render(goal, state, format)`：按 goal 检索 `procedure` 节点（走同一 `is_visible`），渲染为：
  - `skill`：SKILL.md 风格的 markdown（名字、描述、步骤、验证）；
  - `tool`：JSON Schema 工具定义（name / description / parameters / steps 作为 execute 提示）。
- `coordinate` 改为：先查匹配的 procedure，有则按其 `steps` / `tools` 给能力清单，没有才退回现在的六个固定 CAPABILITIES。
- `reflect` 作用在 procedure 节点时同时更新 `outcomes` 计数。
- 从 S2 的 review 里自动生成 procedure 候选：一个 session 里"目标 → 一串工具调用 → 验证通过"的序列，抽成一条 `procedure`，初始 confidence 0。

**进化规则（确定性）**

- `confirmed ≥ 3` 且来自 ≥ 2 个不同 session → 可 `promote_to_knowledge`，成为 workspace 内共享 skill。
- `refuted ≥ 2` 且 `confirmed = 0` → 自动 `mark_stale`。
- 同 trigger 的新 procedure 出现时，旧的挂 `VERSION_OF` 边。

**不做**：在 Python 侧执行 procedure。执行永远是 Harness 的事。

**验收**

- `render(format=skill)` 的输出可以直接放进 `.claude/skills/<name>/SKILL.md` 并被 Claude Code 列出。
- `render(format=tool)` 的输出通过 JSON Schema 校验。
- 跨 project 的 procedure 不会被渲染出来（复用身份测试）。
- 三次 confirmed 后 `coordinate` 的能力清单来源变为该 procedure。

---

### S4 混合检索（P2）

按 ROADMAP P2 执行，这里只写与本 Plan 的耦合点。

- SQLite FTS5 + BM25，中文沿用现有 bigram 分词策略；写入 / 修订 / 遗忘增量更新索引。
- `EmbeddingProvider` 接口，默认关闭；关闭时纯 FTS5。Provider 故障回退到 FTS5，不能让 `recall` 报错。
- RRF 融合，`confidence` 加权，图扩展只做补充召回。
- 新增 `fibmind_explain_recall`。
- `advise` 从标题关键词匹配改为混合检索，并对 action 参数做匹配（例如同为 `rm`，不同路径分别判断）。

**决策门**：新引擎在 P0 数据集（含 S1 导入的真实轨迹）上 Recall@K、stale、forbidden 不得低于当前关键词基线；1 万条节点 P95 < 50 ms。达不到不合并。

---

### S5 Token 预算 + L0 热记忆（P3）

- `max_chars` 升级为 `budget_tokens`，tokenizer 可插拔（默认仍是估算器，接 DeepSeek / Claude tokenizer 后校准）。
- L0 热记忆快照：session 开始时生成，session 内冻结，session 结束由 S2 的 review 更新。预算 800–1200 token，满时合并或拒绝。
- Context Planner 返回 `used_tokens` / `budget_tokens` / 每条记忆成本 / 截断原因。

**验收**：`recall` 返回不超过预算；同 session 热记忆稳定；评测用 token 而非字符比较，30% 节省阈值在真实轨迹上校准。

---

### S6 DSH 原生 Cordis 插件

薄壳，TypeScript，约 300–500 行，放在 `integrations/dsh/`。内部通过 dsh 自带的 mcp-client 调 Python 端，不自管子进程。

| dsh 事件 / 服务 | FibBrain 动作 |
| --- | --- |
| `agent/session-start` | 注入 L0 热记忆；`plan` |
| `agent/pre-step`（waterfall） | `recall`，把 pack 追加进本步消息批 |
| `tools/pre-execute`（waterfall） | `advise`；`reject` → `deny`，带原因给模型 |
| `tools/result` | `observe` |
| `agent/turn-stopping` | `review_session(mode=approve)` + `reflect` |
| `ctx.skills.registerProvider()` | `render(format=skill)` 的结果作为 skill 目录 |
| `ctx.tools.register()` | `render(format=tool)` 的结果按需注册 |

**身份映射**：dsh 只有匿名安装 id。插件从 session meta 取 `cwd` → `workspace_id`（git root）、`project_id`（可配）、`sessionId` → `session_id`，fork 血缘写进 metadata。

**与 dsh goal 的分工**：dsh `ctx.goals` 管本 session 当前目标；FibBrain goal 管跨 session 的历史与 Avoid / Reuse 证据。插件把 dsh goal id 写进 FibBrain goal 的 metadata，不双向同步。

**验收**

- 不写任何 model instruction，新 session 第一步的请求里已含 recall pack（在 dsh Trajectory 视图可见）。
- 一条被 `reflect(refuted)` 的动作，在 dsh 里被 `pre-execute` 拦下并给出原因。
- S3 渗透测试：三次 confirmed 的 procedure 在下一个 dsh session 的 `skill` 目录里出现。

---

### S7 进化闭环 + 可插拔 LLM

前六刀让数据流起来，这一刀让数据改变行为。

**回流点（全部确定性、全部有评测门）**

- `ranking` 权重：按 `reflect` 结果统计各信号（coverage / title / confidence / recency）与 confirmed 的相关性，允许在配置范围内调整；每次调整跑 `evals.runner`，回退则回滚。
- `admit` 门槛：被 `reject` / `mark_stale` 比例高的候选类别自动提高门槛。
- `plan` 启发式：S3 的 procedure 成功率替代 `_HINTS` 正则成为首选来源。
- 知识失效传播：支撑节点被 refuted → 派生知识 confidence 按比例下调；全部支撑失效 → 知识 `stale`。
- 长期未确认知识的定期重验任务（输出为候选，不自动改）。

**可插拔 LLM**

- `Summarizer` 接口替换 `_summarize`；`Planner` 接口包在 `synthesize_plan` 外。默认实现仍是现在的确定性版本，LLM 版本记录模型、prompt 版本、输入来源，可从事件重新生成。
- 蒸馏输出仍要过 `promote_to_knowledge` 的三证据门。

**可观测**：`fibmind_status`：存储规模、索引状态、待审批数、失效知识数、召回命中 / 空召回、自动写入接受率、最近一次评测结果。

**安全**：写入前扫密钥 / token / 私钥、prompt injection 模式、隐形 Unicode。这是接 DSH 真实流量前的必要条件，实际可提前到 S6 之前做。

**验收**

- 一次人工注入的错误权重，经过 N 次 reflect 后被自动纠正，且评测门全程未放行回退。
- 支撑节点 refuted 后，`recall` 不再返回其派生知识。
- 用 LLM Summarizer 重放日志得到的摘要节点，与确定性版本在同一层，且 `folded_into` 溯源不变。

---

## 里程碑

| 版本 | 范围 | 发布条件 |
| --- | --- | --- |
| v0.2 Brain on DSH | S0–S3 | dsh 零代码可用；session review 幂等；能渲染出至少一个被真实任务验证的 skill |
| v0.3 Retrieval MVP | S4–S5 | 混合检索超关键词基线；Token 预算严格执行；真实轨迹上校准节省 |
| v0.4 Native Brain | S6 | dsh 原生插件每步召回、拦截、渲染 skill 全部走通 |
| v1.0 Evolving Brain | S7 | 至少一次自动调参被评测门放行并保留；知识失效传播可演示 |

## 执行记录（2026-09）

| 切片 | 提交 | 结果 |
| --- | --- | --- |
| S0 | 75755e3 | `log_version`、`docs/contract.md`、一致性夹具 |
| S1 | 943a30f | dsh 零代码覆盖层、会话日志导入器；未在本机跑真实 dsh（无 API key） |
| S2 | 7f46511 | 持久 episode、`review_session` 三模式幂等、pending 审批 |
| S3 | a93be4d | `memory_kind` / procedure / `render`，coordinate 优先程序 |
| S4 | 2b6c03d | 倒排索引 + BM25 + 可选向量 + RRF；1 万节点 P95 48 ms；explain |
| S5 | 732f1c7 | token 预算分段渲染、L0 热记忆冻结 |
| S6 | 5fbfa22 | 原生 Cordis 插件（JS，11 个单测，真实 Python 端到端） |
| S7 | 本次 | 安全扫描、知识失效传播、可插拔蒸馏 / 规划、评测门调参、重验候选 |

v1.0 的两条验收都有测试：注入错误权重后 `tune()` 经真实评测门逐步纠正
（`test_a_wrong_weight_is_corrected_by_evidence_with_the_real_gate`）；支撑全部
refuted 后派生知识退出 recall（`test_all_supporters_refuted_retires_the_claim_from_recall`）。

## 之后

- 用真实 dsh 会话（S1 导入器）替代固定小数据集校准 token 节省阈值与调参步长。
- 向量缓存持久化；FTS5 仅在内存索引不够时再考虑。
- admit 门槛自动收紧：现在只报告 `admission_pressure`，是否让它改行为要看真实拒绝率。
- 语义近重复与冲突版本自动识别（`VERSION_OF`）。
