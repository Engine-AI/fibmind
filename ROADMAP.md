# FibMind 后续路线图

## 目标

FibMind 后续开发聚焦四个可验证目标：

1. 减少 Agent 获取历史上下文所需的 Token；
2. 提高跨 Session、跨任务的上下文召回准确率；
3. 自动总结并保存值得长期保留的记忆；
4. 将多次验证过的记忆沉淀为可追溯、可修订的知识。

实施顺序遵循“先证明价值，再扩展能力”：

```text
P0 评测基线
  ↓
P1 架构和身份模型
  ↓
P2 混合检索
  ↓
P3 Token 预算和双层记忆
  ↓
发布 v0.2 Retrieval MVP，并根据评测决定是否继续
  ↓
P4 自动记忆闭环
  ↓
P5 总结、知识治理和安全
```

前三阶段完成后形成第一个可验证价值的 MVP。P3 是继续投入的决策门：如果 FibMind 不能在评测中同时保持或提高准确率并减少 Token，就不继续扩展非核心功能。

## 实施状态

截至 2026-08-03，P0 的第一版评测基础设施已经落地：

- [x] 固定 JSON 数据集格式和校验；
- [x] 无记忆、`AGENTS.md`、Hermes 热记忆和当前 FibMind 四类基线；
- [x] Recall@K、MRR、上下文可回答率、无关/过期污染、越权命中、估算 Token 和延迟指标；
- [x] 命令行表格与完整 JSON 报告；
- [x] 项目历史、过期记忆、冲突记忆和跨 Session 四类示例数据；
- [ ] 使用真实项目轨迹扩充数据集；
- [ ] 接入模型回答评测，取代当前的上下文可回答率代理指标；
- [ ] P3 使用目标模型 tokenizer 后校准 30% Token 节省阈值。

首轮基线随后驱动了一个 P1/P2 治理切片：新增 `active/stale/refuted`
生命周期状态、默认排除非 active 记忆、严格 owner 匹配，以及英文停用词过滤。
在 15 个固定场景中（含跨 project / session scratch 隔离），当前 FibMind
保持 Recall@K 1.0，同时过期污染率和禁止记忆命中率从非零降为 0，无相关
查询正确率从 0 提升到 1.0。这里的示例集很小，结果用于回归而非宣称生产效果。

## P0：建立评测体系

这是最高优先级。没有评测，就无法证明 FibMind 比 `AGENTS.md`、固定热记忆或其他记忆实现更有效。

### 新增结构

```text
evals/
├── datasets/
│   ├── project_history.json
│   ├── stale_memory.json
│   ├── conflicting_memory.json
│   └── cross_session.json
├── runner.py
├── metrics.py
└── baselines.py
```

### 评测场景

- 跨 Session 找回历史设计决策；
- 找到某次 Bug 的原因和验证命令；
- 新旧记忆冲突时不返回过期内容；
- 不同 owner 的个人记忆不能串数据；
- 无相关记忆时不应强行召回；
- 长期积累后仍能找到低频但重要的信息。

### 指标

- `Recall@K`；
- `MRR`；
- 最终答案正确率；
- 无关记忆注入率；
- 过期记忆污染率；
- 上下文输入 Token；
- 查询延迟 P50/P95。

### 对比基线

1. 不使用记忆；
2. 只使用 `AGENTS.md`；
3. Hermes 风格固定热记忆；
4. 当前 FibMind 关键词检索；
5. 优化后的 FibMind。

### 验收

- 评测可以固定数据、固定参数重复执行；
- 报告能同时展示准确率、污染率、Token 和延迟；
- 初始建议目标：最终答案正确率不下降时，动态记忆上下文 Token 减少至少 30%。该阈值应在获得第一轮基线数据后校准。

## P1：调整项目结构和身份模型

第一刀只做身份字段和统一可见性，实施计划见 [`PLAN-identity.md`](PLAN-identity.md)。模块拆分等 `is_visible` 稳定后再搬，不要和身份规则搅在一次改动里。

当前 `graph.py` 同时负责写入、生命周期、压缩、图遍历、权限过滤、知识晋升和事件回放。继续加入检索和自动总结后会难以维护，因此先拆分边界。

### 目标结构

```text
src/fibmind/
├── domain/
│   ├── models.py
│   ├── events.py
│   └── policies.py
├── ingestion/
│   ├── admission.py
│   ├── dedup.py
│   └── security.py
├── retrieval/
│   ├── lexical.py
│   ├── semantic.py
│   ├── fusion.py
│   └── graph_expand.py
├── governance/
│   ├── evidence.py
│   ├── lifecycle.py
│   └── knowledge.py
├── context/
│   ├── hot_memory.py
│   ├── planner.py
│   └── renderer.py
├── storage/
│   ├── sqlite.py
│   └── repository.py
├── service.py
└── mcp_server.py
```

拆分应分步进行，保持现有公共 API 和测试持续可用，不需要一次性搬完所有模块。

### 数据模型扩展

- `workspace_id`；
- `project_id`；
- `session_id`；
- `task_id`；
- `branch`；
- `commit_sha`；
- `source_uri`；
- `content_hash`；
- `valid_from` / `valid_until`；
- `status`: `active` / `stale` / `refuted` / `forgotten`；
- `memory_kind`: `preference` / `decision` / `error` / `requirement` / `summary` / `knowledge`。

### 隔离规则

- 没有显式 owner 时不得默认读取所有 personal memory；
- 项目、workspace 和 owner 过滤必须在检索入口统一执行；
- 图扩展不得绕过与直接检索相同的可见性规则。

### 验收

- 检索、治理、上下文和存储可以独立替换；
- `graph.py` 不再承担所有业务职责；
- 项目、Session、分支和用户隔离有完整测试；
- 现有 MCP 工具保持兼容。

## P2：实现混合检索

> 状态（2026-09）：已落地一版。`retrieval.py` 用倒排索引 + BM25 替代全量扫描，
> 可选 `EmbeddingProvider`（离线 hashing / OpenAI 兼容）经 RRF 融合，
> `fibmind_explain_recall` 输出每条结果的信号与排除原因。1 万节点 P95 约 48 ms，
> 固定评测 Recall@K 1.0 / 过期污染 0 / 越权 0 不变。FTS5 表未采用：内存倒排索引
> 已达标，且不必在 SQLite 与 JSON 两种后端间维护两套索引；向量缓存持久化留待后续。

当前全量关键词遍历适合原型，不适合持续增长的长期记忆库。目标是形成“词法召回 + 语义召回 + 元数据过滤 + 证据排序”的可插拔检索链路。

### P2.1 SQLite FTS5

- 为标题、正文、标签和类别建立 FTS5 索引；
- 使用 BM25 排序；
- 定义可重复的中文分词策略；
- 写入、修订和遗忘时增量更新索引；
- 支持按项目、owner、时间、类型和状态过滤。

### P2.2 Semantic Embedding

定义可插拔接口：

```python
class EmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

要求：

- 支持本地 Embedding；
- 支持 OpenAI-compatible Embedding；
- 禁用 Embedding 时可以回退到纯 FTS5；
- 使用 `content_hash` 缓存向量；
- Provider 故障不能破坏基础词法检索。

### P2.3 融合排序

```text
Query
 ├── BM25 Top 30
 ├── Vector Top 30
 ├── 元数据和可见性过滤
 └── RRF 融合
       ├── confidence 加权
       ├── freshness / validity
       ├── relation boost
       └── rerank → Top K
```

图关系只用于补充召回、来源追踪和排序加权，不再作为第一检索入口。

### 可解释召回

新增 `fibmind_explain_recall`，返回：

- 命中的原因；
- 匹配词；
- 候选来自 BM25、向量还是图扩展；
- 使用的 confidence 和来源证据；
- 旧记忆被排除的原因。

### 验收

- 查询不再每次加载和遍历全部节点；
- 混合检索在 P0 数据集上超过当前关键词基线；
- 建立 1 万和 10 万条记忆的性能测试；
- 所有召回结果都能解释来源和排序信号。

## P3：Token 预算和双层记忆

> 状态（2026-09）：已落地一版。`recall` 按 token 预算分三段渲染（hot / search /
> expand），返回 `budget.used_tokens`、每段用量与截断原因；计数器可插拔，默认为
> 确定性估算。L0 热记忆按 kind（preference / requirement / decision / knowledge）
> 与 confidence 选取，session 内冻结为 `hot` 节点，review 后下一 session 刷新。
> 评测新增 `fibmind_budgeted`（120 token 硬预算 + 热记忆前置）：在固定小数据集上
> Recall@K 仍 1.0，但 MRR 0.84、可回答率 0.67、无关注入 0.44——常驻前置的代价
> 被如实计入。30% 节省阈值需在真实轨迹（S1 导入）上校准，当前数据集太小无法判断。

将当前字符长度限制升级成模型 Token 预算，并结合 Hermes 风格的稳定热记忆与 FibMind 的按需长期召回。

### L0 热记忆

Session 开始时自动注入：

- 用户稳定偏好；
- 项目固定约定；
- 高频环境信息；
- 项目的关键限制。

规则：

- 默认预算约 800–1200 Token；
- 有严格容量限制；
- Session 内使用 frozen snapshot；
- Session 结束后更新下一份快照；
- 容量满时合并、替换或拒绝写入，不能无限增长。

### L1 动态长期记忆

根据当前任务按需召回：

- 历史决策及理由；
- Bug、原因和解决方案；
- 测试结果；
- 需求变化；
- commit/branch 上下文；
- 已知风险和未解决事项。

### Context Planner

```text
总预算
├── 热记忆预算
├── 直接命中预算
├── 关系扩展预算
├── 原始证据预算
└── 安全余量
```

功能：

- 按目标模型计算 Token；
- 按边际价值选择下一条记忆；
- 合并相似内容；
- 超预算时优先保留高置信度原始证据；
- 返回实际使用量、预算和截断原因。

### 验收

- 返回结果不超过指定 Token；
- 相同 Session 的热记忆快照保持稳定；
- 输出 `used_tokens`、`budget_tokens` 和每条记忆的 Token 成本；
- 在 P0 评测中验证 Token 节省，而不是只比较字符长度。

## P4：自动记忆闭环

> 状态（2026-09）：前半已落地（S2）：`observe` 持久化为 session 作用域 episode；
> `fibbrain_review_session` 确定性抽取决策 / 错误 / 纠正 / 风险 / 验证 / 改动文件 /
> 摘要 / 程序（S3），三种模式（candidates / approve / auto），幂等；`pending`
> 状态与 approve / reject 工具。DSH 原生插件（`integrations/dsh/`）在
> `agent/turn-stopping` 自动触发 review，`tools/result` 自动 observe。
> 语义近重复检测与冲突版本识别（VERSION_OF 自动化）未做。

解决当前主要依赖 Agent 主动调用 `fibmind_append` 的问题。

### 写入准入

新增 `MemoryAdmissionPolicy`。

适合保存：

- 用户稳定偏好；
- 重要设计决策及理由；
- 用户纠正；
- 已验证的错误原因；
- 无法轻易重新发现的信息；
- 跨 Session 仍有价值的约定。

默认跳过：

- 临时对话；
- 大段原始日志；
- 可以通过代码快速重新发现的信息；
- 完全重复内容；
- 没有结论的猜测。

### 去重和冲突处理

- 内容哈希去重；
- 语义近重复检测；
- 同一事实的新旧版本识别；
- 冲突时建立 `VERSION_OF` 或触发 revise；
- 不允许两个相互矛盾的事实同时保持 active。

### Session Review

新增 MCP 工具：

- `fibmind_commit_session`；
- `fibmind_review_pending`；
- `fibmind_approve_memory`；
- `fibmind_reject_memory`。

Session 结束后提取：

- objective；
- key decisions；
- changed files；
- tests；
- unresolved risks；
- user corrections；
- 可晋升知识候选。

支持三种模式：

1. 自动写入；
2. 写入前审批；
3. 只生成候选，不保存。

### 验收

- 自动写入是幂等的；
- 重复运行 Session Review 不产生重复记忆；
- 错误准入可以拒绝或撤销；
- 自动写入准确率纳入 P0 评测。

## P5：总结、知识治理、安全和可观测性

### 真正的总结

用可插拔 Summarizer 替换当前摘录拼接。总结必须：

- 保留事实和决策理由；
- 保留文件、commit、测试等来源；
- 区分事实、推测和未解决问题；
- 不把冲突记忆总结成确定结论；
- 可以从原始事件重新生成；
- 记录使用的模型、Prompt 版本和输入来源。

### 知识生命周期

- 多条独立证据才能晋升；
- 支持证据撤回；
- 支撑记忆被 refuted 后，自动降低派生知识可信度；
- 新版本出现时将旧知识标记为 stale；
- 可以解释知识由哪些记忆推导而来；
- 定期重新验证长期未确认的知识。

### 安全

写入前检查：

- API Key、Token、密码和私钥；
- Prompt injection；
- 隐形 Unicode；
- 超长内容和恶意元数据；
- owner/project 越权。

同时提供：

- 写入审批；
- 删除审计；
- 备份、恢复和导出；
- 敏感字段脱敏；
- 本地数据库加密的扩展接口。

### 可观测性

- 查询次数、命中率和空召回率；
- 每次上下文注入 Token；
- 每条记忆被采用、确认、驳回的次数；
- 召回延迟；
- 自动写入接受率和撤销率；
- 热记忆和长期记忆的命中贡献。

新增 `fibmind_status` 和诊断命令，显示存储规模、索引状态、待审批记忆、失效知识和近期评测结果。

## 暂缓功能

在 P0–P3 完成前，不优先投入：

- 增加更多 Fibonacci 层；
- 增加更多图 relation 类型；
- 复杂的可视化编辑器；
- 远程分布式图数据库；
- 多租户 SaaS；
- Agent 自动生成大量关系边；
- Fibonacci 参数调优。

这些功能不能直接证明召回更准确或更省 Token。

## 发布节点

### v0.2 — Retrieval MVP

范围：P0–P3。

发布条件：

- 混合检索超过当前关键词基线；
- Token 预算可被严格执行；
- 热记忆和动态长期记忆可以协同；
- owner/project 隔离测试通过；
- 有可重复的效果、Token 和延迟报告。

### v0.3 — Automatic Memory

范围：P4。

发布条件：

- Session Review 幂等；
- 自动准入、去重和冲突处理可用；
- 支持审批模式；
- 自动写入质量有量化报告。

### v1.0 — Governed Memory

范围：P5。

发布条件：

- 总结可追溯、可重新生成；
- 派生知识支持证据失效传播；
- 安全扫描、备份恢复和诊断能力完整；
- 在真实项目的长期评测中证明收益。
