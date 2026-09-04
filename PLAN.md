# FibMind 判据化改造 Plan

下一刀（身份模型）见 [`PLAN-identity.md`](PLAN-identity.md)。本文是已经落地的判据化改造记录。


## 背景诊断

当前实现(1873 行 core)把工程投入全部放在**生成/供给侧**(store、检索、context 拼装),
判据与治理侧为零。三个结构性问题:

1. **写入无判据** — `service.append()` 只校验字符串非空,MCP 面 5 个工具三读两写零判。
2. **importance 是反向判据** — `_reinforce_node` 只按访问频次加权,且以 0.3 权重进排序,
   形成 `被检索 → importance↑ → 排名↑ → 更易被检索` 的自增强回路。60 次召回打满,
   零次正确性信号。库越用越脏,脏的部分自动升到最前。
3. **状态快照即事实源** — upsert + `_delete_missing`,无事件流。压缩原地改写原节点
   (`layer` → `folded:raw`),不可重放、不可 merge、不可重炼。且无 delete/revise。

外加两个可复现 bug 和一处命名层:

- `promote_node` 把节点设为 ROOT,而 `rank_nodes` 默认排除 ROOT → **标记为重要 = 再也搜不到**。
- `append` 只建 root→node 的 TREE_CHILD 边,图是星型,`context(depth=1)` 展开恒为 0。
- Fibonacci 21/34/55/89 换成 20/40/80/160 行为不变;`low_watermarks` 硬编码值恒等于
  `previous_fibonacci()`,是冗余参数。

## 原则

- **日志是唯一事实源,物化状态是可丢弃的编译产物。**
- **familiarity(熟悉度)不得进入排序权重;只有 confidence(带外部证据)可以。**
- **个人层与知识层在数据模型层面可分,晋升需过统计门槛。**
- 本轮不做真蒸馏(需 LLM + 统计判据),但把它的**门**用代码实现:抽象文本由调用方给,
  门槛由代码强制。

## 执行步骤

### A. 事件日志(不可逆性拆除)

- 新增 `MemoryEvent` 模型 + `EventOp` 枚举:
  `create_node / link / fold / revise / forget / observe / promote`。
- `FibMind` 维护 `events: list[MemoryEvent]`,所有语义变更记事件。
- `SqliteStore` 新增 `event_log` 表(append-only),`JsonStore` 序列化 `events`。
- 新增 `rebuild_from_log(events)` → 从事件流重建整个森林。
- **压缩不再原地改写**:源节点保留 `layer` / `node_type`,新增 `folded_into: str | None`
  指针;配额核算排除已折叠节点。
- familiarity / access_count 明确定义为**快照缓存,不可重放**(重建后归零)——
  它正是"可丢弃的编译产物"。

### B. importance 拆分为 familiarity + confidence

- `MemoryNode.importance` → `familiarity`(访问频次派生),新增 `confidence: float`
  (需外部证据才能动)、`confidence_source: str | None`。
- `from_dict` 兼容读取 legacy `importance` 键。
- `ranking`:删除 familiarity 的评分权重,改为 `WEIGHT_CONFIDENCE * confidence`。
  → 自增强回路断开。
- 新增 `record_outcome(node_id, verdict, source, note)`,verdict ∈
  `confirmed / refuted`,写 `observe` 事件。这是判据的接入点。
- `_retention_key` 改为 confidence 优先,被证伪的先折叠。

### C. revise + forget

- `revise(node_id, ...)`:改内容 → confidence 归零(旧判据对新内容无效),写事件。
- `forget(node_id, reason)`:真删节点与其边;并**红线化日志中该节点的历史 payload**
  (content 替换为 tombstone),保证重放不复活内容。合规要求的"可删除"。

### D. scope 分层 + 知识晋升门

- 新增 `scope: MemoryScope`(`session / personal / knowledge`,默认 `personal`)
  与 `owner: str | None`。SQLite schema v1 → v2 迁移。
- `search` / `search_from` / `context` 支持 scope 过滤。
- `promote_to_knowledge(content, supporting_node_ids, ...)`:**要求 ≥ N 条支撑实例**
  才允许创建 knowledge 节点,建 DERIVED_FROM 边指回每条实例(出处指针),
  confidence 初始为 0。抽象文本由调用方提供,门槛由代码强制。
- knowledge 节点被证伪时可 `forget`,通过 DERIVED_FROM 边可反查污染面。

### E. 两个 bug

- `promote_node` 不再设 ROOT、不再自动 expand;`ROOT` 保留给 `create_tree` 造的结构锚。
  `expand_node_to_tree` 也保持节点为 CONCEPT。→ 提升后仍可检索。
- `append` 自动按现有 ranking 给新节点连 top-N `SIMILAR_TO` 边(同 scope、过阈值),
  让 `depth=1` 展开真的有东西。可配置,默认开。

### F. MCP 面补判据工具

新增 `fibmind_revise` / `fibmind_forget` / `fibmind_record_outcome` /
`fibmind_promote_knowledge`;`append` / `search` / `context` 加 scope 参数。
`CLAUDE.md` / `AGENTS.md` 改为要求 agent **记录结果判据**,不只是 append。

### G. Fibonacci 去冗余

删掉 `DEFAULT_LAYER_POLICY` 里冗余的 `low_watermarks` 硬编码值(保留 override 机制),
docstring 说明容量是条数启发式而非价值判据。

### H. 文档 / demo / 测试

README 诚实描述分层做了什么、没做什么;更新 demo 与 `data/demo-memory.json`;
为 A–E 每项加测试;修正引用 `importance` 的现有测试。

## 验收

- `pytest` 全绿。
- 60 次纯召回后 confidence 仍为 0,排序不变(回路已断)。
- `rebuild_from_log()` 重建结果与物化状态在节点/边/树上一致。
- `forget` 后日志重放不复活内容。
- `promote_node` 后节点仍可被 `search` 命中。
- `context(depth=1)` 在默认 append 路径下产出 expand 节点。
- 少于 N 条支撑实例时 `promote_to_knowledge` 拒绝。
