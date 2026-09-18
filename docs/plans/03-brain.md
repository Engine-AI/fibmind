# FibBrain Plan

身份模型见 [`PLAN-identity.md`](PLAN-identity.md)，已落地。本文是 Brain 协议：在 FibMind 存储上长出 Cognitive Runtime。

不重写 Agent Loop，不做 Plugin Manager，不接 DeepSeek-Harness Cordis。

## 定位

```text
    FibBrain  =  Memory + Planning + Decision + Coordination + Reflection
    FibMind   =  事件日志、身份隔离、检索、治理
    Harness   =  Loop / Session / Tool / Event
```

| 动作 | 含义 | 是否落盘 |
| --- | --- | --- |
| `plan` | 把目标写成 Goal，并给出确定性步骤 | 是，`category=goal` |
| `observe` | 本轮发生了什么 | 否，进程内 episode |
| `recall` | 按 Goal + State 召回 | 否 |
| `coordinate` | 下一步该动哪些能力，并对建议动作 `advise` | Goal 可能被标成 blocked |
| `admit` / `remember` | 值不值得写成长期记忆 | 仅 `write` |
| `advise` | 这个动作该不该跑 | 否 |
| `reflect` | 某条记忆后来对不对 | 是，走 `record_outcome` |
| `complete_goal` | 目标做完了 | 是，节点仍可召回 |

## 写入门槛

`remember` 在同一事务里先 `admit` 再 `append`。跳过：

- 准备写成 `knowledge` 的候选（必须走 `promote_to_knowledge`）
- 没有 `session_id` 的 session scratch
- 过长或像日志 / 堆栈的原文
- 很短且没有结论的猜测
- 同一 owner / workspace / project 下的精确重复

`fibmind_append` 仍可绕过门槛，留给调试和评测。Agent 默认走 `fibbrain_remember`。Goal 走 `plan`，不经过 admit。

## 规划

`plan` 把 objective 写成一条 personal `goal` 记忆（标签 `fibbrain-goal`）。同身份下同一 objective 复用已有未完成 Goal。步骤是确定性的，不调用模型：

1. Recall related history
2. 被证伪过的相关记忆 → `Avoid: …`
3. 已确认的相关记忆 → `Reuse: …`
4. 按 objective 推断的工作能力（code / test / docs / search / review）
5. Verify
6. Reflect and remember

## 建议与协调

`advise` 在当前身份下检索动作名。存在匹配的 **refuted** 记忆则 `reject`；否则 `allow`，并带回相关 active 记忆作注。不做 rewrite，也不自建 Harness Goal 服务。

`coordinate` 先 `plan`，再按 Memory → 工作能力 → Memory 的顺序给出能力清单，并对每个建议动作跑 `advise`。任一 reject 则 Goal 标为 `blocked`。协调只推荐，不执行。

## 验收

- `remember` 写入后能被同 State 的 `recall` 看到
- 重复 / 日志 / 短猜测被 skip，且不落盘
- `observe` 只出现在 episode，不进长期记忆
- 跨 project 的 `recall` / `plan` 不串数据
- 被 refute 过的动作，`advise` 返回 reject，`coordinate` 会 blocked
- `plan` 同一 objective + 同一身份幂等
- `reflect` 可用 `node_id` 或 goal 解析目标记忆
- 现有 `fibmind_*` MCP 工具仍在；新增 `fibbrain_*` 覆盖五项能力
- `pytest` 全绿，P0 评测不回退
- `examples/mcp_smoke.py` 走通 plan → coordinate → remember → recall → advise → reflect → complete

## 之后

```text
Brain v1（本刀：Memory + Planning + Decision + Coordination + Reflection）
    ↓
DSH Cordis 插件：pre-step → recall，turn/end → remember / reflect
    ↓
P2 混合检索（换引擎，不换 Brain 接口）
```
