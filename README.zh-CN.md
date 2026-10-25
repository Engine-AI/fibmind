# FibBrain

[English](README.md) | 简体中文

FibBrain 是面向 agent 的认知运行时（cognitive runtime），**FibMind** 是它底层的记忆引擎。

```text
Agent  =  FibBrain (plan / recall / decide / coordinate / remember)
       +  Harness  (run / execute / observe)
       +  Plugins  (tools / MCP / models)
```

Brain v1 在现有存储之上实现了七个动作：

| 动作 | 回答的问题 |
| --- | --- |
| `plan` | 我在做什么，按什么顺序做？ |
| `recall` | 关于这个目标，我已经知道什么？ |
| `coordinate` | 下一步该让躯体（body）调动哪些能力？ |
| `remember` | 这条内容值得写入长期记忆吗？ |
| `observe` | 当前 episode（情景）里刚发生了什么？ |
| `advise` | 根据过往证据，这个动作该不该执行？ |
| `reflect` | 一条已存储的记忆是否经受住了检验？ |

它不拥有 agent 循环，也不是插件管理器。DeepSeek-Harness（或任何其他 harness）仍然是躯体。规划是一个显式持久化的目标，而不是重写 harness 的 Goal 服务。协调只负责点名能力并给出建议，不执行插件。

记忆引擎这样建模长期知识：

- 多棵分类树，组成一片记忆森林（memory forest）
- 以 append-only 的 event log（事件日志）作为唯一事实来源（source of truth），节点和边只是可重建的缓存
- 用两个独立的信号，分别表示一条记忆有多*熟悉*，以及有多少外部证据表明它是*正确*的
- 作用域（scope）：保证某个 owner 的个人记忆不会出现在其他人的 recall 中，并用统计门槛决定一条观察能否提升为共享知识
- 按斐波那契数列设定容量的层级，分别存放原始、压缩、摘要和长期记忆
- 相关节点之间的图边，以及局部图搜索

图、检索、上下文、准入（admission）和存储模块只依赖 Python 标准库。recall 基于 BM25 倒排索引，再叠加标题、置信度和时效信号；设置 `FIBMIND_EMBEDDING=hashing`（离线）或 `FIBMIND_EMBEDDING=openai` 并配合 `FIBMIND_EMBEDDING_URL` / `_MODEL` / `_API_KEY`，即可通过 reciprocal rank fusion（倒数排名融合）并入一路向量候选。provider 出错时降级为纯词法 recall，而不会让调用失败。面向 agent 的服务端使用 MCP Python SDK 2.x。优先使用 `fibbrain_*` 工具；`fibmind_*` 保留为直接访问存储的 API。

> [!WARNING]
> FibMind 目前是一个本地、单用户的原型。多 owner 隔离和完整的 event log 重放不变量仍在加固中。暂时不要把它当作多租户生产服务，也不要在里面存放密钥等机密信息。

## 快速开始

```bash
cd fibmind
uv sync --locked --extra dev
.venv/bin/python examples/demo.py
.venv/bin/pytest -q
```

演示脚本会把两个示例 JSON 存储写到 `.fibmind/demo/`（已被 gitignore）。`data/` 下纳入版本管理的副本同时充当 `tests/test_contract.py` 的一致性测试夹具（conformance fixture）；只有在有意为之时才用 `examples/demo.py --fixture` 重新生成。持久化的 agent 记忆服务推荐使用 SQLite 后端。

本地 SQLite 运行时在 `.sqlite-version` 中固定为 **3.51.3**。在 macOS 或 Linux 上复现 CPython 3.12.13 环境：

```bash
uv sync --locked --extra dev --python 3.12.13
.venv/bin/python scripts/setup_sqlite_runtime.py
.venv/bin/python -c 'import sqlite3; print(sqlite3.sqlite_version)'
```

安装脚本需要 C 编译器、`make`、Perl，以及 xz、bzip2 和 libffi 开发库（macOS 使用 Command Line Tools 和 Homebrew `xz`）。脚本校验官方源码的哈希，构建包含 SQLite 3.51.3、保留原 OpenSSL 3.5.5 的项目私有解释器，通过 FTS5 和包导入检查后切换 `.venv`。已安装的包和记忆文件保留；源码、日志、解释器和原虚拟环境配置存放在 `.fibmind/runtime/`，使用该虚拟环境期间请保留此目录。重建 `.venv` 后需重新运行安装脚本；已启动的 MCP/守护进程需重启才能使用新运行时。此操作只作用于本项目环境，不更改其他 Python 环境或系统 SQLite 命令。

`uv` 是官方支持的环境创建方式（没有的话可以 `pip install uv`）。以下开发检查在每次 push 时都会由 CI 运行：

```bash
.venv/bin/ruff check src tests evals examples scripts     # 代码检查
.venv/bin/ruff format --check src tests evals examples scripts
.venv/bin/pytest -q
.venv/bin/python -m evals.gate                    # 评测结果不得低于 evals/floor.json
```

评测下限是合并门禁，不是报告。评测结果提升时就上调它；不要为了让构建通过而下调它。

## 检索评测

P0 评测套件在同一组固定数据集上比较四种方案：

- 不持久化任何记忆；
- 静态的 `AGENTS.md` 式上下文；
- 有界的 Hermes 式 hot memory（热记忆）快照；
- FibMind 当前的关键词搜索加图上下文扩展。

运行汇总报告：

```bash
.venv/bin/python -m evals.runner
```

将逐 case 的完整报告写成 JSON：

```bash
.venv/bin/python -m evals.runner \
  --format table \
  --output .fibmind/eval-report.json
```

用 `--baseline fibmind_current` 只跑一个 baseline，重复 `--baseline` 可以选择多个。内置数据集覆盖项目历史 recall、过期与相互冲突的记忆、负向查询、owner 隔离和跨 session recall。`git_history.json` 增加了基于真实仓库提交人工标注的 case，其中包括一条中文查询以及无关查询、项目隔离查询；每条事实都记录了来源提交。报告包含 Recall@K、MRR、上下文可回答性、无关/过期记忆污染、禁止记忆命中、上下文大小以及 P50/P95 延迟。带预算的 baseline 在计算检索指标时，只统计真正渲染进 prompt 的记忆。

若要比较**模型回答**和 **provider 报告的 token 数**，需要用同一个模型为每个 baseline 记录回答，并通过 `--answers path/to/answers.json` 传入一份经过审核的 JSON 文件。每条回答都需要 `baseline`、`case_id`、`answer`、经人工审核的布尔值 `correct`，以及 provider 返回的 `input_tokens` 和 `output_tokens`；文件本身还需要 `model` 和 `judged_by`。报告会为每个 baseline 增加回答准确率、覆盖率以及实际的输入/输出 token 用量。缺失的回答记为覆盖不完整，而不是算作正确。内置的仓库历史数据集不能替代真实的 agent 会话：`evals.import_dsh_session` 可以从真实会话生成候选数据集，但它的自动标注必须经过独立审核，才能用作门禁。

P0 报告的是 `estimated_tokens`：一个确定性的、与模型无关的大小估算，只用于相对比较。它刻意不被表述为 provider 的计费用量。P3 会把它替换为所选模型的 tokenizer。数据集格式、指标定义以及后续实现阶段见 [`ROADMAP.md`](ROADMAP.md)。

## 设计约束

实现围绕以下六条性质展开。之所以要写明，是因为每一条都排除了一种看似合理的设计：

**日志是唯一事实来源。** 每次语义变更都会追加一条 `MemoryEvent`；`FibMind.rebuild_from_log` 能据此重建整片森林。节点表和边表是可以随时丢弃的缓存。正因如此，有了更好的 summarizer 时存储可以重放，跨设备时也可以合并——文本能 diff、能 merge，派生状态不能。recall 计数器（`familiarity`、`access_count`）是例外：它们描述的是使用情况而不是知识，不参与重放。

**被 recall 不等于有证据。** 一条记忆每被读回一次，`familiarity` 就上升一次；它被刻意排除在相关性打分之外。只有 `confidence` 会影响排序——它只能通过 `record_outcome` 改变，而且必须给出具名来源。如果改为按 recall 频率打分，就会形成一个闭环（被读 → 排得更靠前 → 被读得更多），最终收敛到“最熟悉的”，而不是“正确的”。

**未经扫描，一律不准入。** 准入环节会拒绝携带 API key、私钥、连接串、prompt 注入话术或不可见 Unicode 字符的候选，并返回命中的问题类型。一条记忆会被重放进之后的每一个 prompt，所以一次泄露或一条被注入的指令都会是永久性的。

**知识继承其证据。** 一条共享结论（claim）会链接到它由之提升而来的那些观察。驳倒其中一个支撑项，结论的置信度会按仍然成立的支撑项占比缩放；全部驳倒，结论就退出 recall。结论自身的确认计数保持不变：因继承而削弱，并不是对结论本身的判断。

**大脑只能通过门禁自我调优。** `fibbrain_tune` 根据哪些记忆事后被确认、哪些被驳倒，推导出一个单步、有界的排序权重调整提案，在调整前后各跑一次固定评测套件，只有当 Recall@K、过期污染、禁止命中和无相关项准确率都没有变差时才采纳。familiarity 不是可调的权重。

**删除必须触及日志。** `forget` 会删除节点及其边，*并且*在更早的日志条目中把相关内容打上 tombstone（墓碑标记）。只删掉那一行的话，下一次重放会让它复活。

## 核心概念

```text
Node        A memory unit. It may be raw data, compressed data, summary, concept, root, or archive.
Edge        A typed relationship between two nodes.
Tree        A category or topic tree whose root is also a normal node.
Forest      Many trees connected by cross-tree graph edges.
Layer       A capacity band controlled by Fibonacci-sized limits.
Event       One entry in the append-only log. The forest is a function of these.
Scope       session / personal / knowledge — who a memory belongs to.
Status      active / stale / refuted — whether normal recall may return it.
```

### 作用域

```text
session     Scratch state, expected to be forgotten.
personal    An observation about one owner. Never shared: sharing both leaks
            privacy and pollutes other owners' preferences.
knowledge   A claim about the world, distilled from several personal
            observations. The only shareable scope.
```

把它们混在一个桶里，共享层就没法共享——里面会塞满某一个人的偏好——个人层也不再“个人”。`append` 只写入 `personal` 或 `session`；`knowledge` 只能经由 `promote_to_knowledge` 产生，它要求至少三条不同的支撑观察，并把每一条都链接为出处（provenance），这样某条结论日后被证明是错的，也能追溯回去。

常规的 search 和 context 只返回 `active` 记忆。`mark_stale` 让被取代的记忆退役，同时保留其出处；一次 `refuted` 的判定结果同样会把记忆移出常规 recall。维护类查询可以显式请求非活跃状态。`revise` 修正内容，并让记忆重新变为 active、且回到未经证实的状态。personal/session 记忆要求 owner 精确匹配；省略 owner 不会再暴露属于具名 owner 的记忆。

## 默认层级

```text
raw         21 memory weight
compressed 34 memory weight
summary    55 memory weight
long_term  89 memory weight
```

层级容量以记忆权重（memory weight）计量。raw 节点的权重为 `1`；compressed 或 summary 节点的权重等于它所覆盖的源节点权重之和。

关于这部分，有两点需要如实交代：

- 容量回答的是能放下*多少*条记忆，而不是*哪些*值得保留。斐波那契间距给出一组逐级增大、间隔有明确规则的容量带；换成任何类似的递增数列，效果都一样。
- `_summarize` 只是把截断后的片段拼接起来。它的输出与输入处在同一抽象层级，只是更短——这是占位实现，不是提炼。真正把具体内容提升为一般性结论，要走 `promote_to_knowledge`。由于折叠（folding）是用 `folded_into` 指针标记源节点，而不是覆盖它们，原始记忆得以保留，将来可以交给更好的 summarizer 重做。

## 主要操作

```python
memory = FibMind()
tree_id = memory.create_tree("code", title="Code Tree")
node_id = memory.append("code", title="Login bug", content="401 after token expiry")

memory.link_nodes(node_id, other_node_id, RelationType.RELATED_TO)
memory.promote_node(node_id)
results = memory.search_from(node_id, depth=2)
activated = memory.search_from(node_id, depth=2, reinforce=True)

# Attach outside evidence — the only thing that changes how a memory ranks.
memory.record_outcome(node_id, Verdict.CONFIRMED, source="pytest tests/test_auth.py")

# Correct a memory (confidence resets), retire an old one, or delete it for good.
memory.revise(node_id, content="401 only after refresh-token expiry")
memory.mark_stale(old_node_id, reason="superseded by the refresh-token decision")
memory.forget(stale_node_id, reason="user asked for removal")

# Distil repeated observations into one shareable claim (needs >= 3 supporters).
claim_id = memory.promote_to_knowledge(
    title="Uploads need an explicit retry budget",
    content="Large uploads die at the first transient timeout without a retry budget.",
    supporting_node_ids=[first, second, third],
)

# Nodes and edges are a cache; this rebuilds them from the log alone.
replayed = FibMind.rebuild_from_log(memory.events)
```

`search_from()` 默认只读。希望某次查询表现得像记忆激活时，传入 `reinforce=True`——注意它只会提高 `familiarity`，而 familiarity 不影响排序。

`search()` 是基于查询的对应接口：它按与文本查询的关键词相关性对整片森林排序，从不修改节点。传入 `owner=` 可排除其他 owner 的个人记忆，传入 `scopes=` 可限定范围，比如只查共享知识。

## MCP 服务端（为 agent 提供长期记忆）

FibMind 可以作为 [MCP](https://modelcontextprotocol.io/) 服务端运行，让 Codex、Claude Code 这类工具把它当作外部长期记忆。项目把 MCP Python SDK 的版本限定在 `>=2,<3`；MCP 1.x 的服务端导入路径不一样。

```bash
# 创建或同步虚拟环境
uv sync --locked --extra dev

# 可选的冒烟测试：直接运行 stdio 服务端
.venv/bin/python -m fibmind.mcp_server --store .fibmind/memory.db
```

直接运行这条命令时，它会静默等待 stdin 上的 MCP 消息。这是正常的：它不是 HTTP 服务，也不会打印 URL。通常由 MCP host 替你启动和管理这个进程。

### 工具

recall 与记录：

| 工具 | 用途 |
| --- | --- |
| `fibbrain_plan` | 创建或刷新一个持久化目标及其确定性计划。 |
| `fibbrain_coordinate` | 推荐下一步要调动的能力，并逐一给出 advise。 |
| `fibbrain_complete_goal` | 把持久化目标标记为完成；它仍然可以被 recall。 |
| `fibbrain_recall` | recall 一个受 token 预算约束的上下文包：先是冻结的 hot memory，然后是直接命中，最后是相关记忆，并按分区报告 `budget` 用量。 |
| `fibbrain_hot` | 当前 session 冻结的 L0 hot memory 快照（稳定的偏好、约定、有证据支撑的决策）；`refresh` 会重新计算。 |
| `fibbrain_tune` | 根据确认与驳倒的证据，提出一个有界的排序权重调整；只有固定评测套件显示没有退化时才采纳。每次尝试都会记入日志。 |
| `fibbrain_revalidation_candidates` | 超过 N 天未获确认的知识和 procedure（流程）。只输出，不做修改。 |
| `fibbrain_remember` | 先对候选做准入检查，值得保留才写入。 |
| `fibbrain_observe` | 记录一条 episode 事件；传入 `session_id` 时持久化为 session 草稿（scratch）。 |
| `fibbrain_review_session` | 把一个 session 的 episode 提炼为长期记忆候选（`candidates` / `approve` / `auto`）；幂等。 |
| `fibbrain_pending_reviews` | 列出等待审批的待审记忆。 |
| `fibbrain_remember_procedure` | 存储处理某一类任务的可复用做法（trigger、steps、tools、verify、inputs）。 |
| `fibbrain_render` | 把适用于某个目标的 procedure 渲染为 `skill`（SKILL.md）或 `tool`（JSON-Schema）定义，按成熟度过滤。 |
| `fibbrain_approve_memory` / `fibbrain_reject_memory` | 放行一条待审记忆，或将其标为 stale 退役。 |
| `fibbrain_advise` | 根据已存证据，允许或拒绝一个动作。 |
| `fibbrain_reflect` | 记录一条记忆事后被证明是对还是错。 |
| `fibmind_context` | 直接从存储构建上下文包（优先用 `fibbrain_recall`）。 |
| `fibmind_append` | 绕过准入检查的直接写入（优先用 `fibbrain_remember`）。 |
| `fibmind_search` | 按与查询的关键词相关性给记忆排序（只读）。 |
| `fibmind_search_from` | 展开以某个节点为根的关联树。 |
| `fibmind_link` | 在两条记忆之间建立带类型的关系。 |
| `fibmind_explain_recall` | 逐条给出 lexical / vector / fusion / confidence / recency 信号，并说明其他候选为何被排除。 |
| `fibmind_status` | 存储规模、各生命周期状态计数、词法索引大小、embedding 健康状况。 |

判断与维护——防止存储退化的另一半：

| 工具 | 用途 |
| --- | --- |
| `fibmind_record_outcome` | 记录一条记忆事后是对还是错，并注明来源。 |
| `fibmind_revise` | 修正一条记忆并重新激活，置信度重置。 |
| `fibmind_mark_stale` | 让被取代的内容退役，同时保留出处。 |
| `fibmind_forget` | 永久删除一条记忆，包括日志历史中的内容。 |
| `fibmind_promote_knowledge` | 把多条观察合并为一条共享结论。 |

最常用的调用是：开始任务前先 `fibbrain_plan`、再 `fibbrain_recall`，完成后 `fibbrain_remember`。`fibbrain_coordinate` 给出下一步该调动的能力；针对某个具体动作，仍然可以直接调用 `fibbrain_advise`。`fibbrain_remember` 会跳过重复内容和原始转储（dump）。只写不评的存储照样会退化，除非用 `fibbrain_reflect` 记录这些记忆是否经受住了检验。

### Host 验证

Claude Code 和 Codex 都能通过 stdio MCP 驱动 FibBrain。脚本化的可行性验证会把相互隔离的记忆写入 `.fibmind/memory.db`：

| Host | 调用序列 |
| --- | --- |
| Claude Code (`claude -p`) | `plan` → `remember` → `recall` → `coordinate` → `complete` |
| Codex (`codex exec`) | 同一序列，需先为该 server 开启 MCP 自动批准 |

把 [`examples/host_mcp_prompt.md`](examples/host_mcp_prompt.md) 粘贴到一个**新的**交互式会话里，即可手动复现。修改 MCP 配置后要重启或新开任务；已经打开的会话看不到这个 server。

### DeepSeek-Harness

两种方式，都在本仓库中：

- **零代码**：[`examples/dsh/`](examples/dsh/README.zh-CN.md) 通过 `@deepseek-ai/dsh-mcp-client` 挂载 MCP 服务端；由模型调用 `mcp__fibbrain__*` 工具。
- **原生插件**：[`integrations/dsh/`](integrations/dsh/README.zh-CN.md) 是一个 Cordis 插件，在每个 prompt 之前 recall，在每次工具调用之前 advise，记录执行结果，在 turn 结束时 review 整个 session，并把已验证的 procedure 发布为 skill——模型不需要记得主动调用任何东西。

### 在 Codex 中注册

Codex 的注册是全局的，所以要在仓库根目录执行，让 `$PWD` 展开成绝对路径；这样无论 Codex 当前的工作目录在哪里，都能启动这个服务端：

```bash
codex mcp add fibmind -- \
  "$PWD/.venv/bin/python" \
  -m fibmind.mcp_server \
  --store "$PWD/.fibmind/memory.db"
```

检查保存下来的配置：

```bash
codex mcp get fibmind
codex mcp list
```

注册后要重启 Codex 或新开一个任务。MCP 配置在任务启动时加载，已有的任务可能发现不了新加的 server。不需要单独开一个终端跑服务端；Codex 会自己拉起 stdio 进程。

要在新的 Codex 任务里验证连接，可以让它调用 `fibbrain_remember` 写入一条小的测试记忆，再用 `fibbrain_recall` 取回来。本仓库的 `AGENTS.md` 已经要求 Codex：

1. 开始非平凡的工作前，调用 `fibbrain_plan` 和 `fibbrain_recall`；
2. 在执行某项可能被过往证据拦下的能力之前，调用 `fibbrain_coordinate`（或 `fibbrain_advise`）；
3. 实现完成后调用 `fibbrain_remember`，记录决策、改动的文件、测试和尚未解决的风险；
4. 当新证据改变了某条已存记忆时，调用 `fibbrain_reflect`、`fibmind_revise`、`fibmind_mark_stale` 或 `fibmind_forget`。

用 `codex mcp remove fibmind` 移除注册。Codex 的通用配置细节参见 [Codex MCP 官方文档](https://developers.openai.com/codex/mcp/)。

### 在 Claude Code 中注册

project 作用域的 server 从仓库根目录启动，所以可以用相对路径。仓库里提交的 [`.mcp.json`](.mcp.json) 正是下面这条注册：

```bash
claude mcp add --scope project --transport stdio fibmind -- \
  .venv/bin/python \
  -m fibmind.mcp_server \
  --store .fibmind/memory.db
```

然后通过 `CLAUDE.md`（Claude Code）或 `AGENTS.md`（Codex）告诉 agent 什么时候使用它——本仓库两者都已包含。

### 架构

```text
Codex / Claude Code
        │ fibbrain_* / fibmind_* MCP tools
        ▼
MCP Server                (src/fibmind/mcp_server.py)
        ▼
FibBrain                  (plan / recall / coordinate / remember / observe / advise / reflect)
        ▼
MemoryService             (src/fibmind/service.py — load/lock/persist)
        ▼
FibMind Core              (graph, ranking, context, admission)
        ▼
event log  ──rebuild──▶  nodes / edges / trees
(source of truth)        (cache, JsonStore or SqliteStore)
```

### 存储

`--store` 按后缀选择后端：`.json` 对应 `JsonStore`（原子写入 + fsync，适合演示），`.db` / `.sqlite` 对应 `SqliteStore`（WAL，`BEGIN IMMEDIATE` 事务）。两者都会把 event log 与物化状态一起持久化。

SQLite schema v4 增加了 `workspace_id`、`project_id`、`session_id` 和 `task_id`，让 recall 能把不同的工作上下文隔离开。schema v3 增加了生命周期字段 `status` 和 `status_reason`。schema v2 引入了 `scope`、`owner`、`familiarity`、`confidence`、`confidence_source`、`folded_into` 以及 `event_log` 表。旧数据库在打开时原地升级：旧的 `importance` 列变为 `familiarity`（confidence 从零开始，因为那些数值从来没有外部依据），v1 中通过改写 layer 完成折叠的节点会恢复出 `folded_into` 指针。`fibmind-migrate` 可以把 JSON 存储转换为 SQLite。

```bash
.venv/bin/python -m fibmind.migrate \
  --from-json data/demo-memory.json \
  --to-sqlite data/demo-memory.db
```

迁移工具不会覆盖已存在的 SQLite 目标文件。
