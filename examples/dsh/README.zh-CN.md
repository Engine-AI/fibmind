# 在 DeepSeek-Harness 上运行 FibBrain（零代码）

[English](README.md) | 简体中文

本目录通过 [DeepSeek-Harness](https://github.com/deepseek-ai/deepseek-harness)（dsh）的通用 MCP 客户端挂载 FibBrain。不写 TypeScript，也不改 dsh 源码。dsh 自己的第三方记忆示例（`docs/user/guide/mcp-memory.md`）用的也是这种形态。

能得到的：模型可以以 `mcp__fibbrain__<tool>` 的形式调用所有 `fibbrain_*` / `fibmind_*` 工具。得不到的：每一步自动 recall、工具调用拦截、由记忆渲染出的 skill。需要这些的话，请挂载 [`integrations/dsh/`](../../integrations/dsh/README.zh-CN.md) 中的原生 Cordis 插件；它的 overlay 也包含了这里的这一条挂载项。

## 前置条件

- Node 22+（启动器是 `npx @deepseek-ai/dsh`，不需要全局安装任何东西）
- 已同步本仓库：`uv sync --locked --extra dev`
- 已在 dsh 中配置 DeepSeek API key

## 单次运行时启用

[`fibbrain.cordis.yml`](fibbrain.cordis.yml) 以 `cwd` 为基准解析解释器和存储路径，而 `cwd` 必须是绝对路径。把 `/path/to/fibmind` 替换为你的 checkout 路径，或者为它生成一份 overlay，然后在仓库根目录运行：

```sh
.venv/bin/python -m evals.import_dsh_session --print-overlay > .fibmind/fibbrain.cordis.yml
npx @deepseek-ai/dsh web --patch "$PWD/.fibmind/fibbrain.cordis.yml"
```

工具发现是异步的。等工具列表里出现 `mcp__fibbrain__fibbrain_recall` 之后，再发第一个 prompt。

## 跨运行保留

把那一条 `insert` 条目合并进你的用户 patch 层：

- 单个 profile：`$DSH_HOME/profiles/<name>/cordis.patch.yml`
- 所有 profile：`$DSH_HOME/cordis.patch.yml`

不要直接用这个文件覆盖已有的文件；那些文件里可能还有无关的 patch。

## 告诉模型何时使用

把 [`instructions.md`](instructions.md) 粘贴到 dsh 加载的工作区 `AGENTS.md` 中，或者放进你的模型指令里。

## 验证（三步，两个 session）

按 dsh 自己的记忆验证流程来做，这样结果才能和其他记忆服务端对比：

1. Session A：`Remember that my validation drink is lapsang-<unique suffix>.` 确认模型调用了 `mcp__fibbrain__fibbrain_remember`，且结果中包含 `"verdict": "write"`。
2. Session B（同一个 Host，新 session）：`What is my validation drink? Check memory.` 确认它调用了 `mcp__fibbrain__fibbrain_recall`，并答出了那个值。
3. 仍在 B 中：`Use that preference to suggest one drink for the meeting.`

然后直接查看存储：

```sh
sqlite3 .fibmind/memory.db "select title, owner, workspace_id, session_id from nodes where node_type='raw' order by created_at desc limit 5;"
```

## 把真实 session 转成评测数据集

dsh 为每个 session 保存一份 append-only 的 JSONL 日志。FibBrain 自带一个导入工具，把日志里的 `fibbrain_*` 调用转换成一份 P0 候选数据集：

```sh
.venv/bin/python -m evals.import_dsh_session \
  "<dsh-session-root>/--<cwd>--/<session-id>/session.jsonl.zstd" \
  --output evals/candidates/real_dsh.json
```

dsh 默认的日志用 Zstandard 压缩；导入工具会调用 `zstd` 命令行（Homebrew：`brew install zstd`）。如果更想要纯文本行，可以在 `@deepseek-ai/dsh-session-persistence-jsonl` 上设置 `compression: 'none'`。

输出只是一份**候选**：memories 来自实际写入了的 `fibbrain_remember` 调用，cases 来自 `fibbrain_recall` 调用，`relevant_ids` 则按 recall 实际返回的结果自动标注。把文件移入 `evals/datasets/` 之前，务必审核并修正这些标注，否则评测衡量的只是它和自己的一致程度。
