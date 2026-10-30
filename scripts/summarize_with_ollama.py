"""Summarize and import agent sessions (Codex, Claude Code, AGY) using OpenAI-compatible relay/proxy API or local Ollama."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from import_agent_histories import (
    HarvestedSession,
    harvest_agy_sessions,
    harvest_claude_sessions,
    harvest_codex_sessions,
)

from fibmind.models import MemoryScope
from fibmind.service import MemoryService

DISTILLATION_SYSTEM_PROMPT = """你是一个研发团队的资深架构记忆提炼专家。
你的任务是从 AI 编程助手（Codex / Claude Code / AGY）的原始开发会话中，提炼出可供全团队跨工具复用的【工程记忆】。

请分析提供的会话信息（用户目标、修改文件、操作记录），提炼成高质量工程事实，并严格以合法 JSON 格式输出，不要包含任何额外对话解释：

{
  "title": "[工程/模块名] 问题根因与解决方案（一句话精准概括）",
  "category": "必须在以下之一: decision(架构决策/方案设计), error(排错踩坑复盘), procedure(操作流程SOP), project(业务常识)",
  "content": "【背景目标】一句话描述业务需求或遇到的问题\\n【根因/决策】技术选型决策原因，或Bug报错的深层根因\\n【解决方案】具体的修复代码逻辑或配置变更\\n【涉及文件】核心改动文件列表",
  "tags": ["技术标签1", "标签2", "标签3"]
}"""


def _clean_json_text(raw_text: str) -> dict[str, Any]:
    """Clean markdown code block fences and parse JSON robustly."""
    text = raw_text.strip()
    # Strip ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding the first { ... } block
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            return json.loads(match.group(1))
        raise


def call_relay_chat(
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Call OpenAI-compatible /v1/chat/completions endpoint using standard library urllib."""
    clean_base = base_url.rstrip("/")
    if not clean_base.endswith("/v1"):
        clean_base = f"{clean_base}/v1"
    url = f"{clean_base}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    # Some proxies reject response_format if not supported by the downstream model,
    # so we supply it optionally or let the system prompt guide
    if "json" in model.lower() or "gpt" in model.lower() or "deepseek" in model.lower():
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "FibMind-Session-Harvester/1.0",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content_str = body["choices"][0]["message"]["content"]
            return _clean_json_text(content_str)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise ConnectionError(f"中转 API 请求失败 (HTTP {e.code} on {url}): {err_msg}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"无法连接到中转 API 服务 ({url}): {e}") from e
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"中转 API 返回内容无法解析为有效 JSON: {e}") from e


def summarize_and_save(
    session: HarvestedSession,
    service: MemoryService,
    base_url: str,
    api_key: str,
    model: str,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Summarize session via Relay API and persist into FibMind."""
    user_prompt = f"""请提炼以下来自 {session.host} 的开发会话：
- 来源工具: {session.host}
- 关联工程: {session.project_name}
- 会话时间: {session.timestamp}
- 用户目标/需求: {session.user_goal}
- 改动文件列表: {", ".join(session.files_touched) if session.files_touched else "未检测到具体文件改动"}
"""

    print(f"\n[{session.host}] 正在调用中转模型 ({model}) 总结会话: {session.user_goal[:50]}...")
    try:
        distilled = call_relay_chat(user_prompt, base_url=base_url, api_key=api_key, model=model)
    except Exception as exc:
        print(f"  ❌ 总结失败: {exc}")
        return None

    title = distilled.get("title", f"[{session.project_name}] {session.user_goal[:40]}")
    category = distilled.get("category", "decision")
    content = distilled.get("content", session.user_goal)
    tags = list(set(distilled.get("tags", []) + session.tags + [session.host]))

    print(f"  ✨ 提炼成功: {title}")
    print(f"     分类: {category} | 标签: {tags}")

    if not dry_run:
        node_id = service.append(
            category=category,
            title=title,
            content=content,
            tags=tags,
            scope=MemoryScope.PERSONAL,
            owner="abbila",
            workspace_id="fibmind",
            project_id=session.project_name or "fibmind",
            session_id=session.session_id,
            metadata={
                "host": session.host,
                "source_session_id": session.session_id,
                "distilled_by": model,
                "relay_host": base_url,
                "distilled_at": datetime.now().isoformat(),
            },
        )
        print(f"  💾 已落库到 FibMind: {node_id}")

    return distilled


def main():
    default_key = (
        os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("SSS_API_KEY") or ""
    )
    default_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"
    default_model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"

    parser = argparse.ArgumentParser(description="Summarize Codex/Claude/AGY sessions via OpenAI-compatible Relay API")
    parser.add_argument(
        "--base-url", default=default_url, help=f"API Base URL (默认从环境变量获取，当前: {default_url})"
    )
    parser.add_argument("--api-key", default=default_key, help="API Key (默认自动读取 OPENAI_API_KEY / SSS_API_KEY)")
    parser.add_argument("--model", default=default_model, help=f"模型名称 (默认: {default_model})")
    parser.add_argument("--limit", type=int, default=3, help="每个 Agent 提取的会话数量 (默认: 3)")
    parser.add_argument("--dry-run", action="store_true", help="仅提炼展示，不写入数据库")
    parser.add_argument("--store", default=".fibmind/memory.db", help="FibMind 数据库路径")
    args = parser.parse_args()

    if not args.api_key:
        print("❌ 错误: 未提供 API Key！")
        print("可以通过以下方式提供:")
        print('  1. 命令行参数: --api-key "sk-xxx"')
        print('  2. 环境变量: export OPENAI_API_KEY="sk-xxx"')
        sys.exit(1)

    service = MemoryService(args.store)

    masked_key = args.api_key[:6] + "..." + args.api_key[-4:] if len(args.api_key) > 10 else "***"
    print("=== 配置信息 ===")
    print(f"  Base URL: {args.base_url}")
    print(f"  Model   : {args.model}")
    print(f"  API Key : {masked_key}")
    print(f"  Store   : {args.store}")
    print(f"  Dry Run : {args.dry_run}")
    print("================")

    sessions: list[HarvestedSession] = []
    sessions.extend(harvest_agy_sessions(max_sessions=args.limit))
    sessions.extend(harvest_claude_sessions(max_sessions=args.limit))
    sessions.extend(harvest_codex_sessions(max_sessions=args.limit))

    print(f"\n共发现 {len(sessions)} 个候选会话，开始请求中转 API 进行提炼总结...")
    success_count = 0
    for s in sessions:
        res = summarize_and_save(
            s,
            service,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            dry_run=args.dry_run,
        )
        if res:
            success_count += 1

    print(f"\n=== 处理完成: 成功总结并入库 {success_count}/{len(sessions)} 条记忆 ===")


if __name__ == "__main__":
    main()
