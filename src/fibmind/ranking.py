"""Keyword-based ranking for query search over memory nodes.

This module is intentionally dependency-free and pure: it never mutates the
nodes it scores. Semantic/embedding ranking can later replace ``rank_nodes``
behind the same ``ScoredHit`` interface.

Note what is deliberately absent from the formula: recall frequency. Scoring
familiarity closes a loop — recalled nodes rank higher, ranking higher gets them
recalled again — which promotes whatever is familiar rather than whatever is
correct. Only ``confidence``, which moves solely on external evidence, is
trusted here.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta

from fibmind.models import MemoryNode, NodeType, utc_now

# Scoring weights (kept as named constants — no magic numbers in the formula).
WEIGHT_COVERAGE = 1.0  # fraction of query terms found anywhere in the node
WEIGHT_TITLE = 0.6  # fraction of query terms found in the title
WEIGHT_CONFIDENCE = 0.3  # externally evidenced correctness
WEIGHT_RECENCY = 0.2  # how recently the node was updated

# A node updated within this window counts as fully "recent"; older nodes decay
# linearly to zero across it.
RECENCY_HORIZON = timedelta(days=30)

# Function words create false positives in small keyword corpora: a query about
# Kubernetes used to recall unrelated memories solely because both contained
# "the" or "is". Keep the list intentionally small and English-only; CJK runs
# retain their full text and overlapping bigrams.
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF


_CJK_TRADITIONAL = (
    "推進證據錨點內存規劃評測門禁復用認識論狀態緩存線程進程節點邊界驗證決策錯誤優化"
    "結構網絡數據庫調度權限鎖異步並發圖譜檢索壓縮過期刪除遺忘歷史會話任務項目環境"
    "測試運行實現設計問題結果條件異常處理調用服務客戶端協議反者道之動關聯關係傳輸"
    "總結讀寫創建獲取查詢變量參數語言代碼模型執行歸一門檻評估監控統計報告檔案目錄"
    "尋址思考智能體發現規律源頭極限導航鏡像終端命令列表隊列堆棧樹狀鏈接釋放鎖定衝"
    "突矛盾否認證偽確認假設猜想歸納演繹認知記憶脈絡沉澱痕跡拓撲演化晉升降級廢棄清"
    "理乾淨準備就緒完成終止中斷阻塞崩潰恢復備份快照遷移同步複製映射轉換編碼解碼序"
    "列校驗簽名密鑰證書憑證身份授權訪問策略規則攔截嗅探監聽響應呼叫請求線索關鍵字"
    "詞彙模組擴展插件鉤子事件驅動訊息訂閱發布廣播集群流水線批次延遲吞吐負載容災備"
    "援故障診斷追蹤遙測採樣指標度量閾值報警告警提示指引規範標準契約協定規格實例抽"
    "象封裝繼承多態介面函式運算遞歸循環迭代階梯範疇視野洞見決斷辯證質疑證明駁倒淘"
    "汰活躍凍結歸檔摘要提煉萃取精簡冗餘膨脹矩陣向量張量權重偏置偏差損失梯度收斂發"
    "散震盪穩定魯棒脆弱敏感漂移退化衰減增益抑制激活觸發喚醒銷毀構建打包部署維護治"
    "理運營審計查核合規執照憑據密碼雜湊散列倒排搜尋查找匹配篩選過濾排序聚類分類標"
    "記註釋解釋闡述概括備忘筆記記錄足跡軌跡脈衝訊號標誌狀態機遷移圖邊界點鄰接彙總"
    "研判行動反饋復盤反思升華演進變革創新突破瓶頸阻礙壁壘屏障隔離沙箱容器虛擬化實"
    "體虛構實踐啟發定理公理推論引理辨偽勘誤修訂補充完善擴充精煉提純開關標籤類型等"
    "級儲備讀取寫入刪減總量頻率單元節流防抖攔截器適配器裝飾器單例工廠觀察者個這那"
    "麼樣誰哪裡們國會說對經當頭開後從來兩長見問帶讓給還車電與統架時應機制"
)

_CJK_SIMPLIFIED = (
    "推进证据锚点内存规划评测门禁复用认识论状态缓存线程进程节点边界验证决策错误优化"
    "结构网络数据库调度权限锁异步并发图谱检索压缩过期删除遗忘历史会话任务项目环境"
    "测试运行实现设计问题结果条件异常处理调用服务客户端协议反者道之动关联关系传输"
    "总结读写创建获取查询变量参数语言代码模型执行归一门槛评估监控统计报告档案目录"
    "寻址思考智能体发现规律源头极限导航镜像终端命令列表队列堆栈树状链接释放锁定冲"
    "突矛盾否认证伪确认假设猜想归纳演绎认知记忆脉络沉淀痕迹拓扑演化晋升降级废弃清"
    "理干净准备就绪完成终止中断阻塞崩溃恢复备份快照迁移同步复制映射转换编码解码序"
    "列校验签名密钥证书凭证身份授权访问策略规则拦截嗅探监听响应呼叫请求线索关键字"
    "词汇模组扩展插件钩子事件驱动信息订阅发布广播集群流水线批次延迟吞吐负载容灾备"
    "援故障诊断追踪遥测采样指标度量阈值报警告警提示指引规范标准契约协定规格实例抽"
    "象封装继承多态接口函数运算递归循环迭代阶梯范畴视野洞见决断辩证质疑证明驳倒淘"
    "汰活跃冻结归档摘要提炼萃取精简冗余膨胀矩阵向量张量权重偏置偏差损失梯度收敛发"
    "散震荡稳定鲁棒脆弱敏感漂移退化衰减增益抑制激活触发唤醒销毁构建打包部署维护治"
    "理运营审计查核合规执照凭据密码哈希散列倒排搜索查找匹配筛选过滤排序聚类分类标"
    "记注释解释阐述概括备忘笔记记录足迹轨迹脉冲信号标志状态机迁移图边界点邻接汇总"
    "研判行动反馈复盘反思升华演进变革创新突破瓶颈阻碍壁垒屏障隔离沙箱容器虚拟化实"
    "体虚构实践启发定理公理推论引理辨伪勘误修订补充完善扩充精炼提纯开关标签类型等"
    "级储备读取写入删减总量频率单元节流防抖拦截器适配器装饰器单例工厂观察者个这那"
    "么样谁哪里们国会说对经当头开后从来两长见问带让给还车电与统架时应机制"
)

_CJK_TRANS_TABLE = str.maketrans(_CJK_TRADITIONAL, _CJK_SIMPLIFIED)


def normalize_cjk(text: str) -> str:
    """Normalize traditional Chinese characters to simplified Chinese."""
    if not text:
        return ""
    return text.translate(_CJK_TRANS_TABLE)


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text, including overlapping CJK bigrams."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalize_cjk(normalized)
    runs: list[tuple[str, str]] = []
    current: list[str] = []
    current_kind: str | None = None

    for character in normalized:
        kind = "cjk" if _is_cjk(character) else "word" if character.isalnum() else None
        if kind is None:
            if current:
                runs.append((current_kind or "word", "".join(current)))
                current = []
                current_kind = None
            continue
        if current and kind != current_kind:
            runs.append((current_kind or "word", "".join(current)))
            current = []
        current.append(character)
        current_kind = kind

    if current:
        runs.append((current_kind or "word", "".join(current)))

    tokens: list[str] = []
    seen: set[str] = set()
    for kind, run in runs:
        if kind == "word" and run in STOP_WORDS:
            continue
        candidates = [run]
        if kind == "cjk" and len(run) > 1:
            candidates.extend(run[index : index + 2] for index in range(len(run) - 1))
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                tokens.append(candidate)
    return tokens


@dataclass(frozen=True, slots=True)
class ScoredHit:
    """A node together with its relevance score for a query."""

    node: MemoryNode
    score: float
    matched_terms: tuple[str, ...] = field(default_factory=tuple)


def _recency_factor(node: MemoryNode) -> float:
    age = utc_now() - node.updated_at
    if age <= timedelta(0):
        return 1.0
    if age >= RECENCY_HORIZON:
        return 0.0
    return 1.0 - (age / RECENCY_HORIZON)


def score_node(node: MemoryNode, query_terms: set[str]) -> tuple[float, tuple[str, ...]]:
    """Score a single node against the query terms.

    Returns the score and the set of query terms that matched. A score of ``0``
    means no query term appeared in the node's searchable text.
    """
    if not query_terms:
        return 0.0, ()

    title_terms = set(tokenize(node.title))
    body_terms = title_terms.union(
        tokenize(node.content),
        tokenize(node.category),
        *(tokenize(tag) for tag in node.tags),
    )

    matched = tuple(term for term in query_terms if term in body_terms)
    if not matched:
        return 0.0, ()

    coverage = len(matched) / len(query_terms)
    title_matches = sum(1 for term in query_terms if term in title_terms)
    title_coverage = title_matches / len(query_terms)

    score = (
        coverage * WEIGHT_COVERAGE
        + title_coverage * WEIGHT_TITLE
        + max(0.0, min(1.0, node.confidence)) * WEIGHT_CONFIDENCE
        + _recency_factor(node) * WEIGHT_RECENCY
    )
    return score, matched


def rank_nodes(
    nodes: Iterable[MemoryNode],
    query: str,
    top_k: int = 5,
    categories: set[str] | None = None,
    min_score: float = 0.0,
    include_roots: bool = False,
    exclude_categories: set[str] | None = None,
) -> list[ScoredHit]:
    """Rank nodes by relevance to ``query``.

    Root nodes are excluded by default because they are structural anchors
    rather than recallable content. The result is sorted by descending score
    and truncated to ``top_k``.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    hits: list[ScoredHit] = []
    for node in nodes:
        if not include_roots and node.node_type == NodeType.ROOT:
            continue
        if categories is not None and node.category not in categories:
            continue
        if exclude_categories and node.category in exclude_categories:
            continue
        score, matched = score_node(node, query_terms)
        if score <= min_score:
            continue
        hits.append(ScoredHit(node=node, score=score, matched_terms=matched))

    hits.sort(key=lambda hit: (hit.score, hit.node.updated_at), reverse=True)
    return hits[:top_k]
