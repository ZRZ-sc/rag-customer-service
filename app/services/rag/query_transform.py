"""查询改写/扩展（需求文档流程2）：Multi-Query + HyDE。

优化点（阶段3）：
1. **合并为一次 LLM 调用**：原实现 multi_query 与 hyde 各发一次请求（并行仍需 ~2-3s），
   现在一次请求同时产出查询变体与假设性答案，冷启动省约 1s。
2. **结果缓存 24h**：`query_transform:{sha1(question)}`。入参是"上下文改写后的问题"
   （已由 query_rewrite 层完成指代消解），因此缓存键与问题一一对应，可安全复用。
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.logging import logger
from app.services import cache as cache_svc
from app.services.llm import get_llm

_MERGED_PROMPT = """你是检索增强助手。请为下面的用户问题同时产出两类检索辅助内容：

1) queries：{n} 个不同角度的检索查询，用于提高知识库召回率
   （覆盖口语表达、正式表达、同义词等不同角度，保留核心实体与关键词）；
2) hyde：一段 80~150 字的**假设性标准答案**，用于语义检索增强（不给用户看，只用于检索）。

只输出 JSON，不要解释：
{{"queries": ["查询1", "查询2"], "hyde": "假设性答案"}}

用户问题：{question}"""


def _parse(content: str) -> dict:
    """解析 LLM 输出（容忍 ```json 包裹与纯文本兜底）。"""
    text = (content or "").strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        lines = [
            ln.strip().lstrip("-•0123456789.、 ").strip()
            for ln in text.splitlines()
            if ln.strip()
        ]
        return {"queries": lines[:5], "hyde": ""}


async def transform_queries(
    question: str, n: int | None = None, use_hyde: bool | None = None
) -> dict:
    """返回 {"queries": [原问题, ...变体], "hyde": str, "cached": bool}。

    任何异常都退化为"只用原问题检索"，保证主流程不中断（需求文档 N8 降级）。
    """
    n = n or settings.multi_query_count
    use_hyde = settings.hyde_enabled if use_hyde is None else use_hyde

    cache_key = cache_svc.transform_cache_key(question)
    cached = await cache_svc.get_json(cache_key, name="query_transform")
    if cached and cached.get("queries"):
        return {"queries": cached["queries"], "hyde": cached.get("hyde", ""), "cached": True}

    queries: list[str] = [question]
    hyde_text = ""
    try:
        resp = await get_llm().ainvoke(
            [
                SystemMessage(content=_MERGED_PROMPT.format(n=n, question=question)),
                HumanMessage(content=question),
            ]
        )
        data = _parse(resp.content)
        for q in data.get("queries", []) or []:
            q = str(q).strip()
            if q and q != question and q not in queries:
                queries.append(q)
        if use_hyde:
            hyde_text = str(data.get("hyde", "") or "").strip()
    except Exception as exc:
        logger.warning("query transform failed, fallback to raw question: {}", exc)

    queries = queries[: max(1, n)]
    payload = {"queries": queries, "hyde": hyde_text}
    await cache_svc.set_json(
        cache_key, payload, ttl=settings.transform_cache_ttl_seconds, name="query_transform"
    )
    return {**payload, "cached": False}
