"""RAG 主流程编排（需求文档流程2）：
查询改写(Multi-Query+HyDE) → 混合检索(RRF) → Rerank → Prompt+LLM生成。
含阶段耗时统计与 Milvus 故障降级。
"""
import asyncio
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.logging import logger
from app.core.metrics import (
    llm_generation_duration,
    query_transform_duration,
    rag_retrieval_duration,
    rerank_duration,
    tokens_consumed_total,
)
from app.services.llm import extract_usage, get_llm
from app.services.rag.query_transform import hyde, multi_query
from app.services.rag.reranker import reranker
from app.services.rag.retriever import hybrid_retriever

LOW_CONFIDENCE_ANSWER = "抱歉，我暂时无法确定答案，请稍后咨询人工客服。"

_SYSTEM_PROMPT = """你是"优选商城"的智能客服助手。请严格基于提供的【知识库上下文】回答用户问题。

规则：
1. 只使用上下文中的信息回答，不要编造；
2. 如果上下文不足以回答用户问题，直接回复："抱歉，我暂时无法确定答案，请稍后咨询人工客服。"；
3. 回答简洁、口语化，不超过 150 字；
4. 涉及金额、时效、政策的内容以上下文为准。

【知识库上下文】
{context}"""


@dataclass
class RagResult:
    answer: str
    confidence: float
    tokens_used: int = 0
    sources: list[dict] = field(default_factory=list)
    stage_ms: dict = field(default_factory=dict)
    degraded: bool = False


async def run_rag(
    question: str,
    history: list[dict] | None = None,
    session_id: str = "",
    user_id: str = "",
) -> RagResult:
    history = history or []
    stage: dict[str, float] = {}
    t0 = time.perf_counter()

    # ---- 1. 查询改写：Multi-Query 与 HyDE 并行 ----
    t1 = time.perf_counter()
    queries, hyde_text = await asyncio.gather(
        multi_query(question, history, n=settings.multi_query_count),
        hyde(question) if settings.hyde_enabled else asyncio.sleep(0, result=""),
    )
    search_queries = list(dict.fromkeys([question, *queries, hyde_text] if hyde_text else [question, *queries]))
    stage["query_transform_ms"] = (time.perf_counter() - t1) * 1000

    # ---- 2. 混合检索（各查询并行） ----
    t2 = time.perf_counter()
    degraded = False
    try:
        hits = await hybrid_retriever.hybrid_search_multi(
            search_queries, top_k=settings.top_k_candidates
        )
    except Exception as exc:  # Milvus 故障 → PG 全文检索降级
        logger.warning("hybrid search degraded to PG fulltext: {}", exc)
        degraded = True
        hits = await hybrid_retriever.fallback_fulltext(question, top_k=settings.top_k_final)
        stage["retrieval_ms"] = (time.perf_counter() - t2) * 1000
        if not hits:
            return RagResult(answer=LOW_CONFIDENCE_ANSWER, confidence=0.0, stage_ms=stage, degraded=True)
        ranked = hits
        confidence = max(0.0, min(1.0, hits[0].get("rerank_score", 0.6)))
        # 直接进入生成
        return await _generate(question, ranked, confidence, stage, degraded)
    stage["retrieval_ms"] = (time.perf_counter() - t2) * 1000

    candidates = hybrid_retriever.attach_docs(hits)
    if not candidates:
        return RagResult(answer=LOW_CONFIDENCE_ANSWER, confidence=0.0, stage_ms=stage, degraded=degraded)

    # ---- 3. Rerank（CPU 密集 → 线程池） ----
    t3 = time.perf_counter()
    ranked = await asyncio.to_thread(
        reranker.rerank, question, candidates, settings.top_k_final
    )
    stage["rerank_ms"] = (time.perf_counter() - t3) * 1000
    if not ranked:
        return RagResult(answer=LOW_CONFIDENCE_ANSWER, confidence=0.0, stage_ms=stage, degraded=degraded)

    confidence = max(0.0, min(1.0, ranked[0].get("rerank_score", 0.0)))
    return await _generate(question, ranked, confidence, stage, degraded)


async def _generate(
    question: str,
    ranked: list[dict],
    confidence: float,
    stage: dict,
    degraded: bool,
) -> RagResult:
    t0 = time.perf_counter()
    context = "\n\n".join(
        f"[{i+1}] Q: {c['question']}\nA: {c['answer']}" for i, c in enumerate(ranked)
    )
    llm = get_llm()
    resp = await llm.ainvoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=f"用户问题：{question}"),
        ]
    )
    usage = extract_usage(resp)
    stage["llm_ms"] = (time.perf_counter() - t0) * 1000
    stage["total_ms"] = stage.get("query_transform_ms", 0) + stage.get("retrieval_ms", 0) + stage.get("rerank_ms", 0) + stage["llm_ms"]

    if usage["total_tokens"]:
        tokens_consumed_total.labels(model=settings.deepseek_model).inc(usage["total_tokens"])

    sources = [
        {
            "faq_id": c["faq_id"],
            "question": c["question"],
            "category": c.get("category", ""),
            "score": round(c.get("rerank_score", c.get("rrf_score", 0.0)), 4),
        }
        for c in ranked
    ]
    return RagResult(
        answer=resp.content.strip(),
        confidence=confidence,
        tokens_used=usage["total_tokens"],
        sources=sources,
        stage_ms=stage,
        degraded=degraded,
    )
