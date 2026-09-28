"""RAG 主流程编排（需求文档流程2）+ 阶段3 缓存与快速路径优化。

流程（两条路径）：
  ① 基础混合检索（仅原问题，无 LLM，~300ms）
     ├─ 高置信（top1 向量相似度 ≥ 阈值且与 top2 有明显分差）→ **快速路径**：
     │    只对 top-5 候选做重排 → 生成            （CPU 机器上首答 < 2s）
     └─ 否则 → **完整路径**：
          查询扩展 Multi-Query+HyDE（1 次 LLM 调用，24h 缓存）
          → 多查询混合检索(RRF) → Rerank（结果 24h 缓存，候选上限 12）→ 生成

降级：Milvus 故障 → PostgreSQL ILIKE 全文检索；Redis 故障 → 缓存旁路；LLM 故障 → 兜底话术。
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
    rag_fast_path_total,
    rag_retrieval_duration,
    rerank_duration,
    tokens_consumed_total,
)
from app.services import cache as cache_svc
from app.services.llm import extract_usage, get_llm
from app.services.rag.query_transform import transform_queries
from app.services.rag.reranker import reranker
from app.services.rag.retriever import HybridRetriever, hybrid_retriever

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
    path: str = "full"                       # fast | full | fallback
    cache: dict = field(default_factory=dict)  # {"transform": hit|miss|skipped, "rerank": hit|miss}


def _fast_path_eligible(hits: list[dict], candidates: list[dict]) -> bool:
    """高置信判定：top1 向量相似度足够高，且与 top2 有明显分差。

    这类问题（FAQ 原句或近似问法）向量检索已能精准命中，无需 LLM 扩展与全量重排。
    """
    if not settings.fast_path_enabled:
        return False
    if len(candidates) < 3:
        return False
    scores = HybridRetriever.vector_scores(hits)
    if not scores or scores[0] < settings.fast_path_min_score:
        return False
    if len(scores) > 1 and (scores[0] - scores[1]) < settings.fast_path_min_margin:
        return False
    return True


async def _rerank_with_cache(
    question: str, candidates: list[dict], top_k: int
) -> tuple[list[dict], str]:
    """带缓存的重排：key = rerank:{sha1(query)}:{sha1(doc_ids)}。

    跨编码器得分是 (query, doc) 的纯函数，候选集合不变时可直接复用；
    知识库更新时 ingest 脚本会清除 rerank:* 前缀。
    """
    if not candidates:
        return [], "miss"
    doc_ids = [c["faq_id"] for c in candidates]
    key = cache_svc.rerank_cache_key(question, doc_ids)
    cached = await cache_svc.get_json(key, name="rerank")
    if cached and cached.get("scores"):
        score_map = {int(k): float(v) for k, v in cached["scores"].items()}
        for c in candidates:
            c["rerank_score"] = score_map.get(c["faq_id"], 0.0)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        return ranked, "hit"

    ranked = await asyncio.to_thread(reranker.rerank, question, candidates, top_k)
    await cache_svc.set_json(
        key,
        {"scores": {str(c["faq_id"]): float(c.get("rerank_score", 0.0)) for c in candidates}},
        ttl=settings.rerank_cache_ttl_seconds,
        name="rerank",
    )
    return ranked, "miss"


async def run_rag(
    question: str,
    history: list[dict] | None = None,
    session_id: str = "",
    user_id: str = "",
) -> RagResult:
    stage: dict[str, float] = {}
    cache_flags: dict[str, str] = {}
    degraded = False

    # ---- 1. 基础混合检索（仅原问题，不调 LLM） ----
    t = time.perf_counter()
    try:
        base_hits = await hybrid_retriever.hybrid_search(
            question, top_k=settings.top_k_candidates
        )
    except Exception as exc:  # Milvus 故障 → PG 全文检索降级
        logger.warning("hybrid search degraded to PG fulltext: {}", exc)
        degraded = True
        hits = await hybrid_retriever.fallback_fulltext(question, top_k=settings.top_k_final)
        stage["retrieval_ms"] = (time.perf_counter() - t) * 1000
        rag_retrieval_duration.observe(stage["retrieval_ms"] / 1000.0)
        if not hits:
            return RagResult(
                answer=LOW_CONFIDENCE_ANSWER, confidence=0.0, stage_ms=stage,
                degraded=True, path="fallback",
            )
        confidence = max(0.0, min(1.0, hits[0].get("rerank_score", 0.6)))
        return await _generate(question, hits, confidence, stage, degraded, "fallback", cache_flags)

    base_candidates = hybrid_retriever.attach_docs(base_hits)
    base_retrieval_ms = (time.perf_counter() - t) * 1000
    rag_retrieval_duration.observe(base_retrieval_ms / 1000.0)
    stage["base_retrieval_ms"] = base_retrieval_ms
    stage["retrieval_ms"] = base_retrieval_ms

    # ---- 2. 路径选择 ----
    if _fast_path_eligible(base_hits, base_candidates):
        path = "fast"
        rag_fast_path_total.labels(result="fast").inc()
        cache_flags["transform"] = "skipped"
        logger.info(
            "rag fast path | top1_vector_score={:.3f}",
            HybridRetriever.vector_scores(base_hits)[0],
        )
        t = time.perf_counter()
        ranked, rerank_cache = await _rerank_with_cache(
            question, base_candidates[: settings.fast_path_rerank_topk], settings.top_k_final
        )
        cache_flags["rerank"] = rerank_cache
        stage["rerank_ms"] = (time.perf_counter() - t) * 1000
        rerank_duration.observe(stage["rerank_ms"] / 1000.0)
    else:
        path = "full"
        rag_fast_path_total.labels(result="full").inc()

        # ---- 3. 查询扩展（一次 LLM 调用产出 Multi-Query + HyDE，24h 缓存） ----
        t = time.perf_counter()
        transform = await transform_queries(question)
        cache_flags["transform"] = "hit" if transform.get("cached") else "miss"
        search_queries = list(
            dict.fromkeys([*transform["queries"], transform.get("hyde") or ""])
        )
        search_queries = [q for q in search_queries if q]
        stage["query_transform_ms"] = (time.perf_counter() - t) * 1000
        query_transform_duration.observe(stage["query_transform_ms"] / 1000.0)

        # ---- 4. 多查询混合检索（各查询并行 + RRF 融合） ----
        t = time.perf_counter()
        try:
            hits = await hybrid_retriever.hybrid_search_multi(
                search_queries, top_k=settings.top_k_candidates
            )
        except Exception as exc:
            logger.warning("multi-query retrieval degraded to PG fulltext: {}", exc)
            degraded = True
            hits = await hybrid_retriever.fallback_fulltext(question, top_k=settings.top_k_final)
        multi_retrieval_ms = (time.perf_counter() - t) * 1000
        rag_retrieval_duration.observe(multi_retrieval_ms / 1000.0)
        stage["retrieval_ms"] += multi_retrieval_ms

        candidates = hybrid_retriever.attach_docs(hits)
        if not candidates:
            return RagResult(
                answer=LOW_CONFIDENCE_ANSWER, confidence=0.0, stage_ms=stage,
                degraded=degraded, path="fallback", cache=cache_flags,
            )

        # ---- 5. 重排（候选上限 + 24h 缓存） ----
        t = time.perf_counter()
        ranked, rerank_cache = await _rerank_with_cache(
            question, candidates[: settings.rerank_max_candidates], settings.top_k_final
        )
        cache_flags["rerank"] = rerank_cache
        stage["rerank_ms"] = (time.perf_counter() - t) * 1000
        rerank_duration.observe(stage["rerank_ms"] / 1000.0)

    if not ranked:
        return RagResult(
            answer=LOW_CONFIDENCE_ANSWER, confidence=0.0, stage_ms=stage,
            degraded=degraded, path="fallback", cache=cache_flags,
        )

    confidence = max(0.0, min(1.0, ranked[0].get("rerank_score", 0.0)))
    return await _generate(question, ranked, confidence, stage, degraded, path, cache_flags)


async def _generate(
    question: str,
    ranked: list[dict],
    confidence: float,
    stage: dict,
    degraded: bool,
    path: str = "full",
    cache_flags: dict | None = None,
) -> RagResult:
    t = time.perf_counter()
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
    stage["llm_ms"] = (time.perf_counter() - t) * 1000
    llm_generation_duration.observe(stage["llm_ms"] / 1000.0)
    stage["total_ms"] = sum(stage.get(k, 0.0) for k in ("query_transform_ms", "retrieval_ms", "rerank_ms", "llm_ms"))

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
        path=path,
        cache=cache_flags or {},
    )
