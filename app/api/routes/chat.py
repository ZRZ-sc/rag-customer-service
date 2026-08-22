"""聊天接口：POST /api/v1/chat 完整编排。

流程：会话历史 → 意图识别 → 路由分发（精确查询 / RAG）→ 缓存 → 响应 + 日志落库。
"""
import time
from uuid import uuid4

from fastapi import APIRouter, Depends

from app.api.deps import verify_token
from app.config import settings
from app.core.errors import ApiError
from app.core.logging import logger, request_id_var, session_id_var, user_id_var
from app.core.metrics import (
    chat_duration_seconds,
    chat_requests_total,
    concurrent_connections,
    precision_query_total,
    rag_query_total,
)
from app.db.postgres import get_pool
from app.models.chat import ChatData, ChatRequest, ChatResponse
from app.services import cache as cache_svc
from app.services import conversation as conv_svc
from app.services import precise_query
from app.services.intent import classify
from app.services.rag.pipeline import LOW_CONFIDENCE_ANSWER, run_rag

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, _: None = Depends(verify_token)) -> ChatResponse:
    t0 = time.perf_counter()
    user_id = req.user_id or "anonymous"
    session_id = req.session_id or uuid4().hex
    request_id_var.set(uuid4().hex[:16])
    user_id_var.set(user_id)
    session_id_var.set(session_id)
    concurrent_connections.inc()

    source = "rag"
    status = "ok"
    try:
        history = await conv_svc.get_history(session_id)

        # ---- 意图识别与路由 ----
        intent_res = await classify(req.question, history)
        intent = intent_res["intent"]
        logger.info("intent={} method={} entities={}", intent, intent_res["method"], intent_res["entities"])

        retrieval = None
        if intent == "precise_query":
            source = "database"
            precision_query_total.inc()
            data, retrieval = await _handle_precise(req, intent_res["entities"], user_id)
        else:
            rag_query_total.inc()
            data, retrieval = await _handle_knowledge(req, history, user_id)

        # ---- 多轮会话记录 ----
        await conv_svc.add_turn(session_id, "user", req.question)
        await conv_svc.add_turn(session_id, "assistant", data.answer)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        data.latency_ms = latency_ms
        chat_duration_seconds.observe(latency_ms / 1000.0)

        # ---- 问答日志落库（管理后台/统计用） ----
        await _log_dialogue(req, intent, data, latency_ms, retrieval)

        return ChatResponse(data=data)
    except ApiError:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        logger.exception("chat failed: {}", exc)
        raise
    finally:
        concurrent_connections.dec()
        chat_requests_total.labels(source=source, status=status).inc()


async def _handle_precise(req: ChatRequest, entities: dict, user_id: str) -> tuple[ChatData, dict | None]:
    """精确查询路由：订单号/物流单号 → PostgreSQL（30s 短缓存）。"""
    order_id = entities["order_ids"][0] if entities["order_ids"] else None
    tracking_no = entities["tracking_nos"][0] if entities["tracking_nos"] else None

    if order_id:
        key = cache_svc.precise_cache_key(order_id, user_id)
        cached = await cache_svc.get_cached_answer(key)
        if cached:
            return (
                ChatData(**cached, session_id=req.session_id, latency_ms=0),
                None,
            )
        res = await precise_query.query_by_order_id(order_id, user_id)
        payload = {
            "answer": res["answer"], "source": "database",
            "confidence": res["confidence"], "tokens_used": 0,
        }
        await cache_svc.set_cached_answer(key, payload, ttl=settings.precise_cache_seconds)
        return (
            ChatData(**payload, session_id=req.session_id, latency_ms=0),
            res.get("detail"),
        )

    if tracking_no:
        key = cache_svc.precise_cache_key(tracking_no, user_id)
        cached = await cache_svc.get_cached_answer(key)
        if cached:
            return (
                ChatData(**cached, session_id=req.session_id, latency_ms=0),
                None,
            )
        res = await precise_query.query_by_tracking_no(tracking_no, user_id)
        payload = {
            "answer": res["answer"], "source": "database",
            "confidence": res["confidence"], "tokens_used": 0,
        }
        await cache_svc.set_cached_answer(key, payload, ttl=settings.precise_cache_seconds)
        return (
            ChatData(**payload, session_id=req.session_id, latency_ms=0),
            res.get("detail"),
        )

    # 理论上不会走到：意图已判定为精确查询但未提取到实体
    raise ApiError(400, "未识别到订单号或物流单号，请检查输入")


async def _handle_knowledge(req: ChatRequest, history: list[dict], user_id: str) -> tuple[ChatData, list[dict] | None]:
    """知识类路由：Redis 缓存 → 未命中走完整 RAG 流程。"""
    key = cache_svc.answer_cache_key(req.question)
    cached = await cache_svc.get_cached_answer(key)
    if cached:
        return (
            ChatData(
                answer=cached["answer"],
                source="rag",
                confidence=cached["confidence"],
                session_id=req.session_id,
                tokens_used=cached.get("tokens_used", 0),
                latency_ms=0,
            ),
            None,
        )

    res = await run_rag(req.question, history, req.session_id, user_id)
    if res.confidence < settings.confidence_threshold:
        # 低置信度：不缓存，直接兜底话术（转人工入口由前端/后续扩展承接）
        return (
            ChatData(
                answer=LOW_CONFIDENCE_ANSWER,
                source="fallback",
                confidence=res.confidence,
                session_id=req.session_id,
                tokens_used=res.tokens_used,
                latency_ms=0,
            ),
            res.sources,
        )

    await cache_svc.set_cached_answer(
        key,
        {
            "answer": res.answer,
            "confidence": res.confidence,
            "tokens_used": res.tokens_used,
        },
    )
    logger.info("rag ok confidence={:.3f} stages={}", res.confidence, res.stage_ms)
    return (
        ChatData(
            answer=res.answer,
            source="rag",
            confidence=res.confidence,
            session_id=req.session_id,
            tokens_used=res.tokens_used,
            latency_ms=0,
        ),
        res.sources,
    )


async def _log_dialogue(
    req: ChatRequest,
    intent: str,
    data: ChatData,
    latency_ms: int,
    retrieval: dict | list | None,
) -> None:
    """写入 dialogue_logs（失败不影响主流程）。"""
    import json

    try:
        pool = await get_pool()
        await pool.execute(
            """
            INSERT INTO dialogue_logs
                (request_id, session_id, user_id, question, intent, retrieval_results,
                 answer, source, confidence, latency_ms, tokens_used, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11, NOW())
            """,
            request_id_var.get(),
            req.session_id,
            req.user_id or "anonymous",
            req.question,
            intent,
            json.dumps(retrieval, ensure_ascii=False) if retrieval else None,
            data.answer,
            data.source,
            data.confidence,
            latency_ms,
            data.tokens_used,
        )
    except Exception as exc:
        logger.warning("dialogue log write failed: {}", exc)
