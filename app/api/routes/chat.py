"""聊天接口：POST /api/v1/chat 完整编排。

流程：限流 → 会话历史 → **上下文查询改写（指代消解）** → 意图识别 → 路由分发（精确查询 / RAG）
      → 缓存 → 低置信度自动转人工 → 响应 + 日志落库。
"""
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, Request

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
from app.models.chat import ChatData, ChatRequest, ChatResponse, SourceItem
from app.services import cache as cache_svc
from app.services import conversation as conv_svc
from app.services import precise_query
from app.services import rate_limit
from app.services import support as support_svc
from app.services.intent import classify
from app.services.query_rewrite import is_standalone, rewrite_query
from app.services.rag.pipeline import LOW_CONFIDENCE_ANSWER, run_rag

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    _: None = Depends(verify_token),
) -> ChatResponse:
    t0 = time.perf_counter()
    user_id = req.user_id or "anonymous"
    session_id = req.session_id or uuid4().hex
    request_id = uuid4().hex[:16]
    request_id_var.set(request_id)
    user_id_var.set(user_id)
    session_id_var.set(session_id)
    concurrent_connections.inc()

    source = "rag"
    status = "ok"
    try:
        # ---- 限流（429）：放在业务逻辑之前，超限直接卸载 ----
        await rate_limit.enforce("chat", rate_limit.identity_of(req.user_id, request))

        # ---- L1 快速探测：独立问题（无代词/省略）的改写结果就是原问题本身，
        #      因此可直接用原问题查答案缓存，省去历史/实体读取与改写流程（~5ms 命中）
        if is_standalone(req.question):
            early = await cache_svc.get_cached_answer(
                cache_svc.answer_cache_key(req.question)
            )
            if early:
                data = ChatData(
                    answer=early["answer"],
                    source="rag",
                    confidence=early["confidence"],
                    session_id=req.session_id,
                    tokens_used=early.get("tokens_used", 0),
                    latency_ms=0,
                    request_id=request_id,
                )
                await conv_svc.add_turns(
                    session_id, [("user", req.question), ("assistant", data.answer)]
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                data.latency_ms = latency_ms
                data.intent = "knowledge"
                chat_duration_seconds.observe(latency_ms / 1000.0)
                await _log_dialogue(req, "knowledge", data, latency_ms, {"cache": {"answer": "hit"}})
                logger.info("L1 answer cache HIT (early probe) latency_ms={}", latency_ms)
                return ChatResponse(data=data)

        history = await conv_svc.get_history(session_id)
        last_entities = await conv_svc.get_last_entities(session_id, user_id)

        # ---- 上下文查询改写（指代消解）：必须在意图识别之前 ----
        # 例："那什么时候能到？" + 历史中的订单号 → "订单 ORD20240100001 什么时候能到？"
        rewrite_res = await rewrite_query(req.question, history, last_entities)
        effective_question = rewrite_res["rewritten"]

        # ---- 意图识别与路由（基于改写后的问题） ----
        intent_res = await classify(effective_question, history)
        intent = intent_res["intent"]
        # 兜底：精确查询却没有可检索的单号时降级走知识库（防御性，classify 已处理）
        if intent == "precise_query" and not (
            intent_res["entities"]["order_ids"] or intent_res["entities"]["tracking_nos"]
        ):
            logger.info("intent=precise_query 但无单号实体 -> 降级为 knowledge")
            intent = "knowledge"
        logger.info(
            "intent={} method={} entities={} rewritten={}",
            intent, intent_res["method"], intent_res["entities"], rewrite_res["applied"],
        )

        retrieval = None
        if intent == "precise_query":
            source = "database"
            precision_query_total.inc()
            # 记住本次会话涉及的实体，供后续追问消解使用（与当前用户绑定）
            await conv_svc.set_last_entities(session_id, intent_res["entities"], user_id)
            data, retrieval = await _handle_precise(req, intent_res["entities"], user_id)
        else:
            rag_query_total.inc()
            data, retrieval = await _handle_knowledge(req, history, user_id, effective_question)

        # ---- 多轮会话记录（保留用户原始表述，实体记忆另行维护；批量写入省 2 次往返） ----
        await conv_svc.add_turns(
            session_id, [("user", req.question), ("assistant", data.answer)]
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        data.latency_ms = latency_ms
        data.intent = intent
        data.request_id = request_id
        # 知识类回答回传命中的 FAQ 条目，前端展示"答案依据"（缓存命中时无检索详情则为空）
        if retrieval and retrieval.get("sources"):
            try:
                data.sources = [SourceItem(**s) for s in retrieval["sources"]]
            except Exception as exc:  # 证据字段不影响主回答
                logger.warning("build sources failed: {}", exc)
        if rewrite_res["applied"]:
            data.rewritten_question = effective_question

        # ---- F6：低置信度自动转人工（无法回答时建单，接口为 /api/v1/handoff） ----
        if data.source == "fallback" and settings.handoff_auto_on_low_confidence:
            try:
                ticket = await support_svc.create_ticket(
                    session_id=session_id,
                    user_id=req.user_id,
                    reason=f"低置信度自动转人工（confidence={data.confidence:.3f}）",
                    last_question=req.question,
                    source="auto",
                    priority="high" if data.confidence < 0.2 else "normal",
                )
                data.handoff_ticket_id = ticket["ticket_id"]
                data.answer = f"{data.answer}（已为您转接人工客服，工单号 {ticket['ticket_id']}）"
            except Exception as exc:
                logger.warning("auto handoff failed: {}", exc)

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


async def _handle_knowledge(
    req: ChatRequest,
    history: list[dict],
    user_id: str,
    question: str | None = None,
) -> tuple[ChatData, list[dict] | None]:
    """知识类路由：Redis 缓存 → 未命中走完整 RAG 流程。

    question 为上下文改写后的问题（缺省时用原始问题），检索与缓存键都基于它，
    这样"那运费呢？"这类追问也能命中正确的知识条目。
    """
    question = question or req.question
    key = cache_svc.answer_cache_key(question)
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

    res = await run_rag(question, history, req.session_id, user_id)
    # 阶段耗时 / 路径 / 缓存命中情况随日志落库，便于延迟分析与后续管理后台展示
    detail = {
        "sources": res.sources,
        "stage_ms": {k: round(v, 1) for k, v in res.stage_ms.items()},
        "cache": res.cache,
        "path": res.path,
    }
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
            detail,
        )

    await cache_svc.set_cached_answer(
        key,
        {
            "answer": res.answer,
            "confidence": res.confidence,
            "tokens_used": res.tokens_used,
        },
    )
    logger.info(
        "rag ok path={} confidence={:.3f} cache={} stages={}",
        res.path, res.confidence, res.cache, res.stage_ms,
    )
    return (
        ChatData(
            answer=res.answer,
            source="rag",
            confidence=res.confidence,
            session_id=req.session_id,
            tokens_used=res.tokens_used,
            latency_ms=0,
        ),
        detail,
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

    if not settings.dialogue_log_enabled:
        # 高并发压测/降载时可关闭，省去每请求一次 PG 写入
        return

    try:
        pool = await get_pool()
        row = {
            "request_id": request_id_var.get(),
            "session_id": req.session_id,
            "user_id": req.user_id or "anonymous",
            "question": req.question,
            "rewritten": data.rewritten_question,
            "intent": intent,
            "retrieval": json.dumps(retrieval, ensure_ascii=False) if retrieval else None,
            "answer": data.answer,
            "source": data.source,
            "confidence": data.confidence,
            "latency_ms": latency_ms,
            "tokens_used": data.tokens_used,
        }
        try:
            await pool.execute(
                """
                INSERT INTO dialogue_logs
                    (request_id, session_id, user_id, question, rewritten_question, intent,
                     retrieval_results, answer, source, confidence, latency_ms, tokens_used, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12, NOW())
                """,
                row["request_id"], row["session_id"], row["user_id"], row["question"],
                row["rewritten"], row["intent"], row["retrieval"], row["answer"],
                row["source"], row["confidence"], row["latency_ms"], row["tokens_used"],
            )
        except Exception:
            # 兼容旧表结构（无 rewritten_question 列）
            await pool.execute(
                """
                INSERT INTO dialogue_logs
                    (request_id, session_id, user_id, question, intent, retrieval_results,
                     answer, source, confidence, latency_ms, tokens_used, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11, NOW())
                """,
                row["request_id"], row["session_id"], row["user_id"], row["question"],
                row["intent"], row["retrieval"], row["answer"], row["source"],
                row["confidence"], row["latency_ms"], row["tokens_used"],
            )
    except Exception as exc:
        logger.warning("dialogue log write failed: {}", exc)
