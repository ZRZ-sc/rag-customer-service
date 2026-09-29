"""管理后台只读接口（F7）。

| 接口 | 说明 |
|------|------|
| GET /api/v1/admin/logs | 问答日志分页查询（按用户/会话/意图/来源/时间过滤） |
| GET /api/v1/admin/stats | 统计：QPS、缓存命中率、意图分布、转人工数、延迟分位、反馈 |
| GET /api/v1/admin/handoffs | 转人工工单分页查询（按状态/来源/用户/时间过滤） |

鉴权：`Authorization: Bearer <ADMIN_TOKEN>`（与业务 API_TOKEN 分离）。
统计口径：核心指标来自 PostgreSQL（dialogue_logs / handoff_tickets），
缓存命中率与快速路径占比来自 Prometheus 运行时指标（不可用时返回 null，不影响其余字段）。
"""
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, Query

from app.api.deps import verify_admin
from app.config import settings
from app.core.logging import logger
from app.db.postgres import get_pool
from app.models.admin import (
    AdminStats,
    CacheStats,
    DialogueLogItem,
    FeedbackStats,
    HandoffItem,
    HandoffStats,
    HandoffsResponse,
    LatencyStats,
    LogsResponse,
    PaginatedHandoffs,
    PaginatedLogs,
    StatsResponse,
)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

MAX_PAGE_SIZE = 100


def _paginate(total: int, page: int, page_size: int) -> dict:
    pages = (total + page_size - 1) // page_size if page_size else 0
    return {"total": total, "page": page, "page_size": page_size, "pages": pages}


def _window(window_minutes: int) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=window_minutes)


# ---------------- 问答日志 ----------------


@router.get("/logs", response_model=LogsResponse, summary="问答日志（分页 + 过滤）")
async def list_logs(
    _: None = Depends(verify_admin),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
    user_id: str | None = Query(None, description="按用户过滤"),
    session_id: str | None = Query(None, description="按会话过滤"),
    intent: str | None = Query(None, description="precise_query / knowledge / other"),
    source: str | None = Query(None, description="database / rag / fallback"),
    start: datetime | None = Query(None, description="起始时间（含）"),
    end: datetime | None = Query(None, description="结束时间（含）"),
) -> LogsResponse:
    where, args = ["TRUE"], []
    for column, value in (
        ("user_id", user_id), ("session_id", session_id),
        ("intent", intent), ("source", source),
    ):
        if value:
            args.append(value)
            where.append(f"{column} = ${len(args)}")
    if start:
        args.append(start.replace(tzinfo=None))
        where.append(f"created_at >= ${len(args)}")
    if end:
        args.append(end.replace(tzinfo=None))
        where.append(f"created_at <= ${len(args)}")

    clause = " AND ".join(where)
    pool = await get_pool()
    total = await pool.fetchval(f"SELECT count(*) FROM dialogue_logs WHERE {clause}", *args)

    args_page = [*args, page_size, (page - 1) * page_size]
    rows = await pool.fetch(
        f"""
        SELECT id, request_id, session_id, user_id, question, rewritten_question, intent,
               source, answer, confidence, latency_ms, tokens_used, feedback, feedback_comment,
               created_at
          FROM dialogue_logs
         WHERE {clause}
         ORDER BY id DESC
         LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}
        """,
        *args_page,
    )
    items = [
        DialogueLogItem(
            **{**dict(r), "created_at": r["created_at"].isoformat() if r["created_at"] else None}
        )
        for r in rows
    ]
    return LogsResponse(data=PaginatedLogs(**_paginate(int(total), page, page_size), items=items))


# ---------------- 转人工工单 ----------------


@router.get("/handoffs", response_model=HandoffsResponse, summary="转人工工单（分页 + 过滤）")
async def list_handoffs(
    _: None = Depends(verify_admin),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
    status: str | None = Query(None, description="pending / accepted / closed"),
    source: str | None = Query(None, description="manual / auto"),
    user_id: str | None = Query(None),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
) -> HandoffsResponse:
    where, args = ["TRUE"], []
    for column, value in (("status", status), ("source", source), ("user_id", user_id)):
        if value:
            args.append(value)
            where.append(f"{column} = ${len(args)}")
    if start:
        args.append(start.replace(tzinfo=None))
        where.append(f"created_at >= ${len(args)}")
    if end:
        args.append(end.replace(tzinfo=None))
        where.append(f"created_at <= ${len(args)}")

    clause = " AND ".join(where)
    pool = await get_pool()
    total = await pool.fetchval(f"SELECT count(*) FROM handoff_tickets WHERE {clause}", *args)
    rows = await pool.fetch(
        f"""
        SELECT ticket_id, session_id, user_id, reason, last_question, source, priority, status,
               created_at, updated_at
          FROM handoff_tickets
         WHERE {clause}
         ORDER BY created_at DESC
         LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}
        """,
        *args, page_size, (page - 1) * page_size,
    )
    items = [
        HandoffItem(**{
            **dict(r),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
        })
        for r in rows
    ]
    return HandoffsResponse(
        data=PaginatedHandoffs(**_paginate(int(total), page, page_size), items=items)
    )


# ---------------- 统计 ----------------


async def _prometheus_scalar(client: httpx.AsyncClient, expr: str) -> float | None:
    """查询 Prometheus 单个标量；不可用时返回 None（降级，不影响其余统计）。"""
    if not settings.prometheus_url:
        return None
    try:
        resp = await client.get(
            f"{settings.prometheus_url.rstrip('/')}/api/v1/query", params={"query": expr}
        )
        result = resp.json().get("data", {}).get("result") or []
        if not result:
            return None
        return float(result[0]["value"][1])
    except Exception:
        return None


@router.get("/stats", response_model=StatsResponse, summary="统计（QPS/缓存/意图/转人工/反馈）")
async def get_stats(
    _: None = Depends(verify_admin),
    window_minutes: int = Query(60, ge=1, le=10080, description="统计窗口（分钟）"),
) -> StatsResponse:
    since = _window(window_minutes)
    pool = await get_pool()

    # ---- 问答日志聚合（数据库为准，始终可用） ----
    total = int(await pool.fetchval(
        "SELECT count(*) FROM dialogue_logs WHERE created_at >= $1", since
    ) or 0)
    source_rows = await pool.fetch(
        "SELECT coalesce(source,'unknown') AS k, count(*) AS n FROM dialogue_logs "
        "WHERE created_at >= $1 GROUP BY 1", since
    )
    intent_rows = await pool.fetch(
        "SELECT coalesce(intent,'unknown') AS k, count(*) AS n FROM dialogue_logs "
        "WHERE created_at >= $1 GROUP BY 1", since
    )
    status_rows = await pool.fetch(
        "SELECT CASE WHEN source='fallback' THEN 'fallback' ELSE 'ok' END AS k, count(*) AS n "
        "FROM dialogue_logs WHERE created_at >= $1 GROUP BY 1", since
    )
    latency = await pool.fetchrow(
        """
        SELECT coalesce(avg(latency_ms),0) AS avg_ms, coalesce(max(latency_ms),0) AS max_ms,
               coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms),0) AS p50,
               coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),0) AS p95,
               coalesce(percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms),0) AS p99
          FROM dialogue_logs WHERE created_at >= $1 AND latency_ms IS NOT NULL
        """,
        since,
    )
    tokens_total = int(await pool.fetchval(
        "SELECT coalesce(sum(tokens_used),0) FROM dialogue_logs WHERE created_at >= $1", since
    ) or 0)
    feedback_row = await pool.fetchrow(
        """
        SELECT count(*) FILTER (WHERE feedback = 1)  AS up,
               count(*) FILTER (WHERE feedback = -1) AS down
          FROM dialogue_logs WHERE created_at >= $1
        """,
        since,
    )

    # ---- 转人工工单 ----
    handoff_row = await pool.fetchrow(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE status = 'pending') AS pending,
               count(*) FILTER (WHERE source = 'auto')    AS auto,
               count(*) FILTER (WHERE source = 'manual')  AS manual
          FROM handoff_tickets WHERE created_at >= $1
        """,
        since,
    )

    # ---- Prometheus 运行时指标（缓存命中率 / 快速路径占比）----
    cache_stats = CacheStats()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            hit_rate = await _prometheus_scalar(
                client,
                "sum(rate(cache_hits_total[10m])) / clamp_min("
                "sum(rate(cache_hits_total[10m])) + sum(rate(cache_misses_total[10m])), 0.001)",
            )
            fast_ratio = await _prometheus_scalar(
                client,
                'sum(rate(rag_fast_path_total{result="fast"}[10m])) / clamp_min('
                "sum(rate(rag_fast_path_total[10m])), 0.001)",
            )
            cache_stats = CacheStats(hit_rate=hit_rate, fast_path_ratio=fast_ratio)
    except Exception as exc:
        logger.warning("admin stats: prometheus enrichment failed: {}", exc)

    up, down = int(feedback_row["up"]), int(feedback_row["down"])
    positive_rate = (up / (up + down)) if (up + down) else None

    data = AdminStats(
        window_minutes=window_minutes,
        total_requests=total,
        qps=round(total / (window_minutes * 60), 4),
        by_source={r["k"]: int(r["n"]) for r in source_rows},
        by_intent={r["k"]: int(r["n"]) for r in intent_rows},
        by_status={r["k"]: int(r["n"]) for r in status_rows},
        latency=LatencyStats(
            avg_ms=round(float(latency["avg_ms"]), 1),
            p50_ms=round(float(latency["p50"]), 1),
            p95_ms=round(float(latency["p95"]), 1),
            p99_ms=round(float(latency["p99"]), 1),
            max_ms=float(latency["max_ms"]),
        ),
        tokens_total=tokens_total,
        handoffs=HandoffStats(
            total=int(handoff_row["total"]), pending=int(handoff_row["pending"]),
            auto=int(handoff_row["auto"]), manual=int(handoff_row["manual"]),
        ),
        feedback=FeedbackStats(up=up, down=down, positive_rate=positive_rate),
        cache=cache_stats,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    return StatsResponse(data=data)
