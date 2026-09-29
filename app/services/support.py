"""转人工工单（F6）与用户反馈（F7 反馈闭环）。

- 工单持久化在 PostgreSQL（handoff_tickets），状态 pending/accepted/closed
- 预留 Webhook：配置 HANDOFF_WEBHOOK_URL 后，建单会异步推送到人工客服系统（失败不影响主流程）
- 反馈写入 dialogue_logs（feedback / feedback_comment / feedback_at），并做归属校验防篡改
"""
import asyncio
from datetime import datetime
from uuid import uuid4

import httpx

from app.config import settings
from app.core.logging import logger
from app.core.metrics import feedback_total, handoff_tickets_total
from app.db.postgres import get_pool

# 防止后台通知任务被 GC 回收
_background_tasks: set[asyncio.Task] = set()

_FEEDBACK_LABEL = {-1: "negative", 0: "neutral", 1: "positive"}


def new_ticket_id() -> str:
    return f"HO{datetime.now():%Y%m%d}{uuid4().hex[:8].upper()}"


async def create_ticket(
    session_id: str,
    user_id: str | None,
    reason: str | None = None,
    last_question: str | None = None,
    source: str = "manual",
    priority: str = "normal",
) -> dict:
    """创建转人工工单。source: manual(用户主动) / auto(低置信度自动)。"""
    pool = await get_pool()
    ticket_id = new_ticket_id()
    row = await pool.fetchrow(
        """
        INSERT INTO handoff_tickets
            (ticket_id, session_id, user_id, reason, last_question, source, priority, status)
        VALUES ($1,$2,$3,$4,$5,$6,$7,'pending')
        RETURNING ticket_id, status, priority, source, created_at
        """,
        ticket_id, session_id, user_id, reason, last_question, source, priority,
    )
    ticket = dict(row)
    ticket["session_id"] = session_id
    ticket["user_id"] = user_id
    handoff_tickets_total.labels(source=source, priority=priority).inc()
    logger.info(
        "handoff ticket created | ticket_id={} session={} user={} source={} priority={}",
        ticket_id, session_id, user_id, source, priority,
    )
    _dispatch_webhook(ticket)
    return ticket


async def get_ticket(ticket_id: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow(
        """SELECT ticket_id, session_id, user_id, reason, last_question, source,
                  priority, status, created_at, updated_at
             FROM handoff_tickets WHERE ticket_id = $1""",
        ticket_id,
    )
    return dict(row) if row else None


async def pending_count() -> int:
    """排队中的工单数（用于给用户展示排队提示）。"""
    pool = await get_pool()
    return int(await pool.fetchval(
        "SELECT count(*) FROM handoff_tickets WHERE status = 'pending'"
    ) or 0)


def queue_hint(position: int) -> str:
    """根据排队位置给出预计等待时间（粗估：每位约 30 秒）。"""
    if position <= 1:
        return "已进入人工队列，客服将尽快接入"
    return f"已进入人工队列，前方约 {position - 1} 位，预计等待 {max(1, (position - 1) * 30 // 60)} 分钟"


async def save_feedback(
    request_id: str, user_id: str | None, value: int, comment: str | None
) -> tuple[str, dict | None]:
    """写入用户反馈。返回 (状态, 数据)：

    - updated   ：成功
    - not_found ：request_id 不存在（对方未记录该请求）
    - forbidden ：反馈人不属于该问答记录

    注意：必须先做归属校验再执行 UPDATE。若先更新后校验，越权请求虽然对外返回 403，
    数据却已被改写（实测踩过该坑）。
    """
    pool = await get_pool()

    owner_row = await pool.fetchrow(
        "SELECT user_id FROM dialogue_logs WHERE request_id = $1", request_id
    )
    if owner_row is None:
        return "not_found", None

    owner = owner_row["user_id"]
    if user_id and owner and owner != "anonymous" and owner != user_id:
        logger.warning(
            "feedback rejected (owner mismatch) | request_id={} owner={} requester={}",
            request_id, owner, user_id,
        )
        return "forbidden", None

    row = await pool.fetchrow(
        """
        UPDATE dialogue_logs
           SET feedback = $1, feedback_comment = $2, feedback_at = NOW()
         WHERE request_id = $3
        RETURNING user_id, session_id, feedback, feedback_at
        """,
        value, comment, request_id,
    )
    if row is None:      # 极端情况下记录被并发删除
        return "not_found", None

    feedback_total.labels(value=_FEEDBACK_LABEL.get(value, "unknown")).inc()
    logger.info(
        "feedback saved | request_id={} user={} value={} comment={}",
        request_id, owner, value, bool(comment),
    )
    return "updated", {
        "request_id": request_id,
        "feedback": row["feedback"],
        "recorded_at": row["feedback_at"].isoformat() if row["feedback_at"] else None,
    }


# ---- Webhook（预留接口） ----


def _dispatch_webhook(ticket: dict) -> None:
    """异步推送工单，不阻塞响应；未配置 URL 时直接跳过。"""
    if not settings.handoff_webhook_url:
        logger.debug("handoff webhook not configured, skip notify")
        return
    task = asyncio.create_task(_notify_webhook(ticket))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _notify_webhook(ticket: dict) -> None:
    payload = {
        "event": "handoff.created",
        "ticket_id": ticket.get("ticket_id"),
        "session_id": ticket.get("session_id"),
        "user_id": ticket.get("user_id"),
        "reason": ticket.get("reason"),
        "last_question": ticket.get("last_question"),
        "priority": ticket.get("priority"),
        "source": ticket.get("source"),
        "created_at": str(ticket.get("created_at")),
    }
    try:
        async with httpx.AsyncClient(timeout=settings.handoff_webhook_timeout) as client:
            resp = await client.post(settings.handoff_webhook_url, json=payload)
        logger.info(
            "handoff webhook notified | ticket_id={} status={}",
            ticket.get("ticket_id"), resp.status_code,
        )
    except Exception as exc:
        # 通知失败不影响已建单，仅告警（后续可由管理后台补推）
        logger.warning(
            "handoff webhook failed | ticket_id={} error={}", ticket.get("ticket_id"), exc
        )
