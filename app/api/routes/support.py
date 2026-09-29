"""客服支撑接口：转人工（F6）与用户反馈。

| 接口 | 说明 |
|------|------|
| POST /api/v1/handoff | 用户主动转人工（也可由问答低置信度自动触发，见 chat 流程） |
| GET  /api/v1/handoff/{ticket_id} | 查询工单状态 |
| POST /api/v1/feedback | 对某次问答点赞/点踩（request_id 来自问答响应体或 X-Request-ID 头） |

均需 Bearer 鉴权，并受同一套限流保护（scope: handoff / feedback）。
"""
from fastapi import APIRouter, Depends, Request

from app.api.deps import verify_token
from app.config import settings
from app.core.errors import ForbiddenError, NotFoundError
from app.core.logging import logger
from app.models.support import (
    FeedbackData,
    FeedbackRequest,
    FeedbackResponse,
    HandoffData,
    HandoffRequest,
    HandoffResponse,
)
from app.services import rate_limit
from app.services import support as support_svc

router = APIRouter(prefix="/api/v1", tags=["support"])


@router.post("/handoff", response_model=HandoffResponse, summary="转人工（创建工单）")
async def create_handoff(
    req: HandoffRequest,
    request: Request,
    _: None = Depends(verify_token),
) -> HandoffResponse:
    if not settings.human_handoff_enabled:
        raise ForbiddenError("转人工功能当前未启用")

    await rate_limit.enforce("handoff", rate_limit.identity_of(req.user_id, request))

    ticket = await support_svc.create_ticket(
        session_id=req.session_id,
        user_id=req.user_id,
        reason=req.reason,
        last_question=req.last_question,
        source="manual",
        priority=req.priority,
    )
    position = await support_svc.pending_count()
    return HandoffResponse(
        message="已为您转接人工客服",
        data=HandoffData(
            ticket_id=ticket["ticket_id"],
            status=ticket["status"],
            priority=ticket["priority"],
            source=ticket["source"],
            created_at=str(ticket["created_at"]),
            queue_hint=support_svc.queue_hint(position),
        ),
    )


@router.get("/handoff/{ticket_id}", response_model=HandoffResponse, summary="查询工单状态")
async def get_handoff(ticket_id: str, _: None = Depends(verify_token)) -> HandoffResponse:
    ticket = await support_svc.get_ticket(ticket_id)
    if ticket is None:
        raise NotFoundError(f"未找到工单 {ticket_id}")
    return HandoffResponse(
        data=HandoffData(
            ticket_id=ticket["ticket_id"],
            status=ticket["status"],
            priority=ticket["priority"],
            source=ticket["source"],
            created_at=str(ticket["created_at"]),
            queue_hint=None,
        )
    )


@router.post("/feedback", response_model=FeedbackResponse, summary="问答反馈（点赞/点踩）")
async def submit_feedback(
    req: FeedbackRequest,
    request: Request,
    _: None = Depends(verify_token),
) -> FeedbackResponse:
    await rate_limit.enforce("feedback", rate_limit.identity_of(req.user_id, request))

    status, data = await support_svc.save_feedback(
        request_id=req.request_id,
        user_id=req.user_id,
        value=req.feedback,
        comment=req.comment,
    )
    if status == "not_found":
        raise NotFoundError(f"未找到问答记录 {req.request_id}")
    if status == "forbidden":
        raise ForbiddenError("无权限反馈该问答记录")

    logger.info("feedback | request_id={} value={}", req.request_id, req.feedback)
    return FeedbackResponse(
        data=FeedbackData(
            request_id=data["request_id"],
            feedback=data["feedback"],
            recorded_at=data["recorded_at"],
        )
    )
