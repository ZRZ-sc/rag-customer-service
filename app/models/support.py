"""转人工（F6）与用户反馈的请求/响应模型。"""
from typing import Literal

from pydantic import BaseModel, Field


# ---- 转人工 ----


class HandoffRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, description="会话ID")
    user_id: str | None = Field(None, description="用户ID")
    reason: str | None = Field(None, max_length=200, description="转人工原因")
    last_question: str | None = Field(None, max_length=500, description="触发转人工的问题")
    priority: Literal["low", "normal", "high"] = Field("normal", description="优先级")


class HandoffData(BaseModel):
    ticket_id: str
    status: str
    priority: str
    source: str
    created_at: str
    queue_hint: str | None = None


class HandoffResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: HandoffData


# ---- 用户反馈 ----


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=6, max_length=50, description="问答请求ID（响应体/响应头中返回）")
    feedback: Literal[-1, 0, 1] = Field(..., description="1=点赞，-1=点踩，0=取消")
    user_id: str | None = Field(None, description="用户ID（用于归属校验）")
    comment: str | None = Field(None, max_length=500, description="可选文字反馈")


class FeedbackData(BaseModel):
    request_id: str
    feedback: int
    recorded_at: str | None = None


class FeedbackResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: FeedbackData
