"""请求/响应模型（与需求文档 3.5 节 API 一致）。"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话ID，用于多轮对话")
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    user_id: str | None = Field(None, description="用户ID，用于个性化与权限校验")


class SourceItem(BaseModel):
    """命中的知识条目（RAG 检索证据，供前端展示"答案依据"）。"""

    faq_id: str | None = None
    question: str = ""
    category: str = ""
    score: float | None = None

    @field_validator("faq_id", mode="before")
    @classmethod
    def _faq_id_to_str(cls, v):
        """Milvus/PG 里的 faq_id 可能是整数，统一转字符串再返回给前端。"""
        return None if v is None else str(v)


class ChatData(BaseModel):
    answer: str
    source: Literal["database", "rag", "fallback"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    session_id: str
    tokens_used: int = 0
    latency_ms: int = 0
    # 以下为可观测性与闭环扩展字段
    request_id: str | None = Field(None, description="本次问答的请求ID，用于提交反馈（/api/v1/feedback）")
    intent: str | None = Field(None, description="意图识别结果：precise_query/knowledge/other")
    rewritten_question: str | None = Field(
        None, description="多轮上下文改写后的问题（仅当发生改写时返回）"
    )
    handoff_ticket_id: str | None = Field(
        None, description="低置信度自动转人工时返回的工单号（F6）"
    )
    sources: list[SourceItem] | None = Field(
        None, description="知识类回答命中的 FAQ 条目（重排后的 Top-N，含分数）"
    )


class ChatResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: ChatData


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, str]
