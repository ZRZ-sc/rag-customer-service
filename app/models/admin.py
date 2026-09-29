"""管理后台（F7）请求/响应模型：只读查询问答日志、统计数据与转人工工单。"""
from pydantic import BaseModel, Field


class DialogueLogItem(BaseModel):
    id: int
    request_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    question: str | None = None
    rewritten_question: str | None = None
    intent: str | None = None
    source: str | None = None
    answer: str | None = None
    confidence: float | None = None
    latency_ms: int | None = None
    tokens_used: int | None = None
    feedback: int | None = 0
    feedback_comment: str | None = None
    created_at: str | None = None


class PaginatedLogs(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    items: list[DialogueLogItem]


class LogsResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: PaginatedLogs


class LatencyStats(BaseModel):
    avg_ms: float = 0
    p50_ms: float = 0
    p95_ms: float = 0
    p99_ms: float = 0
    max_ms: float = 0


class CacheStats(BaseModel):
    """来自 Prometheus 的运行时指标（不可用时为 None）。"""
    hit_rate: float | None = Field(None, description="答案缓存命中率 0~1")
    hits: float | None = None
    misses: float | None = None
    fast_path_ratio: float | None = Field(None, description="RAG 快速路径占比 0~1")


class HandoffStats(BaseModel):
    total: int = 0
    pending: int = 0
    auto: int = 0
    manual: int = 0


class FeedbackStats(BaseModel):
    up: int = 0
    down: int = 0
    positive_rate: float | None = None


class AdminStats(BaseModel):
    window_minutes: int
    total_requests: int
    qps: float
    by_source: dict[str, int] = {}
    by_intent: dict[str, int] = {}
    by_status: dict[str, int] = {}
    latency: LatencyStats
    tokens_total: int = 0
    handoffs: HandoffStats
    feedback: FeedbackStats
    cache: CacheStats
    generated_at: str


class StatsResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: AdminStats


class HandoffItem(BaseModel):
    ticket_id: str
    session_id: str | None = None
    user_id: str | None = None
    reason: str | None = None
    last_question: str | None = None
    source: str | None = None
    priority: str | None = None
    status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class PaginatedHandoffs(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    items: list[HandoffItem]


class HandoffsResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: PaginatedHandoffs
