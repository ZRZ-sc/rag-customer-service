"""Prometheus 指标定义（与需求文档 3.6 节一致）。"""
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

# ---- 请求量 ----
chat_requests_total = Counter(
    "chat_requests_total", "总请求数", ["source", "status"]
)

# ---- 延迟 ----
chat_duration_seconds = Histogram(
    "chat_duration_seconds",
    "请求耗时",
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
)
query_transform_duration = Histogram(
    "query_transform_duration_seconds", "查询改写耗时", buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
)
rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds", "检索耗时", buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
)
rerank_duration = Histogram(
    "rerank_duration_seconds", "重排耗时", buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)
llm_generation_duration = Histogram(
    "llm_generation_duration_seconds", "LLM生成耗时", buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# ---- 缓存 ----
cache_hits_total = Counter("cache_hits_total", "缓存命中次数")
cache_misses_total = Counter("cache_misses_total", "缓存未命中次数")
cache_write_total = Counter("cache_write_total", "缓存写入次数")

# ---- 并发与用量 ----
concurrent_connections = Gauge("concurrent_connections", "当前并发连接数")
tokens_consumed_total = Counter("tokens_consumed_total", "Token消耗总量", ["model"])

# ---- 路由统计 ----
precision_query_total = Counter("precision_query_total", "精确查询次数")
rag_query_total = Counter("rag_query_total", "RAG检索查询次数")

# ---- 多轮查询改写（指代消解） ----
context_rewrite_total = Counter(
    "context_rewrite_total", "多轮上下文改写次数", ["applied", "method"]
)
context_rewrite_duration = Histogram(
    "context_rewrite_duration_seconds",
    "多轮上下文改写耗时",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# ---- 多级缓存（阶段3） ----
cache_operation_total = Counter(
    "cache_operation_total", "缓存操作次数", ["name", "result"]
)

# ---- RAG 快速路径 ----
rag_fast_path_total = Counter(
    "rag_fast_path_total", "RAG 快速路径命中次数", ["result"]
)

# ---- LLM 生成失败降级（需求文档流程6：超时/不可用 → 固定话术 + 转人工） ----
llm_failure_total = Counter(
    "llm_failure_total", "LLM 生成失败次数（已降级为兜底话术）", ["stage"]
)

# ---- 限流（429） ----
rate_limited_total = Counter(
    "rate_limited_total", "触发限流的请求数", ["scope"]
)

# ---- 转人工与反馈（F6 / F7） ----
handoff_tickets_total = Counter(
    "handoff_tickets_total", "转人工工单数", ["source", "priority"]
)
feedback_total = Counter(
    "feedback_total", "用户反馈数", ["value"]
)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
