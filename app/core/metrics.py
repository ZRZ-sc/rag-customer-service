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


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
