"""Redis 缓存层（多级）：

| 层级 | Key 格式 | TTL | 作用 |
|------|----------|-----|------|
| L1 答案缓存 | `rag:answer:{sha1(question)}` | 1h | 命中直接返回（<10ms） |
| L1' 精确查询短缓存 | `rag:precise:{sha1(user:order)}` | 30s | 订单/物流状态，保证实时性 |
| L2 上下文改写 | `query_rewrite:{sha1(question)}` | 24h | 省一次指代消解 LLM 调用 |
| L3 查询扩展 | `query_transform:{sha1(question)}` | 24h | 省一次 Multi-Query+HyDE LLM 调用 |
| L4 重排结果 | `rerank:{sha1(query)}:{sha1(doc_ids)}` | 24h | 省 CPU 交叉编码器重排（知识库变更即失效） |
"""
import hashlib
import json
from typing import Any

from app.config import settings
from app.core.metrics import cache_hits_total, cache_misses_total, cache_operation_total, cache_write_total
from app.db.redis import get_redis


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


# ---- Key 构造 ----


def answer_cache_key(question: str) -> str:
    return f"rag:answer:{_sha1(_normalize(question))}"


def precise_cache_key(order_id: str, user_id: str) -> str:
    return f"rag:precise:{_sha1(f'{user_id}:{order_id}')}"


def rewrite_cache_key(question: str, context_fingerprint: str = "") -> str:
    """上下文改写结果缓存键（24h）。

    注意：改写结果依赖会话上下文（当前在讨论哪张订单），因此 hash 中必须包含
    上下文指纹（会话实体 + 上一轮用户问题），否则不同会话的同一句话会互相串味
    （A 会话的订单号被 B 会话复用）。
    """
    return f"query_rewrite:{_sha1(f'{_normalize(question)}|{context_fingerprint}')}"


def transform_cache_key(question: str) -> str:
    """Multi-Query / HyDE 扩展结果缓存键（24h）。"""
    return f"query_transform:{_sha1(_normalize(question))}"


def rerank_cache_key(question: str, doc_ids: list[int]) -> str:
    """重排结果缓存键：query + 候选文档集合（集合不变则得分可复用）。"""
    ids = ",".join(str(i) for i in sorted(doc_ids))
    return f"rerank:{_sha1(_normalize(question))}:{_sha1(ids)}"


# ---- 通用读写（可观测：cache_operation_total{name,result}） ----


async def get_json(key: str, name: str = "json") -> Any | None:
    if not settings.cache_enabled:
        cache_operation_total.labels(name=name, result="disabled").inc()
        return None
    try:
        raw = await get_redis().get(key)
        if raw:
            cache_operation_total.labels(name=name, result="hit").inc()
            return json.loads(raw)
        cache_operation_total.labels(name=name, result="miss").inc()
        return None
    except Exception:
        # Redis 故障时跳过缓存（降级）
        cache_operation_total.labels(name=name, result="error").inc()
        return None


async def set_json(key: str, payload: Any, ttl: int, name: str = "json") -> None:
    if not settings.cache_enabled:
        return
    try:
        await get_redis().set(key, json.dumps(payload, ensure_ascii=False), ex=ttl)
        cache_operation_total.labels(name=name, result="write").inc()
    except Exception:
        pass


# ---- 知识类答案缓存（带答案缓存专属指标，供 Grafana 命中率面板使用） ----


async def get_cached_answer(key: str) -> dict | None:
    if not settings.cache_enabled:
        return None
    try:
        raw = await get_redis().get(key)
        if raw:
            cache_hits_total.inc()
            return json.loads(raw)
        cache_misses_total.inc()
        return None
    except Exception:
        return None


async def set_cached_answer(key: str, payload: dict, ttl: int | None = None) -> None:
    if not settings.cache_enabled:
        return
    try:
        await get_redis().set(
            key, json.dumps(payload, ensure_ascii=False), ex=ttl or settings.cache_ttl_seconds
        )
        cache_write_total.inc()
    except Exception:
        pass


# ---- 失效 ----


async def invalidate_prefix(prefix: str) -> int:
    """按前缀清除缓存，返回删除数量。"""
    try:
        r = get_redis()
        keys = []
        async for k in r.scan_iter(match=f"{prefix}*", count=500):
            keys.append(k)
        if keys:
            await r.delete(*keys)
        return len(keys)
    except Exception:
        return 0


async def invalidate_knowledge_cache() -> dict[str, int]:
    """知识库更新后清除受影响缓存：答案 + 查询扩展 + 重排结果。

    重排缓存与文档集合绑定，文档内容变更后旧得分不再可信，需要一并清除。
    """
    return {
        "answer": await invalidate_prefix("rag:answer:"),
        "transform": await invalidate_prefix("query_transform:"),
        "rerank": await invalidate_prefix("rerank:"),
    }
