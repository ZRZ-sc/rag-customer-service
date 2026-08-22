"""Redis 缓存：知识类答案缓存 + 精确查询短缓存。"""
import hashlib
import json

from app.config import settings
from app.core.metrics import cache_hits_total, cache_misses_total, cache_write_total
from app.db.redis import get_redis


def answer_cache_key(question: str) -> str:
    norm = " ".join(question.strip().lower().split())
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
    return f"rag:answer:{digest}"


def precise_cache_key(order_id: str, user_id: str) -> str:
    digest = hashlib.sha1(f"{user_id}:{order_id}".encode("utf-8")).hexdigest()
    return f"rag:precise:{digest}"


async def get_cached_answer(key: str) -> dict | None:
    if not settings.cache_enabled:
        return None
    try:
        r = get_redis()
        raw = await r.get(key)
        if raw:
            cache_hits_total.inc()
            return json.loads(raw)
        cache_misses_total.inc()
        return None
    except Exception:
        # Redis 故障时跳过缓存（降级），不影响主流程
        return None


async def set_cached_answer(key: str, payload: dict, ttl: int | None = None) -> None:
    if not settings.cache_enabled:
        return
    try:
        r = get_redis()
        await r.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl or settings.cache_ttl_seconds)
        cache_write_total.inc()
    except Exception:
        pass


async def invalidate_knowledge_cache() -> int:
    """知识库更新后清除所有知识类答案缓存（dev 阶段用全量清除）。"""
    r = get_redis()
    try:
        keys = []
        async for k in r.scan_iter(match="rag:answer:*", count=500):
            keys.append(k)
        if keys:
            await r.delete(*keys)
        return len(keys)
    except Exception:
        return 0
