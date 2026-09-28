"""Redis 异步客户端（缓存 + 会话存储）。

自愈：Redis 重启或网络中断后，客户端内部连接池可能全部失效。探测失败时丢弃客户端，
下次取用自动重建（缓存/会话本身允许丢失，不影响主流程）。
"""
from redis import asyncio as aioredis

from app.config import settings

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=3,
            health_check_interval=30,
            retry_on_timeout=True,
        )
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None


async def ping() -> bool:
    try:
        return bool(await get_redis().ping())
    except Exception:
        await close_redis()
        return False
