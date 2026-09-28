"""PostgreSQL 异步连接池（asyncpg）。

两个关键点：
1. 超时必须收敛：asyncpg 默认 create_pool/acquire 超时是 60 秒，PostgreSQL 不可达时
   请求会一直挂住（表现为前端转圈 60 秒），需求文档 N8 要求快速降级，故统一 5 秒。
2. 自愈：连接池在数据库重启/网络中断后会持有失效连接，此后所有查询都失败。
   因此探测失败即丢弃连接池（下次请求自动重建）。
"""
import asyncpg

from app.config import settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None or _pool.is_closing():
        _pool = await asyncpg.create_pool(
            host=settings.pg_host,
            port=settings.pg_port,
            user=settings.pg_user,
            password=settings.pg_password,
            database=settings.pg_database,
            min_size=settings.pg_pool_min,
            max_size=settings.pg_pool_max,
            command_timeout=10,
            timeout=5,                      # 建立连接超时
            max_inactive_connection_lifetime=60,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass
        _pool = None


async def ping() -> bool:
    """健康检查：能执行 SELECT 1 即可用；失败则丢弃连接池以便自愈。"""
    try:
        pool = await get_pool()
        async with pool.acquire(timeout=5) as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        await close_pool()
        return False
