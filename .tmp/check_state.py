"""快速检查：对话日志、会话缓存、向量库状态。"""
import asyncio

import asyncpg

from app.config import settings
from app.db import milvus
from app.db.redis import get_redis


async def main():
    conn = await asyncpg.connect(dsn=settings.pg_dsn)
    rows = await conn.fetch(
        "SELECT request_id, session_id, user_id, intent, source, confidence, latency_ms, tokens_used "
        "FROM dialogue_logs ORDER BY id DESC LIMIT 6"
    )
    print("== dialogue_logs (最近6条) ==")
    for r in rows:
        print(dict(r))
    n_orders = await conn.fetchval("SELECT count(*) FROM orders")
    n_faq = await conn.fetchval("SELECT count(*) FROM faq_documents WHERE status='published'")
    print(f"orders={n_orders} faq_docs={n_faq}")
    await conn.close()

    r = get_redis()
    print("== redis keys ==")
    keys = []
    async for k in r.scan_iter(count=100):
        keys.append(k)
    print(keys)

    client = milvus.get_client()
    stats = client.get_collection_stats(settings.milvus_collection)
    print("== milvus stats ==", stats)


if __name__ == "__main__":
    asyncio.run(main())
