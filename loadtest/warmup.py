"""压测前置：缓存预热 / 缓存清空。

用法：
  .venv\\Scripts\\python.exe -m loadtest.warmup --profile cache            # 预热（默认目标 8001）
  .venv\\Scripts\\python.exe -m loadtest.warmup --profile rag --clear       # 清空（保证冷启动）

说明：
- cache 画像需要先预热 L1 答案缓存与 L2-L4 各级缓存，否则测到的是冷启动而非稳态吞吐。
- rag 画像需要清空缓存，保证每个请求都走完整链路（问题本身还带唯一后缀，必然 miss）。
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.config import settings
from app.db.redis import close_redis, get_redis
from app.services import cache as cache_svc

with open(ROOT / "data" / "faq" / "sample_faq.json", encoding="utf-8") as f:
    FAQ = [item["question"] for item in json.load(f)]

ORDERS = ["ORD20240100001", "ORD20240100002", "ORD20240100003", "ORD20240100004"]


async def warm(base: str, concurrency: int = 4) -> None:
    """把 FAQ 问题各问一遍，填满 L1-L4 缓存（预热后测吞吐才有意义）。"""
    headers = {"Authorization": f"Bearer {settings.api_token}", "Content-Type": "application/json"}
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=180) as client:
        async def one(q: str, i: int):
            nonlocal done
            async with sem:
                try:
                    await client.post(
                        f"{base}/api/v1/chat",
                        headers=headers,
                        content=json.dumps(
                            {"session_id": f"warm-{i % 10}", "question": q, "user_id": "user_001"}
                        ).encode(),
                    )
                    done += 1
                except Exception as exc:
                    print(f"  [warn] 预热失败: {q[:20]}... {exc}")

        await asyncio.gather(*[one(q, i) for i, q in enumerate(FAQ)])
        # 精确查询短缓存（30s TTL）——按需预热，压测期间用不到可跳过
        await asyncio.gather(*[one(f"我的订单 {o} 到哪里了？", i) for i, o in enumerate(ORDERS)])

    print(f"[warmup] 预热完成：{done} 个问题 / {time.perf_counter()-t0:.1f}s")


async def clear_all() -> None:
    patterns = {
        "answer": "rag:answer:",
        "precise": "rag:precise:",
        "rewrite": "query_rewrite:",
        "transform": "query_transform:",
        "rerank": "rerank:",
    }
    for name, prefix in patterns.items():
        n = await cache_svc.invalidate_prefix(prefix)
        print(f"[clear] {name}: 删除 {n} 个 key")


async def seed_cache_directly() -> None:
    """直接向 Redis 写入 L1 答案缓存（键格式与生产一致），无需触发真实 RAG。

    用途：内存受限（无法再加载一份本地模型）时，仍能对"缓存命中"这一主流量路径压测。
    写入的 payload 结构与 app 的答案缓存完全一致：{"answer", "confidence", "tokens_used"}。
    """
    r = get_redis()
    ttl = settings.cache_ttl_seconds
    for q in FAQ:
        payload = {
            "answer": f"【压测预置答案】{q} —— 该问题命中 L1 答案缓存。",
            "confidence": 0.99,
            "tokens_used": 0,
        }
        await r.set(cache_svc.answer_cache_key(q), json.dumps(payload, ensure_ascii=False), ex=ttl)
    print(f"[seed] 已直接预置 {len(FAQ)} 条 L1 答案缓存（TTL={ttl}s），无需加载本地模型")


async def main() -> None:
    parser = argparse.ArgumentParser(description="load test warm-up / cache clear / cache seed")
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--profile", default="cache", choices=["cache", "precise", "rag"])
    parser.add_argument("--clear", action="store_true", help="清空缓存")
    parser.add_argument("--seed", action="store_true", help="直接向 Redis 预置 L1 答案缓存（不调用接口）")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    try:
        if args.seed:
            await seed_cache_directly()
            return
        if args.clear or args.profile == "rag":
            await clear_all()
            if args.profile == "rag":
                print("[warmup] rag 画像：缓存已清空，全部请求将走完整链路")
            return
        await warm(args.base, args.concurrency)
    finally:
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
