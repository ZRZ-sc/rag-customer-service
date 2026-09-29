"""缓存预热（需求文档流程3：上线前将高频问题预填充）。

设计：
- 通过 **真实 API**（/api/v1/chat）预热，因此写入的正是生产会命中的那套缓存键
  （L1 答案 / L2 改写 / L3 扩展 / L4 重排），不会与实际链路产生偏差。
- 支持两种触发方式：
  1) 手动一键：`python -m scripts.warmup_cache`
  2) 服务启动自动：配置 `CACHE_WARMUP_ON_STARTUP=true`（默认关闭，因为会产生 LLM 费用）
- 预热清单：默认读 `data/faq/high_frequency.json`（含权重排序），缺失时退回 FAQ 全量。
"""
import asyncio
import json
import time
from pathlib import Path

import httpx

from app.config import settings
from app.core.logging import logger
from app.db.redis import get_redis

ROOT = Path(__file__).resolve().parent.parent.parent   # app/services/xxx.py → 项目根目录
HIGH_FREQ_FILE = ROOT / "data" / "faq" / "high_frequency.json"
FAQ_FILE = ROOT / "data" / "faq" / "sample_faq.json"


def load_warmup_questions(limit: int | None = None, file: str | None = None) -> list[str]:
    """读取高频问题清单（按权重降序），文件缺失时退回 FAQ 全量。"""
    path = Path(file) if file else HIGH_FREQ_FILE
    if not path.exists():
        path = FAQ_FILE
    items = json.loads(path.read_text(encoding="utf-8"))
    if items and isinstance(items[0], dict) and "weight" in items[0]:
        items = sorted(items, key=lambda x: x.get("weight", 0), reverse=True)
    questions = [str(it["question"]).strip() for it in items if it.get("question")]
    if limit:
        questions = questions[:limit]
    return questions


async def cache_status(questions: list[str]) -> dict:
    """统计这些问题的 L1 答案缓存命中数（预热前后各测一次）。"""
    from app.services import cache as cache_svc

    r = get_redis()
    hit = 0
    for q in questions:
        if await r.exists(cache_svc.answer_cache_key(q)):
            hit += 1
    return {"warmed": hit, "total": len(questions)}


async def warm_cache(
    questions: list[str],
    base_url: str | None = None,
    concurrency: int = 2,
    clear_first: bool = False,
    timeout: float = 180.0,
) -> dict:
    """并发调用问答接口预热缓存，返回统计结果。"""
    from app.services import cache as cache_svc

    base_url = (base_url or settings.cache_warmup_base_url).rstrip("/")
    if not questions:
        return {"total": 0, "ok": 0, "failed": 0, "elapsed_ms": 0, "before": {}, "after": {}}

    before = await cache_status(questions)
    if clear_first:
        r = get_redis()
        await r.delete(*[cache_svc.answer_cache_key(q) for q in questions])

    headers = {
        "Authorization": f"Bearer {settings.api_token}",
        "Content-Type": "application/json",
    }
    sem = asyncio.Semaphore(max(1, concurrency))
    stats = {"ok": 0, "failed": 0, "details": []}
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def one(q: str, idx: int):
            async with sem:
                try:
                    resp = await client.post(
                        f"{base_url}/api/v1/chat",
                        headers=headers,
                        content=json.dumps(
                            {"session_id": f"warmup-{idx % 5}", "question": q, "user_id": "warmup"}
                        ).encode("utf-8"),
                    )
                    body = resp.json()
                    data = body.get("data") or {}
                    if resp.status_code == 200 and body.get("code") == 0:
                        stats["ok"] += 1
                        stats["details"].append({
                            "question": q, "source": data.get("source"),
                            "confidence": data.get("confidence"), "latency_ms": data.get("latency_ms"),
                        })
                    else:
                        stats["failed"] += 1
                        logger.warning("warmup failed | q={} status={} msg={}", q, resp.status_code, body.get("message"))
                except Exception as exc:
                    stats["failed"] += 1
                    logger.warning("warmup error | q={} err={}", q, exc)

        await asyncio.gather(*[one(q, i) for i, q in enumerate(questions)])

    after = await cache_status(questions)
    result = {
        "total": len(questions),
        "ok": stats["ok"],
        "failed": stats["failed"],
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "before": before,
        "after": after,
        "details": stats["details"],
    }
    logger.info(
        "cache warmup done | total={} ok={} failed={} cached {}->{}/{} elapsed={}ms",
        result["total"], result["ok"], result["failed"],
        before["warmed"], after["warmed"], len(questions), result["elapsed_ms"],
    )
    return result
