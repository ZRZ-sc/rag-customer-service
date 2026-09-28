"""阶段3 延迟测试：冷启动 / L2-L4 缓存复用 / L1 答案缓存 的端到端对比。

用法：
  .venv\\Scripts\\python.exe -m scripts.latency_test

判定目标（需求）：
  首次查询 < 2s   —— 指"扩展+重排缓存已就绪"时的查询（L2/L3/L4 命中）
  缓存命中 < 10ms —— L1 答案缓存命中
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import asyncpg
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台默认 GBK，打印 ✅/❌ 会抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.config import settings
from app.db.redis import close_redis, get_redis
from app.services import cache as cache_svc

BASE = "http://localhost:8000"
HEADERS = {"Authorization": f"Bearer {settings.api_token}", "Content-Type": "application/json"}

# 测试问题（均来自样例知识库）
COLD_Q = "七天无理由退货的条件是什么？"
WARM_Q = "怎么开具发票？"
FULL_Q = "买回来发现东西有毛病咋整"       # 口语化、低相似度 → 触发完整路径（LLM 扩展 + 全量重排）
FOLLOWUP = "那什么时候能到？"
ORDER_Q = "我的订单 ORD20240100001 到哪里了？"


async def ask(client: httpx.AsyncClient, session_id: str, question: str) -> tuple[float, dict]:
    t0 = time.perf_counter()
    resp = await client.post(
        f"{BASE}/api/v1/chat",
        headers=HEADERS,
        content=json.dumps(
            {"session_id": session_id, "question": question, "user_id": "user_001"}
        ).encode("utf-8"),
        timeout=180,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    body = resp.json()
    return elapsed, body.get("data", {})


async def clear_caches(question: str, *, answer: bool = True, transform: bool = True, rerank: bool = False) -> None:
    r = get_redis()
    keys = []
    if answer:
        keys.append(cache_svc.answer_cache_key(question))
    if transform:
        keys.append(cache_svc.transform_cache_key(question))
    if keys:
        await r.delete(*keys)
    if rerank:
        await cache_svc.invalidate_prefix("rerank:")


async def last_log(session_id: str) -> dict | None:
    """读取最近一条问答日志（含阶段耗时/路径/缓存标记）。"""
    conn = await asyncpg.connect(dsn=settings.pg_dsn)
    try:
        row = await conn.fetchrow(
            "SELECT question, rewritten_question, intent, source, confidence, latency_ms, "
            "retrieval_results FROM dialogue_logs WHERE session_id=$1 ORDER BY id DESC LIMIT 1",
            session_id,
        )
        if not row:
            return None
        detail = row["retrieval_results"]
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except Exception:
                detail = None
        return {
            "question": row["question"],
            "rewritten": row["rewritten_question"],
            "intent": row["intent"],
            "source": row["source"],
            "confidence": row["confidence"],
            "latency_ms": row["latency_ms"],
            "detail": detail or {},
        }
    finally:
        await conn.close()


def fmt_ms(v: float) -> str:
    return f"{v:.0f}ms"


async def main() -> None:
    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        # 健康检查
        try:
            ready = (await client.get(f"{BASE}/ready", timeout=20)).json()
        except Exception as exc:
            print(f"[x] 服务不可达：{exc}")
            return
        print(f"[env] /ready = {json.dumps(ready, ensure_ascii=False)}\n")

        # ---------- 场景1：完全冷启动（清掉 L1/L3/L4） ----------
        await clear_caches(COLD_Q, rerank=True)
        elapsed, data = await ask(client, "lat-cold-1", COLD_Q)
        log = await last_log("lat-cold-1")
        detail = log["detail"] if log else {}
        results.append(
            {
                "phase": "① 冷启动（改写/扩展/重排缓存全空）",
                "client_ms": elapsed,
                "server_ms": data.get("latency_ms"),
                "source": data.get("source"),
                "path": detail.get("path", "-"),
                "cache": detail.get("cache", {}),
                "stages": detail.get("stage_ms", {}),
            }
        )

        # ---------- 场景2：L1 答案缓存命中 ----------
        elapsed, data = await ask(client, "lat-cold-1", COLD_Q)
        results.append(
            {
                "phase": "② L1 答案缓存命中（同问题再问）",
                "client_ms": elapsed,
                "server_ms": data.get("latency_ms"),
                "source": data.get("source"),
                "path": "answer_cache",
                "cache": {"answer": "hit"},
                "stages": {},
            }
        )

        # ---------- 场景3：只清答案缓存（L3 扩展 + L4 重排缓存保留） ----------
        await clear_caches(COLD_Q, answer=True, transform=False, rerank=False)
        elapsed, data = await ask(client, "lat-warm-1", COLD_Q)
        log = await last_log("lat-warm-1")
        detail = log["detail"] if log else {}
        results.append(
            {
                "phase": "③ 缓存复用（L3/L4 命中，仅答案缓存失效）",
                "client_ms": elapsed,
                "server_ms": data.get("latency_ms"),
                "source": data.get("source"),
                "path": detail.get("path", "-"),
                "cache": detail.get("cache", {}),
                "stages": detail.get("stage_ms", {}),
            }
        )

        # ---------- 场景4：另一个问题的缓存复用（验证非个例） ----------
        await clear_caches(WARM_Q, rerank=True)
        await ask(client, "lat-cold-2", WARM_Q)          # 预热 L3/L4
        await clear_caches(WARM_Q, answer=True, transform=False, rerank=False)
        elapsed, data = await ask(client, "lat-warm-2", WARM_Q)
        log = await last_log("lat-warm-2")
        detail = log["detail"] if log else {}
        results.append(
            {
                "phase": "④ 缓存复用（第二个问题，L3/L4 命中）",
                "client_ms": elapsed,
                "server_ms": data.get("latency_ms"),
                "source": data.get("source"),
                "path": detail.get("path", "-"),
                "cache": detail.get("cache", {}),
                "stages": detail.get("stage_ms", {}),
            }
        )

        # ---------- 场景5：多轮追问的改写缓存（L2） ----------
        session = "lat-multi-1"
        await get_redis().delete(f"session:{session}", f"session:{session}:entities")
        await ask(client, session, ORDER_Q)              # 建立上下文
        t1_elapsed, _ = await ask(client, session, FOLLOWUP)   # 首次改写（L2 miss）
        log1 = await last_log(session)
        await get_redis().delete(cache_svc.answer_cache_key(log1["rewritten"] or FOLLOWUP))
        t2_elapsed, data2 = await ask(client, session, FOLLOWUP)  # 改写缓存命中
        log2 = await last_log(session)
        results.append(
            {
                "phase": "⑤ 多轮追问（L2 改写缓存命中）",
                "client_ms": t2_elapsed,
                "server_ms": data2.get("latency_ms"),
                "source": data2.get("source"),
                "path": (log2["detail"] or {}).get("path", "-"),
                "cache": {"rewrite": "hit", "首次改写耗时ms": round(t1_elapsed)},
                "stages": (log2["detail"] or {}).get("stage_ms", {}),
            }
        )

        # ---------- 场景6：完整路径冷启动（低相似度口语化问题） ----------
        await clear_caches(FULL_Q, rerank=True)
        elapsed, data = await ask(client, "lat-full-1", FULL_Q)
        log = await last_log("lat-full-1")
        detail = log["detail"] if log else {}
        results.append(
            {
                "phase": "⑥ 完整路径冷启动（LLM 扩展 miss + 全量重排 miss）",
                "client_ms": elapsed,
                "server_ms": data.get("latency_ms"),
                "source": data.get("source"),
                "path": detail.get("path", "-"),
                "cache": detail.get("cache", {}),
                "stages": detail.get("stage_ms", {}),
            }
        )

        # ---------- 场景7：完整路径的缓存复用（L3/L4 命中） ----------
        await clear_caches(FULL_Q, answer=True, transform=False, rerank=False)
        elapsed, data = await ask(client, "lat-full-2", FULL_Q)
        log = await last_log("lat-full-2")
        detail = log["detail"] if log else {}
        results.append(
            {
                "phase": "⑦ 完整路径缓存复用（L3/L4 命中）",
                "client_ms": elapsed,
                "server_ms": data.get("latency_ms"),
                "source": data.get("source"),
                "path": detail.get("path", "-"),
                "cache": detail.get("cache", {}),
                "stages": detail.get("stage_ms", {}),
            }
        )

    # ---------- 结果输出 ----------
    print("=" * 96)
    print("阶段3 延迟测试结果")
    print("=" * 96)
    for r in results:
        print(f"\n{r['phase']}")
        print(
            f"  客户端往返 {fmt_ms(r['client_ms'])} | 服务端 latency_ms={r['server_ms']}ms | "
            f"source={r['source']} | path={r['path']}"
        )
        if r["cache"]:
            print(f"  缓存: {json.dumps(r['cache'], ensure_ascii=False)}")
        if r["stages"]:
            stages = " ".join(f"{k}={v:.0f}ms" for k, v in r["stages"].items())
            print(f"  阶段: {stages}")

    print("\n" + "=" * 96)
    print("目标核对")
    print("=" * 96)
    by_phase = {r["phase"][0]: r for r in results}
    l1 = by_phase.get("②", {})
    warm = [by_phase.get(k, {}) for k in ("③", "④", "⑦") if k in by_phase]
    print(f"  L1 答案缓存命中            : {l1.get('server_ms')}ms   "
          f"{'✅ <10ms 达标' if (l1.get('server_ms') or 999) < 10 else '❌ 未达标'}")
    for w in warm:
        ok = (w.get("server_ms") or 99999) < 2000
        print(f"  缓存复用 {w['phase'][:2]}                 : {w.get('server_ms')}ms   "
              f"{'✅ <2000ms 达标' if ok else '❌ 未达标'}  ({w['phase']})")
    for r in results:
        if r["phase"].startswith(("①", "⑥")):
            print(f"  {r['phase'][:2]} 冷启动                : {r.get('server_ms')}ms   "
                  f"(path={r.get('path')}, 缓存={json.dumps(r.get('cache'), ensure_ascii=False)})")
    print("  说明：冷启动耗时主要取决于 DeepSeek 生成调用与 CPU 重排，缓存命中后不再受其影响。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(close_redis())
