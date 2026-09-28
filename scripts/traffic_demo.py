"""流量演示/压测脚本：混合发送各类请求，用于点亮 Grafana 仪表盘。

用法：
  .venv\\Scripts\\python.exe -m scripts.traffic_demo                 # 默认 20 轮
  .venv\\Scripts\\python.exe -m scripts.traffic_demo --rounds 50 --concurrency 10

请求类型（会分别打到不同的指标与代码路径）：
  1) 精确查询（订单号）      → precision_query_total / source=database
  2) 物流单号查询            → precision_query_total
  3) 知识类新问题            → rag_query_total / 完整路径或快速路径
  4) 知识类重复问题（缓存）  → cache_hits_total / L1 命中
  5) 多轮追问（指代消解）    → context_rewrite_total
  6) 鉴权失败（401）/ 越权（403） → chat_requests_total{status="error"}
"""
import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.config import settings

BASE = "http://localhost:8000"

ORDERS = ["ORD20240100001", "ORD20240100002", "ORD20240100003", "ORD20240100004"]
TRACKINGS = ["SF123456789012", "YT987654321098"]
KNOWLEDGE = [
    "退货运费谁出？",
    "七天无理由退货的条件是什么？",
    "发票怎么开？",
    "保修期多久？",
    "会员有什么权益？",
    "支持货到付款吗？",
]
FOLLOWUPS = ["那什么时候能到？", "那退货运费谁出？", "它现在到哪了"]


async def one_request(client: httpx.AsyncClient, kind: str, idx: int) -> tuple[str, int, float]:
    session = f"demo-{idx % 5}"
    token = settings.api_token
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if kind == "precise":
        question = f"我的订单 {random.choice(ORDERS)} 到哪里了？"
    elif kind == "tracking":
        question = f"查询物流单号 {random.choice(TRACKINGS)} 的进度"
    elif kind == "knowledge":
        question = random.choice(KNOWLEDGE)
    elif kind == "cache":
        question = random.choice(KNOWLEDGE)
    elif kind == "followup":
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        await client.post(
            f"{BASE}/api/v1/chat", headers=headers,
            content=json.dumps({"session_id": f"multi-{idx % 3}", "question": f"我的订单 {random.choice(ORDERS)} 到哪里了？", "user_id": "user_001"}).encode(),
            timeout=60,
        )
        question = random.choice(FOLLOWUPS)
        session = f"multi-{idx % 3}"
    elif kind == "unauthorized":
        headers = {"Authorization": "wrong-token"}
        question = "退货政策是什么？"
    elif kind == "forbidden":
        question = f"我的订单 {ORDERS[0]} 到哪里了？"
        session = f"demo-{idx}"
        headers = headers
    else:
        question = "你好"

    if kind == "forbidden":
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        body = {"session_id": session, "question": question, "user_id": "user_002"}
    else:
        body = {
            "session_id": session,
            "question": question,
            "user_id": "user_001",
        }

    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE}/api/v1/chat", headers=headers, content=json.dumps(body).encode(), timeout=120
        )
        code = resp.status_code
    except Exception:
        code = 0
    return kind, code, (time.perf_counter() - t0) * 1000


async def main() -> None:
    parser = argparse.ArgumentParser(description="traffic demo for Grafana dashboards")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    # 权重：知识类重复问题占比高（真实客服场景），便于观察缓存命中率
    plan = (
        ["precise"] * 3
        + ["tracking"] * 1
        + ["knowledge"] * 3
        + ["cache"] * 5
        + ["followup"] * 2
        + ["unauthorized"] * 1
        + ["forbidden"] * 1
    )

    counts: Counter[str] = Counter()
    latencies: dict[str, list[float]] = {}
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:
        # 预热一次知识类请求（避免首答把统计拉偏）
        await one_request(client, "knowledge", 0)

        async def worker(i: int):
            kind = random.choice(plan)
            async with sem:
                k, code, ms = await one_request(client, kind, i)
            counts[f"{k}:{code}"] += 1
            latencies.setdefault(k, []).append(ms)

        t0 = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(args.rounds * len(plan) // 6 or args.rounds)])
        total_s = time.perf_counter() - t0

    print(f"== 流量演示完成：{sum(counts.values())} 次请求 / {total_s:.1f}s（并发 {args.concurrency}）==")
    print(f"{'类型:状态':<24}{'次数':>6}{'平均ms':>10}{'P95ms':>10}")
    print("-" * 52)
    by_kind: dict[str, list[float]] = {}
    for key, n in sorted(counts.items()):
        kind = key.split(":")[0]
        by_kind.setdefault(kind, [])
    for kind, samples in latencies.items():
        n = len(samples)
        p95 = statistics.quantiles(samples, n=20)[-1] if n > 1 else samples[0]
        print(f"{kind:<24}{n:>6}{statistics.mean(samples):>10.0f}{p95:>10.0f}")
    print("\n提示：等待 15-30 秒让 Prometheus 抓取两轮，然后打开 Grafana 仪表盘查看曲线。")


if __name__ == "__main__":
    asyncio.run(main())
