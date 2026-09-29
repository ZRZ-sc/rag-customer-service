"""一键缓存预热：把高频 FAQ 预先跑一遍，填满 Redis 各级缓存。

用法：
  .venv\\Scripts\\python.exe -m scripts.warmup_cache                    # 预热默认高频清单（20 条，权重降序）
  .venv\\Scripts\\python.exe -m scripts.warmup_cache --limit 10          # 只预热前 10 条
  .venv\\Scripts\\python.exe -m scripts.warmup_cache --clear-first       # 先清这些问题的答案缓存（强制重新生成）
  .venv\\Scripts\\python.exe -m scripts.warmup_cache --file data/faq/sample_faq.json --limit 0   # 预热整个 FAQ
  .venv\\Scripts\\python.exe -m scripts.warmup_cache --report docs/warmup_report.json

服务启动时自动预热（默认关闭，会产生 LLM 费用）：
  在 .env.local 中设置 CACHE_WARMUP_ON_STARTUP=true（可配 CACHE_WARMUP_LIMIT / CACHE_WARMUP_BASE_URL）
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx

from app.config import settings
from app.db.redis import close_redis
from app.services.cache_warmup import cache_status, load_warmup_questions, warm_cache


async def main() -> None:
    ap = argparse.ArgumentParser(description="高频问题缓存预热（需求文档流程3）")
    ap.add_argument("--base", default=None, help="服务地址（默认取 CACHE_WARMUP_BASE_URL）")
    ap.add_argument("--file", default=None, help="高频问题清单文件（默认 data/faq/high_frequency.json）")
    ap.add_argument("--limit", type=int, default=0, help="预热条数（0=清单全部）")
    ap.add_argument("--concurrency", type=int, default=2, help="并发度（CPU 机器建议 1-3）")
    ap.add_argument("--clear-first", action="store_true", help="先清空这些问题的答案缓存")
    ap.add_argument("--report", default=None, help="把预热统计写入 JSON 文件")
    args = ap.parse_args()

    base = (args.base or settings.cache_warmup_base_url).rstrip("/")
    questions = load_warmup_questions(limit=args.limit or None, file=args.file)
    if not questions:
        print("[x] 预热清单为空，请检查 data/faq/high_frequency.json")
        return

    # 服务连通性预检（避免把 20 次失败都打出去）
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{base}/ready")
            state = r.json()
        if state.get("status") != "ok":
            print(f"[!] 服务未就绪: {json.dumps(state, ensure_ascii=False)}")
    except Exception as exc:
        print(f"[x] 服务不可达 {base}: {exc}")
        return

    before = await cache_status(questions)
    print(f"预热清单 : {len(questions)} 条（当前已缓存 {before['warmed']} 条）")
    print(f"被测服务 : {base}")
    print(f"并发度   : {args.concurrency}{'（先清缓存）' if args.clear_first else ''}\n")

    result = await warm_cache(
        questions,
        base_url=base,
        concurrency=args.concurrency,
        clear_first=args.clear_first,
    )

    print(f"\n{'=' * 62}")
    print(f"预热完成: 成功 {result['ok']} / 失败 {result['failed']} / 共 {result['total']}")
    print(f"缓存覆盖: {result['before']['warmed']} → {result['after']['warmed']} / {result['total']}")
    print(f"总耗时  : {result['elapsed_ms']} ms")
    slow = sorted(result["details"], key=lambda d: d.get("latency_ms") or 0, reverse=True)[:5]
    if slow:
        print("最慢 Top5（首次生成延迟，之后同类问题将命中缓存）:")
        for d in slow:
            print(f"  {d['latency_ms']:>6}ms  {d['source']:<9} {d['question'][:30]}")
    print("=" * 62)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"统计已写入: {out}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(close_redis())
