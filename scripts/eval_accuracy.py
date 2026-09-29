"""N4 评测：批量跑评测集，量化意图识别准确率、答案正确率与延迟。

用法：
  .venv\\Scripts\\python.exe -m scripts.eval_accuracy                       # 全量评测（默认清答案缓存，保证真实回答）
  .venv\\Scripts\\python.exe -m scripts.eval_accuracy --category knowledge  # 只跑某一类
  .venv\\Scripts\\python.exe -m scripts.eval_accuracy --clear-all           # 连扩展/重排缓存一并清空（最冷）
  .venv\\Scripts\\python.exe -m scripts.eval_accuracy --base http://127.0.0.1:8001

输出：控制台摘要 + docs/eval_report.md（可直接作为验收材料）

评测口径：
  意图识别准确率 = 预测 intent == 期望 intent（仅统计期望中标注 intent 且 HTTP 200 的轮次）
  答案正确率     = source 相符 且 关键词断言通过 且 反向断言通过 且 改写断言通过（仅统计期望 status=200 的轮次）
  HTTP 状态符合率 = 实际状态码 == 期望状态码（涵盖 401/403/404/422 等异常用例）
  延迟           = 服务端 latency_ms（200 响应），另附客户端往返耗时
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
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

AUTH_HEADERS = {
    "valid": lambda: {"Authorization": f"Bearer {settings.api_token}"},
    "invalid": lambda: {"Authorization": "Bearer wrong-token-xxxx"},
    "none": lambda: {},
    "malformed": lambda: {"Authorization": settings.api_token},   # 缺少 Bearer 前缀
}


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


async def clear_eval_cache(questions: list[str], clear_all: bool) -> dict:
    """清空评测相关缓存，避免历史/压测预置答案影响正确性判定。"""
    r = get_redis()
    stats = {}
    if clear_all:
        for prefix in ("rag:answer:", "query_transform:", "query_rewrite:", "rerank:"):
            keys = [k async for k in r.scan_iter(match=f"{prefix}*", count=500)]
            if keys:
                await r.delete(*keys)
            stats[prefix] = len(keys)
    else:
        keys = []
        from app.services import cache as cache_svc

        for q in questions:
            keys.append(cache_svc.answer_cache_key(q))
        keys = list(dict.fromkeys(keys))
        if keys:
            stats["rag:answer:"] = await r.delete(*keys)
    return stats


async def run_case(client: httpx.AsyncClient, case: dict, clear_sessions: bool) -> list[dict]:
    """执行一个用例的所有轮次，返回每轮的结果记录。"""
    session = case.get("session") or f"eval-{case['id']}"
    if clear_sessions:
        try:
            await get_redis().delete(f"session:{session}", f"session:{session}:entities")
        except Exception:
            pass

    auth_mode = case.get("auth", "valid")
    headers = dict(AUTH_HEADERS.get(auth_mode, AUTH_HEADERS["valid"])())
    headers["Content-Type"] = "application/json"

    records = []
    for i, turn in enumerate(case["turns"]):
        payload = {"question": turn["question"], "user_id": case.get("user_id")}
        if case.get("broken") != "no_session":
            payload["session_id"] = session

        t0 = time.perf_counter()
        try:
            resp = await client.post(
                "/api/v1/chat",
                headers=headers,
                content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )
            elapsed = (time.perf_counter() - t0) * 1000
            try:
                body = resp.json()
            except Exception:
                body = {}
        except Exception as exc:
            records.append({
                "case": case["id"], "category": case["category"], "turn": i + 1,
                "question": turn["question"][:40], "status": 0, "client_ms": 0,
                "server_ms": None, "error": str(exc), "expect": turn["expect"],
                "checks": {"status": False}, "answer_correct": False,
            })
            continue

        data = body.get("data") or {}
        expect = turn["expect"]
        answer_text = str(data.get("answer") or "")
        rewritten = str(data.get("rewritten_question") or "")

        checks = {"status": resp.status_code == expect.get("status", 200)}
        if resp.status_code == 200:
            if "intent" in expect:
                checks["intent"] = data.get("intent") == expect["intent"]
            if "source" in expect:
                checks["source"] = data.get("source") == expect["source"]
            if expect.get("must_contain"):
                checks["must_contain"] = all(k in answer_text for k in expect["must_contain"])
            if expect.get("must_contain_any"):
                checks["must_contain_any"] = any(k in answer_text for k in expect["must_contain_any"])
            if expect.get("must_not_contain"):
                checks["must_not_contain"] = not any(k in answer_text for k in expect["must_not_contain"])
            if expect.get("rewritten_contains"):
                checks["rewritten_contains"] = expect["rewritten_contains"] in rewritten
            if expect.get("rewritten_not_contains"):
                checks["rewritten_not_contains"] = expect["rewritten_not_contains"] not in rewritten
            if expect.get("handoff_ticket"):
                checks["handoff_ticket"] = bool(data.get("handoff_ticket_id"))
        else:
            if "code" in expect:
                checks["code"] = body.get("code") == expect["code"]

        answer_keys = {"source", "must_contain", "must_contain_any", "must_not_contain",
                       "rewritten_contains", "rewritten_not_contains", "handoff_ticket"}
        answer_correct = checks["status"] and all(
            v for k, v in checks.items() if k in answer_keys or k == "status"
        )
        records.append({
            "case": case["id"], "category": case["category"], "turn": i + 1,
            "question": turn["question"][:48], "status": resp.status_code,
            "client_ms": round(elapsed), "server_ms": data.get("latency_ms"),
            "intent": data.get("intent"), "source": data.get("source"),
            "confidence": data.get("confidence"), "rewritten": rewritten[:60],
            "answer": answer_text[:80], "expect": expect,
            "checks": checks, "answer_correct": bool(answer_correct),
        })
    return records


def summarize(records: list[dict], cases_total: int, elapsed_s: float, base: str, clear_stats: dict) -> tuple[str, dict]:
    turns_total = len(records)
    status_ok = sum(1 for r in records if r["checks"].get("status"))
    intent_turns = [r for r in records if "intent" in r["expect"]]
    intent_ok = sum(1 for r in intent_turns if r["checks"].get("intent"))
    answer_turns = [r for r in records if r["expect"].get("status", 200) == 200]
    answer_ok = sum(1 for r in answer_turns if r["answer_correct"])
    server_lat = [r["server_ms"] for r in records if r.get("server_ms")]
    client_lat = [r["client_ms"] for r in records if r.get("client_ms")]

    intent_acc = intent_ok / len(intent_turns) if intent_turns else 0.0
    answer_acc = answer_ok / len(answer_turns) if answer_turns else 0.0
    status_acc = status_ok / turns_total if turns_total else 0.0

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    def cat_row(cat: str, rs: list[dict]) -> str:
        it = [r for r in rs if "intent" in r["expect"]]
        it_ok = sum(1 for r in it if r["checks"].get("intent"))
        an = [r for r in rs if r["expect"].get("status", 200) == 200]
        an_ok = sum(1 for r in an if r["answer_correct"])
        st_ok = sum(1 for r in rs if r["checks"].get("status"))
        lat = [r["server_ms"] for r in rs if r.get("server_ms")]
        return (
            f"| {cat} | {len(rs)} | {st_ok / len(rs) * 100:.1f}% | "
            f"{(it_ok / len(it) * 100 if it else float('nan')):.1f}% | "
            f"{(an_ok / len(an) * 100 if an else float('nan')):.1f}% | "
            f"{(statistics.mean(lat) if lat else 0):.0f}ms |"
        )

    failures = [r for r in records if not (r["checks"].get("status") and r["answer_correct"])]

    lines = []
    lines.append("# RAG 智能客服 —— 准确率评测报告（N4）\n")
    lines.append(f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- 被测服务：`{base}`")
    lines.append(f"- 评测集：`data/eval/eval_set.json`（用例 {cases_total} 条 / 请求轮次 {turns_total} 次）")
    lines.append(f"- 评测耗时：{elapsed_s:.1f}s")
    lines.append(f"- 缓存处理：{json.dumps(clear_stats, ensure_ascii=False)}")
    lines.append(f"- 关键配置：fast_path={settings.fast_path_enabled} confidence_threshold={settings.confidence_threshold} "
                 f"rerank_max_candidates={settings.rerank_max_candidates}\n")

    lines.append("## 一、核心指标\n")
    lines.append("| 指标 | 结果 | 需求目标 | 判定 |")
    lines.append("|------|------|---------|------|")
    lines.append(f"| **意图识别准确率** | **{intent_acc * 100:.1f}%**（{intent_ok}/{len(intent_turns)}） | >90% | {'✅ 达标' if intent_acc > 0.9 else '❌ 未达标'} |")
    lines.append(f"| **答案正确率** | **{answer_acc * 100:.1f}%**（{answer_ok}/{len(answer_turns)}） | >85% | {'✅ 达标' if answer_acc > 0.85 else '❌ 未达标'} |")
    lines.append(f"| HTTP 状态符合率 | {status_acc * 100:.1f}%（{status_ok}/{turns_total}） | — | — |")
    lines.append("")
    lines.append("## 二、延迟（服务端 latency_ms）\n")
    lines.append("| 统计量 | 值 |")
    lines.append("|--------|-----|")
    if server_lat:
        lines.append(f"| 平均 | {statistics.mean(server_lat):.0f}ms |")
        lines.append(f"| P50 | {pct(server_lat, 0.50):.0f}ms |")
        lines.append(f"| P95 | {pct(server_lat, 0.95):.0f}ms |")
        lines.append(f"| **P99** | **{pct(server_lat, 0.99):.0f}ms** |")
        lines.append(f"| 最大 | {max(server_lat):.0f}ms |")
    if client_lat:
        lines.append(f"| 客户端往返均值 | {statistics.mean(client_lat):.0f}ms |")
    lines.append("")
    lines.append("## 三、分类别表现\n")
    lines.append("| 类别 | 轮次 | 状态符合率 | 意图准确率 | 答案正确率 | 平均耗时 |")
    lines.append("|------|------|-----------|-----------|-----------|---------|")
    for cat in sorted(by_cat):
        lines.append(cat_row(cat, by_cat[cat]))
    lines.append("")
    lines.append("## 四、失败用例明细\n")
    if not failures:
        lines.append("无失败用例 ✅\n")
    else:
        lines.append(f"共 {len(failures)} 条（最多展示 25 条）：\n")
        lines.append("| 用例 | 轮次 | 问题 | 期望 | 实际 | 失败断言 |")
        lines.append("|------|------|------|------|------|---------|")
        for r in failures[:25]:
            bad = [k for k, v in r["checks"].items() if not v]
            exp = json.dumps(r["expect"], ensure_ascii=False)[:70]
            act = f"status={r['status']} intent={r.get('intent')} source={r.get('source')} conf={r.get('confidence')}"
            lines.append(f"| {r['case']} | {r['turn']} | {r['question']} | {exp} | {act} | {','.join(bad)} |")
        lines.append("")

    lines.append("## 五、评测口径说明\n")
    lines.append("- **意图识别准确率**：仅统计评测集中标注了 `intent` 且返回 200 的轮次")
    lines.append("- **答案正确率**：`source` 相符 且 关键词断言通过 且 反向断言通过 且 改写断言通过")
    lines.append("  - `must_contain` 要求全部命中；`must_contain_any` 任一命中即可（规避 LLM 措辞差异）")
    lines.append("  - 多轮追问额外校验 `rewritten_contains`（指代消解后应带订单号）与 `rewritten_not_contains`（政策护栏不应注入订单号）")
    lines.append("- **HTTP 状态符合率**：覆盖 401（鉴权）/403（越权）/404（实体不存在）/422（参数非法）等异常路径")
    lines.append("- 未标注内容断言的用例（闲聊、乱码等）仅以 HTTP 状态判定其正确性")
    lines.append("- **延迟**：服务端 `latency_ms`（含缓存命中与冷启动混合分布），客户端往返含网络与 Python 序列化开销")
    lines.append("- 评测前默认清空评测问题的**答案缓存**，确保测的是真实生成而非历史缓存\n")

    summary = {
        "intent_acc": intent_acc, "answer_acc": answer_acc, "status_acc": status_acc,
        "turns": turns_total, "cases": cases_total,
        "mean_ms": statistics.mean(server_lat) if server_lat else 0,
        "p50_ms": pct(server_lat, 0.50), "p95_ms": pct(server_lat, 0.95), "p99_ms": pct(server_lat, 0.99),
    }
    return "\n".join(lines), summary


async def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 客服系统准确率评测（N4）")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--set", default=str(ROOT / "data" / "eval" / "eval_set.json"))
    ap.add_argument("--report", default=str(ROOT / "docs" / "eval_report.md"))
    ap.add_argument("--category", default=None, help="只跑某一类别")
    ap.add_argument("--limit", type=int, default=0, help="限制用例数（调试用）")
    ap.add_argument("--clear-all", action="store_true", help="连 query_transform/rerank 缓存一并清空")
    ap.add_argument("--keep-cache", action="store_true", help="不清缓存（快速冒烟）")
    args = ap.parse_args()

    spec = json.loads(Path(args.set).read_text(encoding="utf-8"))
    cases = spec["cases"]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]

    questions = [t["question"] for c in cases for t in c["turns"]]
    clear_stats = {} if args.keep_cache else await clear_eval_cache(questions, args.clear_all)

    print(f"评测集: {args.set}（用例 {len(cases)} 条 / 轮次 {sum(len(c['turns']) for c in cases)}）")
    print(f"被测服务: {args.base}")
    print(f"缓存处理: {clear_stats or '保留'}\n")

    records: list[dict] = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.base, timeout=180) as client:
        for i, case in enumerate(cases, 1):
            recs = await run_case(client, case, clear_sessions=True)
            records.extend(recs)
            mark = "✅" if all(r["checks"].get("status") and r["answer_correct"] for r in recs) else "❌"
            print(f"[{i:>2}/{len(cases)}] {mark} {case['id']:<26} {recs[-1]['question'][:26]} "
                  f"→ status={recs[-1]['status']} source={recs[-1].get('source')} "
                  f"{recs[-1]['server_ms'] or 0}ms")
    elapsed = time.perf_counter() - t0

    report, summary = summarize(records, len(cases), elapsed, args.base, clear_stats)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"意图识别准确率 : {summary['intent_acc'] * 100:.1f}%   （目标 >90%）")
    print(f"答案正确率     : {summary['answer_acc'] * 100:.1f}%   （目标 >85%）")
    print(f"HTTP 状态符合率: {summary['status_acc'] * 100:.1f}%")
    print(f"延迟 平均/P50/P95/P99: {summary['mean_ms']:.0f} / {summary['p50_ms']:.0f} / "
          f"{summary['p95_ms']:.0f} / {summary['p99_ms']:.0f} ms")
    print(f"评测耗时       : {elapsed:.1f}s")
    print(f"报告已写入     : {out}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(close_redis())
