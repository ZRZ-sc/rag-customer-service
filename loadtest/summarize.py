"""汇总 Locust 压测结果：分阶段吞吐/延迟/错误率 + 资源占用峰值。

用法：
  .venv\\Scripts\\python.exe -m loadtest.summarize --prefix loadtest/results/cache
  .venv\\Scripts\\python.exe -m loadtest.summarize --prefix loadtest/results/rag --resource loadtest/results/resource_rag.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def read_stats(prefix: Path) -> list[dict]:
    path = prefix.with_name(prefix.name + "_stats.csv")
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_history(prefix: Path) -> list[dict]:
    path = prefix.with_name(prefix.name + "_stats_history.csv")
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("Name") == "Aggregated"]


def _num(value, default: float = 0.0) -> float:
    """Locust 在无样本时写入 'N/A'，需容错。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def staged_table(history: list[dict]) -> list[dict]:
    """按并发档位聚合（取该档位后半段作为稳定期），避免逐 10s 噪声。"""
    rows = []
    for r in history:
        rows.append({
            "ts": (r.get("Timestamp") or "")[11:19],
            "users": int(_num(r.get("User Count"))),
            "rps": _num(r.get("Requests/s")),
            "p50": _num(r.get("50%")),
            "p95": _num(r.get("95%")),
            "p99": _num(r.get("99%")),
            "fails": int(_num(r.get("Total Failure Count"))),
        })
    # 按并发档位分组，取该档位后半段（稳定期）均值
    buckets: dict[int, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["users"], []).append(row)
    out = []
    for users in sorted(buckets):
        samples = buckets[users]
        stable = samples[len(samples) // 2:] or samples
        out.append({
            "users": users,
            "seconds": len(samples) * 10,
            "rps": sum(s["rps"] for s in stable) / len(stable),
            "rps_peak": max(s["rps"] for s in samples),
            "p50": max(s["p50"] for s in stable),
            "p95": max(s["p95"] for s in stable),
            "p99": max(s["p99"] for s in stable),
            "fails": samples[-1]["fails"] - samples[0]["fails"],
        })
    return out


def resource_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None
    cpu = [float(r["cpu_pct"]) for r in rows if r.get("cpu_pct")]
    free = [float(r["mem_free_mb"]) for r in rows if r.get("mem_free_mb")]
    py = [float(r["python_mem_mb"]) for r in rows if r.get("python_mem_mb")]
    return {
        "samples": len(rows),
        "cpu_avg": sum(cpu) / len(cpu) if cpu else 0,
        "cpu_peak": max(cpu) if cpu else 0,
        "mem_free_min": min(free) if free else 0,
        "py_mem_peak": max(py) if py else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Locust 结果汇总")
    ap.add_argument("--prefix", required=True, help="Locust --csv 前缀（不含扩展名）")
    ap.add_argument("--resource", default=None, help="资源采样 CSV")
    args = ap.parse_args()

    prefix = Path(args.prefix)
    stats = read_stats(prefix)
    history = read_history(prefix)

    print(f"=== {prefix.name} ===")
    print("\n[总体]")
    print(f"{'名称':<26}{'请求数':>8}{'失败':>7}{'RPS':>9}{'中位':>8}{'P95':>9}{'P99':>9}{'最大':>10}")
    for s in stats:
        if not s.get("Name"):
            continue
        print(f"{s['Name']:<26}{int(s['Request Count']):>8}{int(s['Failure Count']):>7}"
              f"{float(s['Requests/s']):>9.1f}{float(s['50%']):>8.0f}{float(s['95%']):>9.0f}"
              f"{float(s['99%']):>9.0f}{float(s['100%']):>10.0f}")

    print("\n[分档位（稳定期取该档位后半段）]")
    print(f"{'并发':>6}{'持续s':>8}{'RPS均值':>10}{'RPS峰值':>9}{'P50':>8}{'P95':>9}{'P99':>9}{'新增失败':>10}")
    for b in staged_table(history):
        print(f"{b['users']:>6}{b['seconds']:>8}{b['rps']:>10.1f}{b['rps_peak']:>9.1f}"
              f"{b['p50']:>8.0f}{b['p95']:>9.0f}{b['p99']:>9.0f}{b['fails']:>10}")

    if args.resource:
        r = resource_summary(Path(args.resource))
        if r:
            print("\n[资源占用]")
            print(f"  采样点 {r['samples']}  CPU 均值 {r['cpu_avg']:.0f}%  峰值 {r['cpu_peak']:.0f}%")
            print(f"  空闲内存最低 {r['mem_free_min'] / 1024:.2f}GB  Python 进程内存峰值 {r['py_mem_peak']:.0f}MB")


if __name__ == "__main__":
    main()
