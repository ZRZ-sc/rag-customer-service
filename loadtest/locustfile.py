"""Locust 压测脚本：三套负载画像，定位系统瓶颈。

用法（由 scripts/loadtest.ps1 调用，也可手动执行）：
  $env:LT_PROFILE="cache";   locust -f loadtest/locustfile.py --headless -u 200 -r 20 -t 90s --csv loadtest/results/cache
  $env:LT_PROFILE="precise"; locust -f loadtest/locustfile.py --headless -u 200 -r 20 -t 60s --csv loadtest/results/precise
  $env:LT_PROFILE="rag";     locust -f loadtest/locustfile.py --headless -u 20  -r 2  -t 120s --csv loadtest/results/rag

负载画像（关键：不同画像的瓶颈完全不同）：
  cache   —— 缓存命中为主（知识类重复问题 + 精确查询）：测吞吐上限，不加载本地模型
  precise —— 100% 订单/物流查询：测 PostgreSQL 连接池与异步吞吐
  rag     —— 100% 冷启动知识类（问题带唯一后缀，永不命中缓存）：测 CPU 重排 + LLM API 饱和点

环境变量：
  LT_BASE     被测服务地址（默认 http://127.0.0.1:8001，压测专用实例）
  LT_TOKEN    API Token（默认 dev-token-change-me）
  LT_PROFILE  cache | precise | rag | mixed（默认 mixed）
  LT_STAGES   负载阶梯，格式 "用户数:持续秒,用户数:持续秒"，设置后启用阶梯模式
"""
import itertools
import json
import os
import random
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

ROOT = Path(__file__).resolve().parent.parent
BASE = os.getenv("LT_BASE", "http://127.0.0.1:8001")
TOKEN = os.getenv("LT_TOKEN", "dev-token-change-me")
PROFILE = os.getenv("LT_PROFILE", "mixed")
STAGES = os.getenv("LT_STAGES", "")

# ---- 测试数据（复用知识库样例，保证问题真实可检索） ----
with open(ROOT / "data" / "faq" / "sample_faq.json", encoding="utf-8") as f:
    FAQ = [item["question"] for item in json.load(f)]

ORDERS = ["ORD20240100001", "ORD20240100002", "ORD20240100003", "ORD20240100004"]
TRACKINGS = ["SF123456789012", "YT987654321098"]

# 订单/物流 -> 归属用户：必须匹配，否则命中越权校验返回 403，污染容量测试结果
# （实测踩过：随机订单 + 固定 user_001 导致 1172 次 403，把错误率推到 10%）
ORDER_OWNERS = {
    "ORD20240100001": "user_001",
    "ORD20240100002": "user_001",
    "ORD20240100003": "user_001",
    "ORD20240100004": "user_002",
    "ORD20240100005": "user_002",
    "ORD20240100006": "user_002",
    "ORD20240100007": "user_003",
    "ORD20240100008": "user_003",
}
TRACKING_OWNERS = {"SF123456789012": "user_001", "YT987654321098": "user_002"}
ORDERS = list(ORDER_OWNERS)
TRACKINGS = list(TRACKING_OWNERS)

_counter = itertools.count(1)


class _Base(HttpUser):
    abstract = True
    host = BASE
    wait_time = between(0.01, 0.05)   # 压测下尽量少等待，靠并发数控制压力

    def ask(self, question: str, session_id: str, user_id: str = "user_001", token: str = TOKEN):
        """发送一次问答请求（使用 requests 会话，禁用 keep-alive 重试干扰）。"""
        payload = json.dumps(
            {"session_id": session_id, "question": question, "user_id": user_id}
        ).encode("utf-8")
        with self.client.post(
            "/api/v1/chat",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            name="POST /api/v1/chat",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
                return
            try:
                body = resp.json()
            except Exception:
                resp.failure("非 JSON 响应")
                return
            if body.get("code") != 0:
                resp.failure(f"业务错误码 {body.get('code')}: {body.get('message')}")
                return
            resp.success()


class CacheHitUser(_Base):
    """画像1：缓存命中为主（FAQ 重复问题 70% + 精确查询 30%）。"""

    weight = 7

    @task(7)
    def faq_cached(self):
        sid = f"lt-cache-{next(_counter) % 50}"
        self.ask(random.choice(FAQ), sid)

    @task(3)
    def precise(self):
        sid = f"lt-cache-{next(_counter) % 50}"
        if random.random() < 0.7:
            oid = random.choice(ORDERS)
            self.ask(f"我的订单 {oid} 到哪里了？", sid, user_id=ORDER_OWNERS[oid])
        else:
            trk = random.choice(TRACKINGS)
            self.ask(f"查询物流单号 {trk} 的进度", sid, user_id=TRACKING_OWNERS[trk])


class PreciseOnlyUser(_Base):
    """画像2：100% 精确查询（压力集中在 PostgreSQL 连接池）。"""

    weight = 10

    @task
    def precise(self):
        sid = f"lt-precise-{next(_counter) % 100}"
        if random.random() < 0.6:
            oid = random.choice(ORDERS)
            self.ask(f"我的订单 {oid} 到哪里了？", sid, user_id=ORDER_OWNERS[oid])
        else:
            trk = random.choice(TRACKINGS)
            self.ask(f"查询物流单号 {trk} 的进度", sid, user_id=TRACKING_OWNERS[trk])


class ColdRagUser(_Base):
    """画像3：100% 冷启动 RAG（唯一后缀保证不命中任何缓存）。

    用于测量 CPU 重排 + DeepSeek API 的饱和吞吐，以及排队导致的延迟膨胀。
    """

    weight = 1

    @task
    def cold_rag(self):
        n = next(_counter)
        sid = f"lt-rag-{n % 20}"
        self.ask(f"{random.choice(FAQ)}（场景编号{n}）", sid)


class UnauthorizedUser(_Base):
    """画像4：鉴权失败路径（401）。

    仅 mixed 画像启用：它会与目标接口争抢压测机资源，单独占用一个 Locust 统计行
    （POST /api/v1/chat [401]），容量测试中不参与主接口的吞吐统计。
    """

    weight = 1 if PROFILE == "mixed" else 0

    @task
    def bad_token(self):
        payload = json.dumps(
            {"session_id": "lt-auth", "question": "退货运费谁出？", "user_id": "user_001"}
        ).encode("utf-8")
        with self.client.post(
            "/api/v1/chat",
            data=payload,
            headers={"Authorization": "Bearer wrong-token", "Content-Type": "application/json"},
            name="POST /api/v1/chat [401]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 401:
                resp.success()
            else:
                resp.failure(f"期望 401，实际 {resp.status_code}")


# ---- 按画像启用/禁用用户类 ----
if PROFILE == "cache":
    PreciseOnlyUser.weight = 0
    ColdRagUser.weight = 0
elif PROFILE == "precise":
    CacheHitUser.weight = 0
    ColdRagUser.weight = 0
elif PROFILE == "rag":
    CacheHitUser.weight = 0
    PreciseOnlyUser.weight = 0


class StagedShape(LoadTestShape):
    """阶梯加压：LT_STAGES="20:30,50:30,100:45,200:60"（用户数:持续秒）。"""

    def __init__(self):
        super().__init__()
        self.plan = []
        if STAGES:
            for part in STAGES.split(","):
                users, secs = part.split(":")
                self.plan.append((int(users), int(secs)))

    def tick(self):
        if not self.plan:
            return None
        run_time = self.get_run_time()
        elapsed = 0
        for users, secs in self.plan:
            elapsed += secs
            if run_time < elapsed:
                return (users, max(1, users // 10))
        return None
