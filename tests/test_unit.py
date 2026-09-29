"""单元测试：意图识别（正则）、检索工具（tokenize/RRF）、多轮上下文改写。"""
import pytest

from app.services.intent import (
    classify,
    detect_entities,
    has_knowledge_marker,
    has_precise_signal,
    has_status_intent,
)
from app.services.query_rewrite import (
    _parse_rewrite,
    _rewrite_with_memory,
    _strip_injected_entities,
    is_standalone,
    needs_rewrite,
    rewrite_query,
)
from app.services.rag.retriever import HybridRetriever, _tokenize


class _FakeMsg:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """替身 LLM：ainvoke 交由注入的协程处理，用于模拟成功/失败两种路径。"""

    def __init__(self, handler):
        self._handler = handler

    async def ainvoke(self, _messages):
        return await self._handler()


@pytest.fixture
def no_rewrite_cache(monkeypatch):
    """隔离真实 Redis：改写相关单测默认不读写缓存。"""
    from app.services import cache as cache_module

    async def _none(*args, **kwargs):
        return None

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(cache_module, "get_json", _none)
    monkeypatch.setattr(cache_module, "set_json", _noop)


@pytest.mark.parametrize(
    "question, expect_order, expect_tracking",
    [
        ("我的订单ORD20240100001到哪了？", ["ORD20240100001"], []),
        ("查询物流单号SF123456789012", [], ["SF123456789012"]),
        ("ORD20240100001和SF123456789012分别是什么状态", ["ORD20240100001"], ["SF123456789012"]),
        ("你们怎么退货", [], []),
        ("退款多久到账", [], []),
    ],
)
def test_detect_entities(question, expect_order, expect_tracking):
    ents = detect_entities(question)
    assert ents["order_ids"] == expect_order
    assert ents["tracking_nos"] == expect_tracking


@pytest.mark.parametrize(
    "question, expected",
    [
        ("我的订单到哪了", True),
        ("退货运费谁出", False),
        ("你好", False),
    ],
)
def test_has_precise_signal(question, expected):
    assert has_precise_signal(question, detect_entities(question)) is expected


def test_tokenize_zh_bigram():
    tokens = _tokenize("退货政策是什么")
    assert isinstance(tokens, list)
    assert len(tokens) >= 4
    assert "退货" in tokens


def test_rrf_merge():
    a = [{"faq_id": 1}, {"faq_id": 2}, {"faq_id": 3}]
    b = [{"faq_id": 2}, {"faq_id": 4}, {"faq_id": 1}]
    merged = HybridRetriever._rrf_merge([a, b], top_k=3)
    assert [m["faq_id"] for m in merged] == [2, 1, 4]
    assert "rrf_score" in merged[0]


def test_rrf_merge_methods_tag():
    a = [{"faq_id": 1, "method": "bm25"}]
    b = [{"faq_id": 1, "method": "vector"}]
    merged = HybridRetriever._rrf_merge([a, b], top_k=5)
    assert merged[0]["methods"] == ["bm25", "vector"]


# ---- 多轮上下文改写（指代消解） ----

_HISTORY = [
    {"role": "user", "content": "我的订单 ORD20240100001 到哪里了？"},
    {"role": "assistant", "content": "您的订单 ORD20240100001 当前状态：已签收。"},
]
_MEMORY = {"order_ids": ["ORD20240100001"], "tracking_nos": []}


@pytest.mark.parametrize(
    "question, expected",
    [
        # 独立问题：不改写
        ("我的订单 ORD20240100001 到哪里了？", False),
        ("退货运费谁出？", False),
        # 代词 / 省略追问：需改写
        ("那什么时候能到？", True),
        ("它现在到哪了", True),
        ("这个订单还能退吗", True),
        # 无实体的状态追问 + 会话有实体记忆：需改写
        ("什么时候能到？", True),
        ("发货了吗", True),
    ],
)
def test_needs_rewrite(question, expected):
    assert needs_rewrite(question, _HISTORY, _MEMORY) is expected


def test_needs_rewrite_requires_history():
    # 无历史（新会话）不触发改写 → 保证 session 隔离
    assert needs_rewrite("那什么时候能到？", [], _MEMORY) is False


def test_needs_rewrite_skips_when_entity_present():
    assert needs_rewrite("那 ORD20240100009 呢", _HISTORY, _MEMORY) is False


def test_parse_rewrite_json_and_plain():
    assert _parse_rewrite(
        '{"rewritten": "订单 ORD1 什么时候能到？", "used_context": true}'
    ) == "订单 ORD1 什么时候能到？"
    assert _parse_rewrite('```json\n{"rewritten": "发货了吗？"}\n```') == "发货了吗？"
    assert _parse_rewrite("") is None


def test_rewrite_with_memory():
    assert _rewrite_with_memory("什么时候能到？", _MEMORY) == "订单 ORD20240100001：什么时候能到？"
    assert _rewrite_with_memory("到哪了", {"order_ids": [], "tracking_nos": ["SF123"]}) == (
        "物流单号 SF123：到哪了"
    )
    assert _rewrite_with_memory("到哪了", {"order_ids": [], "tracking_nos": []}) is None


async def test_rewrite_query_falls_back_to_memory_without_llm(monkeypatch, no_rewrite_cache):
    """LLM 不可用时，仍应通过会话实体记忆完成指代消解（降级路径）。"""

    async def _boom():
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("app.services.query_rewrite.get_llm", lambda: _FakeLLM(_boom))
    res = await rewrite_query("那什么时候能到？", _HISTORY, _MEMORY)
    assert res["applied"] is True
    assert res["method"] == "entity_memory"
    assert "ORD20240100001" in res["rewritten"]


async def test_rewrite_query_llm_path(monkeypatch, no_rewrite_cache):
    async def _ok():
        return _FakeMsg('{"rewritten": "订单 ORD20240100001 什么时候能到？", "used_context": true}')

    monkeypatch.setattr("app.services.query_rewrite.get_llm", lambda: _FakeLLM(_ok))
    res = await rewrite_query("那什么时候能到？", _HISTORY, _MEMORY)
    assert res["applied"] is True
    assert res["method"] == "llm"
    assert res["rewritten"].startswith("订单 ORD20240100001")


async def test_rewrite_query_cache_hit_skips_llm(monkeypatch):
    """改写缓存命中时不得再调用 LLM。"""

    async def _hit(key, name="json"):
        return {"rewritten": "订单 ORD20240100001 什么时候能到？", "method": "llm"}

    async def _noop(*args, **kwargs):
        return None

    def _boom():
        raise AssertionError("缓存命中时不应调用 LLM")

    from app.services import cache as cache_module

    monkeypatch.setattr(cache_module, "get_json", _hit)
    monkeypatch.setattr(cache_module, "set_json", _noop)
    monkeypatch.setattr("app.services.query_rewrite.get_llm", lambda: _FakeLLM(_boom))
    res = await rewrite_query("那什么时候能到？", _HISTORY, _MEMORY)
    assert res["cached"] is True
    assert res["applied"] is True
    assert "ORD20240100001" in res["rewritten"]


async def test_rewrite_query_noop_for_standalone_question():
    res = await rewrite_query("退货运费谁出？", _HISTORY, _MEMORY)
    assert res["applied"] is False
    assert res["rewritten"] == "退货运费谁出？"


# ---- 政策类问题护栏（避免被误路由到精确查询） ----


@pytest.mark.parametrize(
    "question, expected",
    [
        ("那退货运费谁出？", True),
        ("退货流程是什么", True),
        ("怎么申请售后", True),
        ("怎么开发票", True),
        # N4 评测发现的漏词：曾导致多轮追问被误路由到数据库
        ("那还能换货吗？", True),
        ("那退款多久到账？", True),
        ("这个订单的维修政策是什么", True),
        ("我的订单什么时候能到", False),
        ("那什么时候能到？", False),
        ("物流到哪了", False),
    ],
)
def test_has_knowledge_marker(question, expected):
    assert has_knowledge_marker(question) is expected


async def test_classify_policy_followups_stay_knowledge():
    """N4 评测回归：带订单号的政策类追问必须走知识类，而不是查订单状态。"""
    for q in ("订单 ORD20240100001 还能换货吗？", "订单 ORD20240100001 退款多久到账？"):
        res = await classify(q)
        assert res["intent"] == "knowledge", q
        assert res["method"] == "policy_keyword", q


# ---- 压测发现的跨用户上下文泄漏回归 ----


@pytest.mark.parametrize(
    "question, expected",
    [
        ("那什么时候能到？", True),
        ("发货了吗", True),
        ("它现在到哪了", True),
        ("我的订单什么时候能到", True),
        ("可以指定快递公司吗？", False),      # 规则类问题，主题词不是状态意图
        ("退货的运费谁出？", False),
        ("怎么开具发票？", False),
    ],
)
def test_has_status_intent(question, expected):
    assert has_status_intent(question) is expected


def test_complete_faq_question_is_not_followup():
    """压测回归：完整 FAQ 问题（即使较短、以"吗"结尾）不得被判为追问。

    否则 LLM 会把同一 session 里其他用户的订单号注入改写结果，
    导致知识问答被误路由为订单查询并触发越权 403。
    """
    assert needs_rewrite("可以指定快递公司吗？", _HISTORY, _MEMORY) is False
    assert needs_rewrite("怎么开具发票？", _HISTORY, _MEMORY) is False
    assert is_standalone("可以指定快递公司吗？") is True


async def test_rewrite_guard_blocks_entity_injection_for_non_status_question(
    monkeypatch, no_rewrite_cache
):
    """即使会话中存在订单记忆，非状态类问题的改写结果也不得带订单号。"""

    async def _inject():
        return _FakeMsg(
            '{"rewritten": "订单 ORD20240100001 可以指定快递公司吗？", "used_context": true}'
        )

    monkeypatch.setattr("app.services.query_rewrite.get_llm", lambda: _FakeLLM(_inject))
    res = await rewrite_query("可以指定快递公司吗？", _HISTORY, _MEMORY)
    assert "ORD20240100001" not in res["rewritten"]
    assert "快递公司" in res["rewritten"]


async def test_entity_memory_is_user_bound(monkeypatch):
    """实体记忆与写入者绑定：他人读取应得到空结果（跨用户隔离）。"""
    from app.services import conversation as conv

    stored = {}

    class _FakeRedis:
        async def get(self, key):
            return stored.get(key)

        async def set(self, key, value, ex=None):
            stored[key] = value

        async def delete(self, *keys):
            for k in keys:
                stored.pop(k, None)

    monkeypatch.setattr(conv, "get_redis", lambda: _FakeRedis())

    await conv.set_last_entities("sess-1", {"order_ids": ["ORD20240100001"], "tracking_nos": []}, "user_001")
    mine = await conv.get_last_entities("sess-1", "user_001")
    other = await conv.get_last_entities("sess-1", "user_002")
    assert mine["order_ids"] == ["ORD20240100001"]
    assert other["order_ids"] == []           # 他人不可见


def test_strip_injected_entities():
    assert (
        _strip_injected_entities("订单 ORD20240100002 的退货运费由谁承担？")
        == "退货运费由谁承担？"
    )
    assert (
        _strip_injected_entities("物流单号 SF123456789012 的运费谁出？") == "运费谁出？"
    )
    # 无实体时原样返回
    assert _strip_injected_entities("退货运费谁出？") == "退货运费谁出？"


async def test_classify_policy_marker_beats_entity():
    """带订单号的政策类问题应判为 knowledge，而不是查订单状态。"""
    res = await classify("订单 ORD20240100002 的退货运费谁出？")
    assert res["intent"] == "knowledge"
    assert res["method"] == "policy_keyword"


async def test_classify_status_question_still_precise():
    res = await classify("我的订单 ORD20240100002 到哪了？")
    assert res["intent"] == "precise_query"
    assert res["method"] == "regex"


async def test_rewrite_guard_strips_entity_for_policy_question(monkeypatch, no_rewrite_cache):
    """LLM 给政策类问题注入了订单号 → 护栏必须剥离，避免误路由到数据库。"""

    async def _ok():
        return _FakeMsg(
            '{"rewritten": "订单 ORD20240100001 的退货运费由谁承担？", "used_context": true}'
        )

    monkeypatch.setattr("app.services.query_rewrite.get_llm", lambda: _FakeLLM(_ok))
    res = await rewrite_query("那退货运费谁出？", _HISTORY, _MEMORY)
    assert res["applied"] is True
    assert res["method"] == "llm+policy_guard"
    assert "ORD20240100001" not in res["rewritten"]
    assert "运费" in res["rewritten"]


# ---- 阶段3：多级缓存与快速路径 ----


def test_cache_key_formats():
    from app.services import cache as cache_svc

    assert cache_svc.answer_cache_key("  退货  政策 ").startswith("rag:answer:")
    assert cache_svc.rewrite_cache_key("那什么时候能到？", "fp").startswith("query_rewrite:")
    assert cache_svc.transform_cache_key("退货政策").startswith("query_transform:")
    assert cache_svc.rerank_cache_key("退货政策", [1, 2]).startswith("rerank:")


def test_answer_cache_key_normalized():
    """大小写/多余空格归一化后应命中同一个键。"""
    from app.services import cache as cache_svc

    assert cache_svc.answer_cache_key("退货  Policy") == cache_svc.answer_cache_key("  退货 policy  ")


def test_rerank_cache_key_order_insensitive_but_content_sensitive():
    from app.services import cache as cache_svc

    # 候选集合相同（顺序无关）→ 同一个键
    assert cache_svc.rerank_cache_key("q", [3, 1, 2]) == cache_svc.rerank_cache_key("q", [2, 3, 1])
    # 候选集合变化 → 不同键（知识库变更后自然失效）
    assert cache_svc.rerank_cache_key("q", [1, 2]) != cache_svc.rerank_cache_key("q", [1, 2, 3])
    # 问题变化 → 不同键
    assert cache_svc.rerank_cache_key("q1", [1, 2]) != cache_svc.rerank_cache_key("q2", [1, 2])


def test_rewrite_cache_key_includes_context_fingerprint():
    """同一句话在不同上下文（不同订单）下必须产生不同缓存键，防止跨会话串味。"""
    from app.services import cache as cache_svc

    k1 = cache_svc.rewrite_cache_key("那什么时候能到？", "ORD20240100001||前文A")
    k2 = cache_svc.rewrite_cache_key("那什么时候能到？", "ORD20240100002||前文A")
    assert k1 != k2


@pytest.mark.parametrize(
    "question, expected",
    [
        ("七天无理由退货的条件是什么？", True),
        ("怎么开具发票？", True),
        ("那什么时候能到？", False),
        ("它到哪了", False),
        ("什么时候能到？", False),
        ("发货了吗", False),
    ],
)
def test_is_standalone(question, expected):
    """独立问题可跳过历史读取直接探测 L1 缓存；追问类不行。"""
    assert is_standalone(question) is expected


def test_fast_path_eligibility():
    from app.services.rag import pipeline

    cands = [{"faq_id": i} for i in (1, 2, 3)]
    # 高相似度且分差明显 → 走快速路径
    assert pipeline._fast_path_eligible(
        [{"faq_id": 1, "vector_score": 0.92}, {"faq_id": 2, "vector_score": 0.70},
         {"faq_id": 3, "vector_score": 0.60}],
        cands,
    ) is True
    # 分差不足 → 走完整路径
    assert pipeline._fast_path_eligible(
        [{"faq_id": 1, "vector_score": 0.90}, {"faq_id": 2, "vector_score": 0.88},
         {"faq_id": 3, "vector_score": 0.60}],
        cands,
    ) is False
    # 相似度不足 → 走完整路径
    assert pipeline._fast_path_eligible(
        [{"faq_id": 1, "vector_score": 0.55}, {"faq_id": 2, "vector_score": 0.20},
         {"faq_id": 3, "vector_score": 0.10}],
        cands,
    ) is False
    # 候选过少 → 走完整路径
    assert pipeline._fast_path_eligible([{"faq_id": 1, "vector_score": 0.95}], [{"faq_id": 1}]) is False


async def test_rerank_cache_hit_reconstructs_ranking(monkeypatch):
    """命中重排缓存时，应按缓存的分数重建排序，且不调用 CPU 重排。"""
    from app.services.rag import pipeline

    cached = {"scores": {"1": 0.90, "2": 0.30, "3": 0.55}}
    called = {"rerank": 0}

    async def fake_get(key, name="json"):
        return cached

    async def fake_set(*args, **kwargs):
        return None

    def fake_rerank(*args, **kwargs):
        called["rerank"] += 1
        return []

    monkeypatch.setattr(pipeline.cache_svc, "get_json", fake_get)
    monkeypatch.setattr(pipeline.cache_svc, "set_json", fake_set)
    monkeypatch.setattr(pipeline.reranker, "rerank", fake_rerank)

    cands = [
        {"faq_id": 1, "question": "q1", "answer": "a1"},
        {"faq_id": 2, "question": "q2", "answer": "a2"},
        {"faq_id": 3, "question": "q3", "answer": "a3"},
    ]
    ranked, flag = await pipeline._rerank_with_cache("q", cands, top_k=2)
    assert flag == "hit"
    assert called["rerank"] == 0
    assert [c["faq_id"] for c in ranked] == [1, 3]


# ---- 第一阶段补齐：429 限流 / 转人工（F6） / 反馈 API ----

from app.core.errors import RateLimitError  # noqa: E402


class _FakeScript:
    """替身 Redis Lua 脚本：返回预设结果或抛出异常。"""

    def __init__(self, result=None, error=None):
        self.result = result if result is not None else [1, 59]
        self.error = error

    async def __call__(self, keys=None, args=None):
        if self.error:
            raise self.error
        return self.result


class _FakeRedis:
    def __init__(self, script):
        self._script = script

    def register_script(self, _lua):
        return self._script


class _FakeRequest:
    def __init__(self, host="10.0.0.9"):
        self.client = type("Client", (), {"host": host})()


@pytest.fixture(autouse=True)
def _reset_rate_limit_script():
    """Lua 脚本对象是模块级缓存，逐用例重置避免串扰。"""
    from app.services import rate_limit

    rate_limit._script = None
    yield
    rate_limit._script = None


async def test_rate_limit_allows_within_bucket(monkeypatch):
    from app.services import rate_limit

    monkeypatch.setattr(rate_limit, "get_redis", lambda: _FakeRedis(_FakeScript([1, 59])))
    allowed, remaining = await rate_limit.check_bucket("k", capacity=60, refill_per_sec=1.0)
    assert allowed is True
    assert remaining == 59


async def test_rate_limit_blocks_when_exhausted(monkeypatch):
    from app.services import rate_limit

    monkeypatch.setattr(rate_limit, "get_redis", lambda: _FakeRedis(_FakeScript([0, 0])))
    allowed, remaining = await rate_limit.check_bucket("k", capacity=60, refill_per_sec=1.0)
    assert allowed is False
    assert remaining == 0


async def test_rate_limit_fail_open_on_redis_error(monkeypatch):
    """Redis 故障时必须放行（需求文档 N8：依赖故障不得让服务整体不可用）。"""
    from app.services import rate_limit

    monkeypatch.setattr(
        rate_limit, "get_redis", lambda: _FakeRedis(_FakeScript(error=RuntimeError("redis down")))
    )
    allowed, remaining = await rate_limit.check_bucket("k", capacity=42, refill_per_sec=1.0)
    assert allowed is True
    assert remaining == 42


async def test_enforce_raises_429_with_retry_after(monkeypatch):
    from app.services import rate_limit

    async def _blocked(*args, **kwargs):
        return False, 0

    monkeypatch.setattr(rate_limit, "check_bucket", _blocked)
    with pytest.raises(RateLimitError) as err:
        await rate_limit.enforce("chat", "user:u1")
    assert err.value.code == 429
    assert err.value.http_status == 429
    assert err.value.headers.get("Retry-After")


async def test_enforce_disabled_allows_all(monkeypatch):
    from app.services import rate_limit

    monkeypatch.setattr(rate_limit.settings, "rate_limit_enabled", False)

    async def _should_not_call(*args, **kwargs):
        raise AssertionError("关闭限流后不应执行桶判定")

    monkeypatch.setattr(rate_limit, "check_bucket", _should_not_call)
    remaining = await rate_limit.enforce("chat", "user:u1")
    assert remaining > 0


def test_rate_limit_identity_prefers_user_id():
    from app.services import rate_limit

    assert rate_limit.identity_of("user_001", _FakeRequest()) == "user:user_001"
    assert rate_limit.identity_of(None, _FakeRequest("127.0.0.1")) == "ip:127.0.0.1"


def test_ticket_id_format_and_queue_hint():
    from app.services import support

    tid = support.new_ticket_id()
    assert tid.startswith("HO")
    assert len(tid) == 2 + 8 + 8              # HO + YYYYMMDD + 8 位随机
    assert "尽快" in support.queue_hint(1)
    assert "分钟" in support.queue_hint(5)


def test_handoff_and_feedback_model_validation():
    from app.models.support import FeedbackRequest, HandoffRequest

    assert HandoffRequest(session_id="s1").priority == "normal"
    assert FeedbackRequest(request_id="abc123", feedback=1).feedback == 1
    assert FeedbackRequest(request_id="abc123", feedback=-1).feedback == -1
    with pytest.raises(Exception):
        FeedbackRequest(request_id="abc123", feedback=5)        # 仅允许 1/-1/0
    with pytest.raises(Exception):
        HandoffRequest(session_id="s1", priority="urgent")       # 仅允许 low/normal/high


def _pool_returning(row):
    class _Pool:
        async def fetchrow(self, *args, **kwargs):
            return row

    async def _get_pool():
        return _Pool()

    return _get_pool


def _pool_scripted(owner_row, update_row="__same__"):
    """按 SQL 分流：SELECT 查归属，UPDATE 写反馈。记录被执行的语句以便断言。"""
    calls = []

    class _Pool:
        async def fetchrow(self, sql, *args, **kwargs):
            calls.append(sql)
            if "SELECT user_id" in sql:
                return owner_row
            return owner_row if update_row == "__same__" else update_row

    async def _get_pool():
        return _Pool()

    return _get_pool, calls


async def test_save_feedback_status_mapping(monkeypatch):
    from app.services import support

    # request_id 不存在
    monkeypatch.setattr(support, "get_pool", _pool_returning(None))
    status, data = await support.save_feedback("r1", "user_001", 1, None)
    assert status == "not_found" and data is None

    # 归属不符（他人反馈）→ 必须拒绝，且不得执行 UPDATE
    pool_factory, calls = _pool_scripted(owner_row={"user_id": "user_002"})
    monkeypatch.setattr(support, "get_pool", pool_factory)
    status, _ = await support.save_feedback("r1", "user_001", 1, None)
    assert status == "forbidden"
    assert not any("UPDATE dialogue_logs" in sql for sql in calls), "越权请求不得写入数据库"

    # 正常写入
    pool_factory, calls = _pool_scripted(
        owner_row={"user_id": "user_001"},
        update_row={"user_id": "user_001", "session_id": "s1", "feedback": 1, "feedback_at": None},
    )
    monkeypatch.setattr(support, "get_pool", pool_factory)
    status, data = await support.save_feedback("r1", "user_001", 1, "很有帮助")
    assert status == "updated"
    assert data["feedback"] == 1
    assert any("UPDATE dialogue_logs" in sql for sql in calls)


async def test_create_ticket_persists(monkeypatch):
    from app.services import support

    captured = {}

    class _Pool:
        async def fetchrow(self, sql, *args):
            captured["sql"] = sql
            captured["args"] = args
            return {
                "ticket_id": args[0],
                "status": "pending",
                "priority": args[6],
                "source": args[5],
                "created_at": "2026-01-01T00:00:00",
            }

    async def _get_pool():
        return _Pool()

    monkeypatch.setattr(support, "get_pool", _get_pool)
    monkeypatch.setattr(support.settings, "handoff_webhook_url", "")   # 不触发 webhook

    ticket = await support.create_ticket(
        session_id="s1", user_id="user_001", reason="低置信度", last_question="??",
        source="auto", priority="high",
    )
    assert ticket["ticket_id"].startswith("HO")
    assert ticket["status"] == "pending"
    assert ticket["source"] == "auto"
    assert "INSERT INTO handoff_tickets" in captured["sql"]
