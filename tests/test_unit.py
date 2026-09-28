"""单元测试：意图识别（正则）、检索工具（tokenize/RRF）、多轮上下文改写。"""
import pytest

from app.services.intent import classify, detect_entities, has_knowledge_marker, has_precise_signal
from app.services.query_rewrite import (
    _parse_rewrite,
    _rewrite_with_memory,
    _strip_injected_entities,
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


async def test_rewrite_query_falls_back_to_memory_without_llm(monkeypatch):
    """LLM 不可用时，仍应通过会话实体记忆完成指代消解（降级路径）。"""

    async def _boom():
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("app.services.query_rewrite.get_llm", lambda: _FakeLLM(_boom))
    res = await rewrite_query("那什么时候能到？", _HISTORY, _MEMORY)
    assert res["applied"] is True
    assert res["method"] == "entity_memory"
    assert "ORD20240100001" in res["rewritten"]


async def test_rewrite_query_llm_path(monkeypatch):
    async def _ok():
        return _FakeMsg('{"rewritten": "订单 ORD20240100001 什么时候能到？", "used_context": true}')

    monkeypatch.setattr("app.services.query_rewrite.get_llm", lambda: _FakeLLM(_ok))
    res = await rewrite_query("那什么时候能到？", _HISTORY, _MEMORY)
    assert res["applied"] is True
    assert res["method"] == "llm"
    assert res["rewritten"].startswith("订单 ORD20240100001")


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
        ("我的订单什么时候能到", False),
        ("物流到哪了", False),
    ],
)
def test_has_knowledge_marker(question, expected):
    assert has_knowledge_marker(question) is expected


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


async def test_rewrite_guard_strips_entity_for_policy_question(monkeypatch):
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
