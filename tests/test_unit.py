"""单元测试：意图识别（正则）、检索工具（tokenize/RRF）。"""
import pytest

from app.services.intent import detect_entities, has_precise_signal
from app.services.rag.retriever import HybridRetriever, _tokenize


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
