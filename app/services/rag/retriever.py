"""混合检索：BM25(关键词) + Milvus向量(语义) + RRF 无参数融合（需求文档流程2）。"""
import asyncio
import re
import threading
import time

import numpy as np
from rank_bm25 import BM25Okapi

from app.db.milvus import search as milvus_search
from app.db.postgres import get_pool
from app.services.embeddings import embed_texts

RRF_K = 60  # RRF 融合常数

_CJK_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    """中文按 bigram + 整词，英文/数字按 token 切分。"""
    text = text.lower()
    tokens: list[str] = []
    for seg in _CJK_RE.findall(text):
        if seg.isascii():
            tokens.append(seg)
        elif len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
            tokens.append(seg)
    return tokens


class HybridRetriever:
    def __init__(self):
        self.docs: list[dict] = []          # [{faq_id, question, answer, category}]
        self._bm25: BM25Okapi | None = None
        self._id2idx: dict[int, int] = {}
        self._lock = threading.Lock()
        self._last_load_attempt: float = 0.0
        self._load_retry_interval: float = 30.0

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    async def ensure_loaded(self) -> None:
        """自愈：启动时 PostgreSQL 不可用会导致语料为空，这里按冷却时间自动重试加载。"""
        if self.ready:
            return
        now = time.monotonic()
        if now - self._last_load_attempt < self._load_retry_interval:
            return
        self._last_load_attempt = now
        try:
            await self.load()
        except Exception:
            pass

    async def load(self) -> int:
        """从 PostgreSQL 加载已发布 FAQ 并构建 BM25 索引（启动时 + 知识库更新时调用）。"""
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT id, question, answer, category FROM faq_documents "
            "WHERE status='published' ORDER BY id"
        )
        docs = [
            {
                "faq_id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "category": r["category"],
            }
            for r in rows
        ]
        with self._lock:
            self.docs = docs
            self._id2idx = {d["faq_id"]: i for i, d in enumerate(docs)}
            corpus = [_tokenize(f"{d['question']} {d['answer']}") for d in docs]
            self._bm25 = BM25Okapi(corpus) if corpus else None
        return len(docs)

    def doc_by_id(self, faq_id: int) -> dict | None:
        idx = self._id2idx.get(faq_id)
        return self.docs[idx] if idx is not None else None

    def attach_docs(self, hits: list[dict]) -> list[dict]:
        """为检索命中补充原文（question/answer/category）。"""
        out = []
        for h in hits:
            d = self.doc_by_id(h["faq_id"])
            if d:
                h.update(d)
                out.append(h)
        return out

    # ---- 单路检索 ----

    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        if not self._bm25 or not self.docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        out = []
        for i in order:
            if scores[i] <= 0:
                continue
            out.append({"faq_id": self.docs[i]["faq_id"], "score": float(scores[i]), "method": "bm25"})
        return out

    async def vector_search(self, query: str, top_k: int) -> list[dict]:
        vec = await asyncio.to_thread(embed_texts, [query])
        hits = await asyncio.to_thread(milvus_search, vec[0].tolist(), top_k)
        return [
            {"faq_id": h["faq_id"], "score": h["score"], "method": "vector"} for h in hits
        ]

    # ---- RRF 融合 ----

    @staticmethod
    def _rrf_merge(ranked_lists: list[list[dict]], top_k: int) -> list[dict]:
        fused: dict[int, dict] = {}
        for ranked in ranked_lists:
            for rank, item in enumerate(ranked):
                fid = item["faq_id"]
                entry = fused.setdefault(
                    fid, {"faq_id": fid, "rrf_score": 0.0, "methods": set()}
                )
                entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
                entry["methods"].add(item.get("method", "?"))
                # 保留向量原始相似度：快速路径判定（高置信直出）需要它
                if item.get("method") == "vector":
                    entry["vector_score"] = max(
                        entry.get("vector_score", 0.0), float(item.get("score", 0.0))
                    )
        merged = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
        for m in merged:
            m["methods"] = sorted(m["methods"])
        return merged

    @staticmethod
    def vector_scores(hits: list[dict]) -> list[float]:
        """提取命中结果中的向量相似度（无向量命中记 0）。"""
        return [float(h.get("vector_score", 0.0)) for h in hits]

    # ---- 组合检索 ----

    async def hybrid_search(self, query: str, top_k: int = 50) -> list[dict]:
        """BM25 与向量检索并行执行，RRF 融合。"""
        await self.ensure_loaded()      # 语料为空时按冷却时间自动重载（自愈）
        bm25_res, vec_res = await asyncio.gather(
            asyncio.to_thread(self.bm25_search, query, top_k),
            self.vector_search(query, top_k),
        )
        return self._rrf_merge([bm25_res, vec_res], top_k)

    async def hybrid_search_multi(self, queries: list[str], top_k: int = 50) -> list[dict]:
        """Multi-Query：各查询并行混合检索，全局 RRF 融合。"""
        results = await asyncio.gather(*[self.hybrid_search(q, top_k) for q in queries])
        return self._rrf_merge(results, top_k)

    # ---- 降级 ----

    async def fallback_fulltext(self, query: str, top_k: int = 5) -> list[dict]:
        """Milvus 故障降级：PostgreSQL 模糊检索（需求文档流程6）。"""
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT id, question, answer, category FROM faq_documents "
            "WHERE status='published' AND (question ILIKE '%' || $1 || '%' "
            "OR answer ILIKE '%' || $1 || '%') LIMIT $2",
            query,
            top_k,
        )
        return [
            {
                "faq_id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "category": r["category"],
                "method": "fulltext",
                "rerank_score": 0.6,
            }
            for r in rows
        ]


hybrid_retriever = HybridRetriever()
