"""重排序：Cross-Encoder (BAAI/bge-reranker-*)，本地 CPU 运行。

性能/内存优化（针对 16GB 内存的 CPU 机器）：
- int8 动态量化：内存 ~700MB，速度约为 fp32 的 2-3 倍；
- 候选批量 32 打分；失败时回退 fp32。
"""
import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings


class Reranker:
    def __init__(self):
        self._model: torch.nn.Module | None = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._max_length = 512
        self._batch_size = 32

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model)
                    fp32_model = AutoModelForSequenceClassification.from_pretrained(
                        settings.rerank_model,
                        low_cpu_mem_usage=True,
                    ).eval()
                    try:
                        self._model = torch.quantization.quantize_dynamic(
                            fp32_model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                    except Exception:
                        self._model = fp32_model
        return self._model

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """对候选逐对打分（sigmoid 归一化 0~1），返回精排 top_k。"""
        if not candidates:
            return []
        model = self._get_model()
        pairs = [[query, f"{c['question']}\n{c['answer']}"] for c in candidates]
        all_scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self._batch_size):
                batch = pairs[i : i + self._batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                logits = model(**inputs).logits.squeeze(-1)  # (batch,)
                scores = torch.sigmoid(logits.float()).tolist()
                all_scores.extend(scores)
        for c, s in zip(candidates, all_scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def preload(self) -> None:
        """启动预热：加载模型并做一次打分。"""
        self.rerank("预热", [{"question": "退货运费谁出", "answer": "非质量问题的运费由买家承担"}], top_k=1)


reranker = Reranker()
