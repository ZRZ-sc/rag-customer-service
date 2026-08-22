"""本地嵌入模型：BAAI/bge-small-zh-v1.5（sentence-transformers，CPU）。"""
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """批量向量化（L2 归一化，维度=settings.embedding_dim）。"""
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)
    vecs = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0].tolist()


def preload() -> None:
    """启动预热：加载模型并做一次推理，避免首个请求卡顿。"""
    import numpy as np

    vecs = embed_texts(["预热"])
    assert vecs.shape[1] == settings.embedding_dim
