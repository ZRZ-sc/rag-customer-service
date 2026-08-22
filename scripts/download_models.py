"""预下载本地模型（bge-small-zh-v1.5 嵌入 + bge-reranker-v2-m3 重排），避免首次请求卡顿。

用法：.venv\\Scripts\\python.exe scripts\\download_models.py
"""
from app.config import settings


def main() -> None:
    print(f"[1/2] downloading embedding model: {settings.embedding_model}")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(settings.embedding_model, device="cpu")
    print("[ok] embedding model ready")

    print(f"[2/2] downloading rerank model: {settings.rerank_model}")
    from FlagEmbedding import FlagReranker
    FlagReranker(settings.rerank_model, use_fp16=False, device="cpu")
    print("[ok] rerank model ready")
    print("全部模型下载完成")


if __name__ == "__main__":
    main()
