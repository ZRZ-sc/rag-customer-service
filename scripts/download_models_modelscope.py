"""从 ModelScope 下载本地模型（HuggingFace 不可达时的备选源）。

用法：.venv\\Scripts\\python.exe scripts\\download_models_modelscope.py

下载完成后将 .env 中 EMBEDDING_MODEL / RERANK_MODEL 指向本地目录：
  EMBEDDING_MODEL=models/bge-small-zh-v1.5
  RERANK_MODEL=models/bge-reranker-v2-m3
"""
import os

from modelscope import snapshot_download

MODELS = {
    "BAAI/bge-small-zh-v1.5": "models/bge-small-zh-v1.5",
    "BAAI/bge-reranker-v2-m3": "models/bge-reranker-v2-m3",
}


def main() -> None:
    for model_id, local_dir in MODELS.items():
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            print(f"[skip] {model_id} already in {local_dir}")
            continue
        print(f"[download] {model_id} -> {local_dir}")
        snapshot_download(model_id, local_dir=local_dir)
        print(f"[ok] {model_id}")
    print("全部模型下载完成")


if __name__ == "__main__":
    main()
