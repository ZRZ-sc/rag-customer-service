"""知识库导入（需求文档流程5）：FAQ JSON → PostgreSQL → 向量化 → Milvus → 清缓存。

用法：
  .venv\\Scripts\\python.exe scripts\\ingest_kb.py --file data/faq/sample_faq.json
  .venv\\Scripts\\python.exe scripts\\ingest_kb.py --file xxx.json --category 退换货
"""
import argparse
import asyncio
import json

from app.config import settings
from app.db import milvus
from app.db.postgres import close_pool, get_pool
from app.services.cache import invalidate_knowledge_cache
from app.services.embeddings import embed_texts


async def ingest_file(path: str, category: str | None) -> None:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not items:
        print("empty file, nothing to do")
        return

    pool = await get_pool()
    faq_ids: list[int] = []
    texts: list[str] = []
    for it in items:
        q = it["question"].strip()
        a = it["answer"].strip()
        cat = category or it.get("category", "general")
        row = await pool.fetchrow(
            """INSERT INTO faq_documents (question, answer, category, version, status, created_at, updated_at)
               VALUES ($1,$2,$3,1,'published', NOW(), NOW()) RETURNING id""",
            q, a, cat,
        )
        faq_ids.append(row["id"])
        texts.append(f"{q} {a}")

    # 批量向量化
    vecs = embed_texts(texts)
    milvus.ensure_collection(dim=settings.embedding_dim)
    milvus.insert_embeddings(
        [{"faq_id": fid, "embedding": vec.tolist()} for fid, vec in zip(faq_ids, vecs)]
    )
    # 同步向量回 PG（备用降级检索）
    for fid, vec in zip(faq_ids, vecs):
        vec_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
        await pool.execute(
            "UPDATE faq_documents SET embedding=$1::vector WHERE id=$2", vec_str, fid
        )

    cleared = await invalidate_knowledge_cache()
    print(f"[ok] ingested {len(items)} docs -> faq_ids={faq_ids}")
    print(f"[ok] milvus upserted, knowledge cache cleared: {cleared} keys")
    print("[!] 提示：如需应用内 BM25 索引立即刷新，请重启服务（或后续调用管理接口）")


async def delete_faq(faq_ids: list[int]) -> None:
    pool = await get_pool()
    await pool.execute(
        "UPDATE faq_documents SET status='archived' WHERE id = ANY($1::int[])", faq_ids
    )
    milvus.delete_by_faq_ids(faq_ids)
    await invalidate_knowledge_cache()
    print(f"[ok] archived faq_ids={faq_ids} and removed from milvus")


async def main() -> None:
    parser = argparse.ArgumentParser(description="FAQ knowledge base ingestion")
    parser.add_argument("--file", help="FAQ JSON 文件路径")
    parser.add_argument("--category", default=None, help="覆盖分类")
    parser.add_argument("--delete", nargs="+", type=int, help="按 faq_id 归档删除")
    args = parser.parse_args()

    try:
        if args.delete:
            await delete_faq(args.delete)
        elif args.file:
            await ingest_file(args.file, args.category)
        else:
            parser.print_help()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
