"""Milvus 向量库客户端：Collection 管理、写入、检索。"""
from pymilvus import DataType, MilvusClient

from app.config import settings

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=settings.milvus_uri)
    return _client


def ensure_collection(dim: int | None = None) -> None:
    """确保 Collection 存在（幂等）。"""
    client = get_client()
    name = settings.milvus_collection
    if client.has_collection(name):
        return
    dim = dim or settings.embedding_dim
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="faq_id", datatype=DataType.INT64)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type=settings.milvus_index_type,
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(
        collection_name=name, schema=schema, index_params=index_params
    )
    client.load_collection(name)


def insert_embeddings(rows: list[dict]) -> None:
    """rows: [{"faq_id": int, "embedding": [float, ...]}]（collection 为 auto_id 主键）"""
    client = get_client()
    client.insert(collection_name=settings.milvus_collection, data=rows)


def delete_by_faq_ids(faq_ids: list[int]) -> None:
    client = get_client()
    ids_expr = ", ".join(str(i) for i in faq_ids)
    client.delete(
        collection_name=settings.milvus_collection,
        filter=f"faq_id in [{ids_expr}]",
    )


def search(query_vector: list[float], top_k: int) -> list[dict]:
    """返回 [{"faq_id": int, "score": float}]，score 为余弦相似度(0~1)。"""
    client = get_client()
    res = client.search(
        collection_name=settings.milvus_collection,
        data=[query_vector],
        limit=top_k,
        output_fields=["faq_id"],
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
        timeout=5,
    )
    hits = []
    for item in res[0]:
        hits.append({"faq_id": int(item["entity"]["faq_id"]), "score": float(item["distance"])})
    return hits


async def ping() -> bool:
    try:
        get_client().list_collections()
        return True
    except Exception:
        return False
