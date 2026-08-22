"""全局配置：从 .env 读取，pydantic-settings 自动映射环境变量。"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_file 列表：后列文件优先，故 .env.local 置于其后以覆盖 .env 中的配置
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "rag-customer-service"
    app_version: str = "0.1.0"
    debug: bool = False

    # ---- LLM (DeepSeek, OpenAI 兼容接口) ----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_timeout: float = 30.0
    llm_max_tokens: int = 1024

    # ---- 本地嵌入 / 重排模型 ----
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---- PostgreSQL ----
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "rag"
    pg_password: str = "ragpass"
    pg_database: str = "rag"
    pg_pool_min: int = 2
    pg_pool_max: int = 20

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"

    # ---- Milvus ----
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "faq_embeddings"
    milvus_index_type: str = "HNSW"

    # ---- 认证 ----
    api_token: str = "dev-token-change-me"

    # ---- RAG 参数 ----
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600          # 知识类答案缓存 TTL
    precise_cache_seconds: int = 30        # 精确查询短缓存（订单/物流实时性要求高）
    top_k_candidates: int = 50             # 混合检索召回数
    top_k_final: int = 5                   # Rerank 后精排数
    multi_query_count: int = 3             # Multi-Query 扩展数
    hyde_enabled: bool = True              # 是否启用 HyDE
    confidence_threshold: float = 0.5      # RAG Top-1 低于该值转人工/兜底

    # ---- 会话 ----
    session_ttl_seconds: int = 1800        # 会话 30 分钟过期
    session_max_turns: int = 10            # 最多保留 10 轮

    # ---- 可观测性 ----
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-customer-service"

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def llm_ready(self) -> bool:
        return bool(self.deepseek_api_key and not self.deepseek_api_key.startswith("sk-xxxx"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
