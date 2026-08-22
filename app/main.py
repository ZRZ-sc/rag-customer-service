"""FastAPI 应用入口。"""
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.config import settings
from app.core.errors import ApiError
from app.core.logging import logger, request_id_var, setup_logging, trace_id_var
from app.core.metrics import metrics_response
from app.db import milvus, postgres, redis
from app.services.rag.retriever import hybrid_retriever


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging(json_logs=not settings.debug)
    logger.info("starting {} v{}", settings.app_name, settings.app_version)

    # LangSmith 链路追踪（可选）
    if settings.langsmith_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        if settings.langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        logger.info("langsmith tracing enabled")

    # Milvus Collection 幂等创建（失败则进入降级模式，不影响启动）
    try:
        milvus.ensure_collection()
        logger.info("milvus collection '{}' ready", settings.milvus_collection)
    except Exception as exc:
        logger.warning("milvus unavailable (degraded mode): {}", exc)

    # 加载 FAQ 语料构建 BM25 索引
    try:
        n = await hybrid_retriever.load()
        logger.info("faq corpus loaded: {} docs", n)
    except Exception as exc:
        logger.warning("corpus load failed: {}", exc)

    # 预热本地模型（嵌入 + 重排），避免首个请求内加载导致超时
    try:
        import torch

        torch.set_num_threads(8)
        from app.services import embeddings
        from app.services.rag.reranker import reranker

        embeddings.preload()
        reranker.preload()
        logger.info("local models preloaded")
    except Exception as exc:
        logger.warning("model preload failed (will lazy-load on first request): {}", exc)

    yield

    await postgres.close_pool()
    await redis.close_redis()
    logger.info("shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = uuid.uuid4().hex[:16]
    request_id_var.set(rid)
    trace_id_var.set(rid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.info("{} {} -> 500 ({:.0f}ms)", request.method, request.url.path, (time.perf_counter() - t0) * 1000)
        raise
    response.headers["X-Request-ID"] = rid
    logger.info(
        "{} {} -> {} ({:.0f}ms)",
        request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000,
    )
    return response


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    logger.warning("api error code={} msg={}", exc.code, exc.message)
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error: {}", exc)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "服务器内部错误", "data": None},
    )


app.include_router(chat_router, prefix="/api/v1")
app.include_router(health_router)


@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    return metrics_response()
