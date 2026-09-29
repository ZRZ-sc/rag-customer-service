"""FastAPI 应用入口。"""
import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.support import router as support_router
from app.api.routes.admin import router as admin_router
from app.config import settings
from app.core.errors import ApiError
from app.core.logging import logger, request_id_var, setup_logging, trace_id_var
from app.core.masking import mask_text
from app.core.metrics import metrics_response
from app.db import milvus, postgres, redis
from app.services.rag.retriever import hybrid_retriever

# 后台任务引用集合（防止 asyncio 任务被 GC 回收）
_background_tasks: set[asyncio.Task] = set()


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
    # 可通过 PRELOAD_MODELS=false 关闭（低内存部署、或纯缓存命中场景无需常驻模型）
    if settings.preload_models:
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
    else:
        logger.info("model preload disabled (lazy load on first RAG request)")

    # 缓存预热（流程3）：启动后延迟自调用 /api/v1/chat，把高频问题预先写入各级缓存。
    # 默认关闭（会产生 LLM 调用费用），用 CACHE_WARMUP_ON_STARTUP=true 打开。
    if settings.cache_warmup_on_startup:
        async def _warmup_after_startup() -> None:
            await asyncio.sleep(settings.cache_warmup_delay)
            try:
                from app.services.cache_warmup import load_warmup_questions, warm_cache

                questions = load_warmup_questions(limit=settings.cache_warmup_limit)
                stats = await warm_cache(
                    questions, base_url=settings.cache_warmup_base_url, concurrency=2
                )
                logger.info(
                    "startup cache warmup | ok={} failed={} cached {}->{}/{} elapsed={}ms",
                    stats["ok"], stats["failed"], stats["before"]["warmed"],
                    stats["after"]["warmed"], stats["total"], stats["elapsed_ms"],
                )
            except Exception as exc:
                logger.warning("startup cache warmup failed: {}", exc)

        task = asyncio.create_task(_warmup_after_startup())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        logger.info(
            "startup cache warmup scheduled (limit={} base={})",
            settings.cache_warmup_limit, settings.cache_warmup_base_url,
        )

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
    """访问日志：JSON 结构化字段（service / duration_ms / status 等），并做敏感信息脱敏。"""
    rid = uuid.uuid4().hex[:16]
    request_id_var.set(rid)
    trace_id_var.set(rid)
    t0 = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        path = request.url.path
        query = request.url.query
        # 结构化字段：loguru serialize=True 会把 bind 的字段并入 JSON 的 extra
        log = logger.bind(
            event="http_access",
            service=settings.app_name,
            method=request.method,
            path=path,
            status=status_code,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            client=request.client.host if request.client else "-",
            query=mask_text(query) if query else "",
            user_agent=(request.headers.get("user-agent") or "")[:80],
        )
        # 探针/指标抓取避免刷屏
        if path in ("/metrics", "/health", "/ready"):
            log.debug("http_access")
        else:
            log.info("http_access")


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    logger.warning("api error code={} msg={}", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.http_status, content=exc.to_dict(), headers=exc.headers or None
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error: {}", exc)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "服务器内部错误", "data": None},
    )


app.include_router(chat_router, prefix="/api/v1")
app.include_router(support_router)          # /api/v1/handoff、/api/v1/feedback（自带前缀）
app.include_router(admin_router)            # /api/v1/admin/*（管理后台，ADMIN_TOKEN 鉴权）
app.include_router(health_router)

# ---- 静态资源：极简 Web 聊天界面（F1）与管理后台页面（F7） ----
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index_page():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/admin", include_in_schema=False)
    async def admin_page():
        return FileResponse(_STATIC_DIR / "admin.html")


@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    return metrics_response()
