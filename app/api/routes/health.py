"""健康检查与就绪探针。"""
from fastapi import APIRouter

from app.config import settings
from app.db import milvus, postgres, redis

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": settings.app_version, "service": settings.app_name}


@router.get("/ready")
async def ready() -> dict:
    components = {
        "postgres": "up" if await postgres.ping() else "down",
        "redis": "up" if await redis.ping() else "down",
        "milvus": "up" if await milvus.ping() else "down",
        "llm": "ready" if settings.llm_ready else "not_configured",
    }
    overall = "ok" if components["postgres"] == "up" and components["milvus"] == "up" else "degraded"
    return {"status": overall, "version": settings.app_version, "components": components}
