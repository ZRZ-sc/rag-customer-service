"""健康检查与就绪探针。"""
from fastapi import APIRouter

from app.config import settings
from app.db import milvus, postgres, redis
from app.services.llm import breaker as llm_breaker

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": settings.app_version, "service": settings.app_name}


@router.get("/ready")
async def ready() -> dict:
    # llm 三种状态：ready / not_configured / circuit_open（熔断打开时会直接返回兜底话术，运维需可见）
    if llm_breaker.is_open:
        llm_state = "circuit_open"
    elif settings.llm_ready:
        llm_state = "ready"
    else:
        llm_state = "not_configured"
    components = {
        "postgres": "up" if await postgres.ping() else "down",
        "redis": "up" if await redis.ping() else "down",
        "milvus": "up" if await milvus.ping() else "down",
        "llm": llm_state,
    }
    overall = "ok" if components["postgres"] == "up" and components["milvus"] == "up" else "degraded"
    return {"status": overall, "version": settings.app_version, "components": components}
