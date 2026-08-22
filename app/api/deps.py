"""API 依赖：Bearer Token 认证。"""
from fastapi import Header

from app.config import settings
from app.core.errors import UnauthorizedError


async def verify_token(authorization: str = Header(default="")) -> None:
    """校验 Authorization: Bearer <token>。"""
    if not authorization.startswith("Bearer "):
        raise UnauthorizedError()
    token = authorization[len("Bearer "):].strip()
    if token != settings.api_token:
        raise UnauthorizedError()
