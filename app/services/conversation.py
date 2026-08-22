"""多轮会话管理：Redis 存储，Key=session:{session_id}，Value=最近 N 轮对话 JSON。"""
import json

from app.config import settings
from app.db.redis import get_redis


def _key(session_id: str) -> str:
    return f"session:{session_id}"


async def get_history(session_id: str) -> list[dict]:
    """取最近 session_max_turns 轮对话（[{role, content}]）。"""
    try:
        r = get_redis()
        raw = await r.get(_key(session_id))
        if not raw:
            return []
        turns = json.loads(raw)
        return turns[-settings.session_max_turns:]
    except Exception:
        return []


async def add_turn(session_id: str, role: str, content: str) -> list[dict]:
    """追加一轮对话并刷新 TTL，返回完整历史。"""
    r = get_redis()
    key = _key(session_id)
    try:
        raw = await r.get(key)
        turns = json.loads(raw) if raw else []
    except Exception:
        turns = []
    turns.append({"role": role, "content": content})
    turns = turns[-settings.session_max_turns:]
    try:
        await r.set(
            key, json.dumps(turns, ensure_ascii=False), ex=settings.session_ttl_seconds
        )
    except Exception:
        pass
    return turns


async def clear(session_id: str) -> None:
    try:
        await get_redis().delete(_key(session_id))
    except Exception:
        pass
