"""多轮会话管理：Redis 存储对话历史 + 会话实体记忆。

- 历史：Key=session:{session_id}，Value=最近 N 轮对话 JSON
- 实体记忆：Key=session:{session_id}:entities，记录该会话中最近提到的订单号/物流单号，
  用于多轮追问时的指代消解兜底（LLM 不可用时也能把"那什么时候能到？"关联到具体订单）。
"""
import json

from app.config import settings
from app.db.redis import get_redis


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def _entity_key(session_id: str) -> str:
    return f"session:{session_id}:entities"


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


# ---- 会话实体记忆（指代消解的关键状态） ----


async def get_last_entities(session_id: str) -> dict:
    """取会话中最近提到的实体：{"order_ids": [...], "tracking_nos": [...]}。"""
    try:
        raw = await get_redis().get(_entity_key(session_id))
        if not raw:
            return {"order_ids": [], "tracking_nos": []}
        data = json.loads(raw)
        return {
            "order_ids": data.get("order_ids", []),
            "tracking_nos": data.get("tracking_nos", []),
        }
    except Exception:
        return {"order_ids": [], "tracking_nos": []}


async def set_last_entities(session_id: str, entities: dict) -> None:
    """记录/合并会话实体（与对话历史同 TTL）。"""
    order_ids = [e for e in (entities.get("order_ids") or []) if e]
    tracking_nos = [e for e in (entities.get("tracking_nos") or []) if e]
    if not order_ids and not tracking_nos:
        return
    try:
        current = await get_last_entities(session_id)
        merged = {
            "order_ids": list(dict.fromkeys(order_ids + current["order_ids"]))[:5],
            "tracking_nos": list(dict.fromkeys(tracking_nos + current["tracking_nos"]))[:5],
        }
        await get_redis().set(
            _entity_key(session_id),
            json.dumps(merged, ensure_ascii=False),
            ex=settings.session_ttl_seconds,
        )
    except Exception:
        pass


async def clear(session_id: str) -> None:
    try:
        await get_redis().delete(_key(session_id), _entity_key(session_id))
    except Exception:
        pass
