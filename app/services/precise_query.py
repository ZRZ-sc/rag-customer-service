"""精确查询服务：订单/物流信息从 PostgreSQL 实时查询（含归属校验）。"""
from app.core.errors import ForbiddenError, NotFoundError
from app.db.postgres import get_pool

STATUS_TEXT = {
    "pending": "待处理",
    "paid": "已支付",
    "shipped": "已发货",
    "delivered": "已签收",
    "refunded": "已退款",
    "cancelled": "已取消",
}

_ORDER_SQL = """
SELECT o.order_id, o.user_id, o.status, o.tracking_no, o.quantity, o.created_at, o.updated_at,
       p.name AS product_name, p.price,
       u.phone
FROM orders o
JOIN products p ON p.product_id = o.product_id
JOIN users u ON u.user_id = o.user_id
WHERE o.order_id = $1
"""

_TRACKING_SQL = """
SELECT o.order_id, o.user_id, o.status, o.tracking_no, o.quantity, o.created_at, o.updated_at,
       p.name AS product_name, p.price,
       u.phone
FROM orders o
JOIN products p ON p.product_id = o.product_id
JOIN users u ON u.user_id = o.user_id
WHERE o.tracking_no = $1
"""


def _format_order_answer(row: dict) -> str:
    status_text = STATUS_TEXT.get(row["status"], row["status"])
    parts = [f"您的订单 {row['order_id']} 当前状态：{status_text}"]
    if row["product_name"]:
        parts.append(f"商品：{row['product_name']} ×{row['quantity']}")
    if row["tracking_no"]:
        parts.append(f"物流单号：{row['tracking_no']}")
    if row["status"] == "shipped":
        parts.append("商品已发出，正在运输中，请留意物流更新。")
    elif row["status"] == "delivered":
        parts.append("商品已签收，感谢您的购买！")
    elif row["status"] == "refunded":
        parts.append("退款已完成，款项将原路退回，请注意查收。")
    return "。".join(parts)


async def query_by_order_id(order_id: str, user_id: str) -> dict:
    pool = await get_pool()
    row = await pool.fetchrow(_ORDER_SQL, order_id)
    if row is None:
        raise NotFoundError(f"未查询到订单 {order_id} 的信息")
    if row["user_id"] != user_id:
        raise ForbiddenError("无权限查询该订单（订单不属于当前用户）")
    return {
        "answer": _format_order_answer(dict(row)),
        "confidence": 1.0,
        "detail": {
            "order_id": row["order_id"],
            "status": row["status"],
            "tracking_no": row["tracking_no"],
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        },
    }


async def query_by_tracking_no(tracking_no: str, user_id: str) -> dict:
    pool = await get_pool()
    row = await pool.fetchrow(_TRACKING_SQL, tracking_no)
    if row is None:
        raise NotFoundError(f"未查询到物流单号 {tracking_no} 的信息")
    if row["user_id"] != user_id:
        raise ForbiddenError("无权限查询该物流信息（不属于当前用户）")
    return {
        "answer": _format_order_answer(dict(row)),
        "confidence": 1.0,
        "detail": {
            "order_id": row["order_id"],
            "status": row["status"],
            "tracking_no": row["tracking_no"],
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        },
    }
