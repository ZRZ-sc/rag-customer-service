"""初始化数据库：建表 + 种子数据（用户/商品/订单）。

用法：.venv\\Scripts\\python.exe scripts\\init_db.py
"""
import asyncio
from datetime import datetime, timedelta

import asyncpg

from app.config import settings

DDL = [
    """CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(50) PRIMARY KEY,
        username VARCHAR(50),
        phone VARCHAR(20),
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS products (
        product_id VARCHAR(20) PRIMARY KEY,
        name VARCHAR(200),
        category VARCHAR(50),
        price DECIMAL(10,2)
    )""",
    """CREATE TABLE IF NOT EXISTS orders (
        order_id VARCHAR(20) PRIMARY KEY,
        user_id VARCHAR(50) REFERENCES users(user_id),
        product_id VARCHAR(20) REFERENCES products(product_id),
        quantity INT DEFAULT 1,
        status VARCHAR(20),
        tracking_no VARCHAR(30),
        shipping_address VARCHAR(200),
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS faq_documents (
        id SERIAL PRIMARY KEY,
        question TEXT,
        answer TEXT,
        category VARCHAR(50),
        embedding VECTOR(512),
        version INT DEFAULT 1,
        status VARCHAR(20) DEFAULT 'published',
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS dialogue_logs (
        id SERIAL PRIMARY KEY,
        request_id VARCHAR(50),
        session_id VARCHAR(50),
        user_id VARCHAR(50),
        question TEXT,
        intent VARCHAR(30),
        retrieval_results JSON,
        answer TEXT,
        source VARCHAR(20),
        confidence FLOAT,
        latency_ms INT,
        tokens_used INT,
        feedback INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_tracking ON orders(tracking_no)",
    "CREATE INDEX IF NOT EXISTS idx_faq_status ON faq_documents(status)",
    "CREATE INDEX IF NOT EXISTS idx_logs_session ON dialogue_logs(session_id)",
]

USERS = [
    ("user_001", "张三", "138****1234"),
    ("user_002", "李四", "139****5678"),
    ("user_003", "王五", "137****9012"),
]

PRODUCTS = [
    ("P1001", "无线蓝牙耳机 Pro", "数码", 299.00),
    ("P1002", "智能手表 S2", "数码", 899.00),
    ("P1003", "20000mAh 便携充电宝", "数码", 129.00),
    ("P1004", "87键机械键盘", "电脑外设", 459.00),
    ("P1005", "27寸 4K 显示器", "电脑外设", 1999.00),
    ("P1006", "纯棉基础款T恤", "服饰", 59.90),
    ("P1007", "运动跑鞋", "服饰", 399.00),
]


def _orders() -> list[tuple]:
    """构造订单：order_id 形如 ORD+13位数字（与意图识别正则一致），物流单号 SF+12位。"""
    base = datetime(2024, 1, 10, 10, 0, 0)
    specs = [
        # (user, product, qty, status, tracking, days_ago)
        ("user_001", "P1001", 1, "delivered", "SF123456789012", 20),
        ("user_001", "P1004", 1, "shipped", "SF123456789034", 3),
        ("user_001", "P1003", 2, "refunded", None, 15),
        ("user_002", "P1005", 1, "shipped", "YT987654321098", 2),
        ("user_002", "P1002", 1, "paid", None, 0),
        ("user_002", "P1006", 3, "delivered", "YT987654321076", 25),
        ("user_003", "P1007", 1, "pending", None, 0),
        ("user_003", "P1001", 1, "cancelled", None, 10),
    ]
    out = []
    for i, (uid, pid, qty, status, tracking, days) in enumerate(specs, start=1):
        created = base + timedelta(days=days)
        updated = created + timedelta(hours=6)
        out.append((
            f"ORD202401{i:05d}", uid, pid, qty, status, tracking,
            "广东省深圳市南山区科技园路 1 号", created, updated,
        ))
    return out


async def main() -> None:
    conn = await asyncpg.connect(dsn=settings.pg_dsn)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        for stmt in DDL:
            await conn.execute(stmt)
        print("[ok] schema ready (pgvector extension ensured)")

        for uid, name, phone in USERS:
            await conn.execute(
                "INSERT INTO users (user_id, username, phone) VALUES ($1,$2,$3) ON CONFLICT (user_id) DO NOTHING",
                uid, name, phone,
            )
        for pid, name, cat, price in PRODUCTS:
            await conn.execute(
                "INSERT INTO products (product_id, name, category, price) VALUES ($1,$2,$3,$4) ON CONFLICT (product_id) DO NOTHING",
                pid, name, cat, price,
            )
        for o in _orders():
            await conn.execute(
                """INSERT INTO orders (order_id, user_id, product_id, quantity, status, tracking_no, shipping_address, created_at, updated_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT (order_id) DO NOTHING""",
                *o,
            )
        print("[ok] seeded users/products/orders")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
