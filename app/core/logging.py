"""结构化日志：Loguru + JSON 序列化，contextvars 注入 request_id 等全链路字段。"""
import sys
from contextvars import ContextVar

from loguru import logger

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")
user_id_var: ContextVar[str] = ContextVar("user_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


def _context_filter(record) -> bool:
    record["extra"]["request_id"] = request_id_var.get()
    record["extra"]["trace_id"] = trace_id_var.get()
    record["extra"]["user_id"] = user_id_var.get()
    record["extra"]["session_id"] = session_id_var.get()
    return True


def setup_logging(json_logs: bool = True, level: str = "INFO") -> None:
    logger.remove()
    if json_logs:
        # loguru 内置 JSON 序列化（serialize=True），extra 字段自动包含 request_id 等
        logger.add(
            sys.stdout,
            filter=_context_filter,
            serialize=True,
            level=level,
            enqueue=True,
        )
    else:
        logger.add(
            sys.stdout,
            filter=_context_filter,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | "
                "req={extra[request_id]} | user={extra[user_id]} | {message}"
            ),
            level=level,
        )
    logger.debug("logging initialized")


def get_logger():
    return logger
