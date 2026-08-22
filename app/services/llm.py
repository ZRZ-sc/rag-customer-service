"""LLM 客户端：DeepSeek API（OpenAI 兼容接口，经 LangChain ChatOpenAI 封装）。"""
import threading

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.core.errors import DegradedError

_lock = threading.Lock()
_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """懒加载 DeepSeek ChatOpenAI 实例。"""
    global _llm
    if _llm is None:
        if not settings.llm_ready:
            raise DegradedError("LLM 服务未配置，请设置 DEEPSEEK_API_KEY")
        with _lock:
            if _llm is None:
                _llm = ChatOpenAI(
                    model=settings.deepseek_model,
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                    temperature=settings.llm_temperature,
                    timeout=settings.llm_timeout,
                    max_tokens=settings.llm_max_tokens,
                    max_retries=1,
                )
    return _llm


def reset_llm() -> None:
    global _llm
    _llm = None


def extract_usage(msg: AIMessage) -> dict:
    """从 AIMessage 中提取 token 用量（DeepSeek 返回 usage 元数据）。"""
    um = getattr(msg, "usage_metadata", None) or {}
    return {
        "prompt_tokens": um.get("input_tokens", 0),
        "completion_tokens": um.get("output_tokens", 0),
        "total_tokens": um.get("total_tokens", 0),
    }
