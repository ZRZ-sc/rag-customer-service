"""LLM 客户端：DeepSeek API（OpenAI 兼容接口，经 LangChain ChatOpenAI 封装）。

除客户端本身外，这里还提供**调用侧熔断**（`ainvoke`）：

- 背景：DeepSeek 侧超时/不可达时，OpenAI SDK 自身会重试并指数退避，实测一次失败调用
  可占用 12~55 秒；若并发请求都卡在失败重试上，会拖慢整个接口（需求文档 N8：依赖故障
  不得让系统整体不可用，流程6：LLM 不可用要快速给固定话术 + 转人工）。
- 做法：连续失败达到阈值后**打开熔断**，冷却期内所有调用立即失败（毫秒级），
  由上层各自的降级分支处理：意图识别→默认知识类；改写→实体记忆兜底；扩展→原问题；
  生成→固定话术 + 自动转人工。冷却期后放行一次探测，成功即恢复。
"""
import threading
import time

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.core.errors import DegradedError
from app.core.logging import logger
from app.core.metrics import llm_failure_total

_lock = threading.Lock()
_llm: ChatOpenAI | None = None


class LlmCircuitOpen(RuntimeError):
    """熔断打开：直接失败，不再等待上游超时/重试。"""


class LlmCircuitBreaker:
    """极简连续失败熔断器（进程内，线程安全）。

    - threshold: 连续失败多少次后打开
    - cooldown:  打开后多少秒内直接短路；到期放行一次"探测"请求
    - hard 失败（超时/连接错误，单次代价最高）**一次即打开**，不等累计到阈值
    """

    def __init__(self, threshold: int, cooldown: float):
        self.threshold = max(1, threshold)
        self.cooldown = max(0.0, cooldown)
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at == 0.0:
                return False
            return (time.monotonic() - self._opened_at) < self.cooldown

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def allow(self) -> bool:
        """是否放行本次调用（冷却期结束后放行一次探测）。"""
        return not self.is_open

    def record_success(self) -> None:
        with self._lock:
            if self._failures or self._opened_at:
                logger.info("llm circuit closed (recovered) after {} failures", self._failures)
            self._failures = 0
            self._opened_at = 0.0

    def record_failure(self, hard: bool = False) -> None:
        with self._lock:
            self._failures += 1
            reached = self._failures >= self.threshold or hard
            if reached and self._opened_at == 0.0:
                self._opened_at = time.monotonic()
                logger.warning(
                    "llm circuit OPEN after {} consecutive failures (hard={}, cooldown {}s)",
                    self._failures, hard, self.cooldown,
                )

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0


# 超时/连接类异常：单次就足以说明上游不可达，直接打开熔断（不必等累计到阈值）
_HARD_FAILURE_NAMES = {
    "APITimeoutError", "APIConnectionError", "APIConnectionTimeoutError",
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
    "PoolTimeout", "TimeoutError", "ConnectionError", "ConnectionResetError", "OSError",
}


def _is_hard_failure(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return type(exc).__name__ in _HARD_FAILURE_NAMES


breaker = LlmCircuitBreaker(
    threshold=settings.llm_circuit_threshold, cooldown=settings.llm_circuit_cooldown
)


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
                    max_retries=settings.llm_max_retries,
                )
    return _llm


async def ainvoke(messages: list, stage: str = "unknown"):
    """带熔断的 LLM 调用（所有调用点都应走这里，便于统一降级与观测）。

    失败时**原样抛出异常**（调用方各自的降级逻辑保持不变，只是不会再等几十秒）。
    """
    if not breaker.allow():
        llm_failure_total.labels(stage=f"{stage}:circuit_open").inc()
        raise LlmCircuitOpen(
            f"LLM 熔断中（连续失败 {breaker.failures} 次），{breaker.cooldown:.0f}s 后重试"
        )
    try:
        resp = await get_llm().ainvoke(messages)
    except Exception as exc:
        breaker.record_failure(hard=_is_hard_failure(exc))
        llm_failure_total.labels(stage=stage).inc()
        raise
    breaker.record_success()
    return resp


def reset_llm() -> None:
    global _llm
    _llm = None
    breaker.reset()


def extract_usage(msg: AIMessage) -> dict:
    """从 AIMessage 中提取 token 用量（DeepSeek 返回 usage 元数据）。"""
    um = getattr(msg, "usage_metadata", None) or {}
    return {
        "prompt_tokens": um.get("input_tokens", 0),
        "completion_tokens": um.get("output_tokens", 0),
        "total_tokens": um.get("total_tokens", 0),
    }
