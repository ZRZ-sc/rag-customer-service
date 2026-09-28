"""上下文感知的查询改写（多轮指代消解）。

在**意图识别与路由之前**执行：若当前问题包含代词/指示词（"那""它""这个"…）或省略了
核心实体（如订单号），则结合最近几轮对话改写成一个独立完整的问题，再送入意图识别。

示例：
    第1轮：我的订单 ORD20240100001 到哪里了？      （source=database）
    第2轮：那什么时候能到？  →  订单 ORD20240100001 什么时候能到？  （仍走 database）

三级策略（保证 LLM 不可用时也能完成消解，符合需求文档 N8 降级要求）：
1. 无需改写：首轮 / 问题自带实体 / 无上下文依赖标记 → 原样返回（零额外延迟、零 LLM 调用）
2. LLM 改写：结合最近 6 轮对话生成独立问题
3. 实体记忆兜底：LLM 不可用或未产出时，用会话中最近提到的订单号/物流单号补全问题
"""
import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import logger
from app.core.metrics import context_rewrite_total, context_rewrite_duration
from app.services.intent import (
    ORDER_ID_RE,
    TRACKING_RE,
    detect_entities,
    has_knowledge_marker,
    has_precise_signal,
)
from app.services.llm import get_llm

# 指代/省略标记：句首"那"、"它"、"这个/那个"、"该订单"、"上面/刚才"、"还有/另外"等
_PRONOUN_RE = re.compile(
    r"(^那|它|这个|那个|这单|那单|该订单|该单|此单|上述|上面|刚才|之前|^还有|^另外|^再)"
)
# 极简追问："呢？""多久？""为什么？""还要多久？"
_SHORT_FOLLOWUP_RE = re.compile(
    r"^(呢|那呢|什么时候|多久|还要|要多久|多少钱|为什么|怎么|如何|能不能|可以吗)"
)

_REWRITE_SYSTEM_PROMPT = """你是多轮对话的查询改写助手。任务：把用户当前问题改写为**不依赖上下文即可独立理解**的问题。

规则：
1. 补全省略的主语与指代（如"那""它""这个""上面的"），从对话历史中找出用户实际指代的对象；
2. 如果当前问题是在询问某个订单/物流的**状态**（时效、进度、发货、签收、退款到账等），
   请把历史对话中出现的订单号或物流单号一并写入改写结果；
3. 如果当前问题是在询问**平台规则/政策**（退换货政策、运费规则、优惠券、发票等），
   **绝对不要**引入订单号或物流单号，只需把问题本身补全为完整问句；
4. 不要回答问题、不要解释，只输出改写后的问题；
5. 若当前问题本身已经完整独立，原样返回。

只输出 JSON：{"rewritten": "改写后的问题", "used_context": true}"""


def needs_rewrite(question: str, history: list[dict], last_entities: dict) -> bool:
    """判断是否需要结合上下文改写（避免对独立问题做无谓的 LLM 调用）。"""
    if not history:
        return False
    q = (question or "").strip()
    if not q:
        return False
    entities = detect_entities(q)
    # 已自带订单号/物流单号 → 独立完整，无需改写
    if entities["order_ids"] or entities["tracking_nos"]:
        return False
    if _PRONOUN_RE.search(q):
        return True
    if len(q) <= 12 and _SHORT_FOLLOWUP_RE.match(q):
        return True
    if len(q) <= 12 and (q.endswith("呢") or q.endswith("呢？") or q.endswith("吗") or q.endswith("吗？")):
        return True
    # 省略主语的追问：疑似在问订单/物流状态却未带单号，而会话中存在实体记忆
    has_memory = bool(last_entities.get("order_ids") or last_entities.get("tracking_nos"))
    if has_memory and has_precise_signal(q, entities):
        return True
    return False


def _parse_rewrite(content: str) -> str | None:
    """从 LLM 输出中解析改写结果（容忍 ```json 包裹与纯文本输出）。"""
    text = (content or "").strip()
    if not text:
        return None
    cleaned = text.strip("`").strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
        rewritten = str(data.get("rewritten", "")).strip()
        return rewritten or None
    except Exception:
        # 非 JSON：取第一行非空文本作为改写结果
        for line in text.splitlines():
            line = line.strip().strip('"').strip()
            if line and not line.startswith("{"):
                return line
        return None


def _rewrite_with_memory(question: str, last_entities: dict) -> str | None:
    """确定性兜底：用会话实体记忆补全问题（不依赖 LLM）。"""
    if last_entities.get("order_ids"):
        return f"订单 {last_entities['order_ids'][0]}：{question}"
    if last_entities.get("tracking_nos"):
        return f"物流单号 {last_entities['tracking_nos'][0]}：{question}"
    return None


def _strip_injected_entities(text: str) -> str:
    """剥离 LLM 误注入的订单号/物流单号（政策类问题不该带实体）。

    例："订单 ORD20240100002 的退货运费由谁承担？" → "退货运费由谁承担？"
    """
    cleaned = ORDER_ID_RE.sub("", text)
    cleaned = TRACKING_RE.sub("", cleaned)
    cleaned = re.sub(r"的?\s*[（(]\s*[)）]", "", cleaned)          # 去掉空括号
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    # 去掉残留的前缀标签与"的"："订单  的退货运费…" → "退货运费…"
    cleaned = re.sub(
        r"^[\s，,：:]*((订单号?|物流单号|运单号?|单号|本单|此单|该单)\s*)*的?\s*", "", cleaned
    )
    cleaned = cleaned.strip(" ，,。:：")
    return cleaned or text


async def rewrite_query(
    question: str,
    history: list[dict] | None = None,
    last_entities: dict | None = None,
) -> dict:
    """上下文改写入口。

    返回：{"original": str, "rewritten": str, "applied": bool, "method": "none|llm|entity_memory|llm+entity_memory"}
    """
    history = history or []
    last_entities = last_entities or {"order_ids": [], "tracking_nos": []}
    result = {"original": question, "rewritten": question, "applied": False, "method": "none"}

    if not needs_rewrite(question, history, last_entities):
        context_rewrite_total.labels(applied="false", method="none").inc()
        return result

    t0 = time.perf_counter()
    rewritten: str | None = None
    method = "none"
    try:
        hist_text = "\n".join(
            f"{'用户' if t.get('role') == 'user' else '客服'}：{t.get('content', '')}"
            for t in history[-6:]
        )
        resp = await get_llm().ainvoke(
            [
                SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=f"对话历史：\n{hist_text}\n\n当前问题：{question}"),
            ]
        )
        rewritten = _parse_rewrite(resp.content)
        if rewritten:
            method = "llm"
    except Exception as exc:
        logger.warning("query rewrite via LLM failed, fallback to entity memory: {}", exc)

    if not rewritten:
        rewritten = _rewrite_with_memory(question, last_entities)
        if rewritten:
            method = "entity_memory"

    if not rewritten:
        context_rewrite_total.labels(applied="false", method="failed").inc()
        context_rewrite_duration.observe(time.perf_counter() - t0)
        return result

    # ---- 护栏：原始问题是政策/规则类时，剥离注入的实体 ----
    # 否则"那退货运费谁出？"可能被改写成带订单号的句子，进而被正则误判为精确查询，
    # 最终用订单状态回答了一个规则类问题（实测踩过）。
    if has_knowledge_marker(question):
        stripped = _strip_injected_entities(rewritten)
        if stripped != rewritten:
            logger.info(
                "query rewrite guard(policy) | {!r} -> {!r}", rewritten, stripped
            )
            rewritten = stripped
            if method == "llm":
                method = "llm+policy_guard"

    # 二次兜底：LLM 改写后仍无实体，但问题明显指向订单/物流状态 → 用记忆补全
    if method in ("llm", "llm+policy_guard"):
        ents = detect_entities(rewritten)
        if (
            not ents["order_ids"]
            and not ents["tracking_nos"]
            and has_precise_signal(rewritten, ents)
        ):
            mem = _rewrite_with_memory(rewritten, last_entities)
            if mem:
                rewritten = mem
                method = "llm+entity_memory"

    context_rewrite_duration.observe(time.perf_counter() - t0)
    context_rewrite_total.labels(applied="true", method=method).inc()
    result.update(rewritten=rewritten, applied=True, method=method)
    logger.info(
        "query rewritten | original={!r} -> rewritten={!r} method={}", question, rewritten, method
    )
    return result
