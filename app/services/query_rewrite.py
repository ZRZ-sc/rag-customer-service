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

from app.config import settings
from app.core.logging import logger
from app.core.masking import mask_text
from app.core.metrics import context_rewrite_total, context_rewrite_duration
from app.services import cache as cache_svc
from app.services.intent import (
    ORDER_ID_RE,
    TRACKING_RE,
    detect_entities,
    has_knowledge_marker,
    has_status_intent,
)
from app.services.llm import get_llm

# 指代/省略标记：句首"那"、"它"、"这个/那个"、"该订单"、"上面/刚才"、"还有/另外"等
_PRONOUN_RE = re.compile(
    r"(^那|它|这个|那个|这单|那单|该订单|该单|此单|上述|上面|刚才|之前|^还有|^另外|^再)"
)
# 极简追问："呢？""什么时候？""多久？""为什么？""还要多久？"
# 注意：不要把"怎么/如何"列入——"怎么开具发票？"是完整独立问题，而非省略主语的追问
_SHORT_FOLLOWUP_RE = re.compile(
    r"^(呢|那呢|什么时候|多久|还要|要多久|多少钱|为什么|几点|多长|几天)"
)

# 护栏版本号：参与改写缓存键，护栏逻辑变更时自动失效历史缓存
_GUARD_VERSION = "g2"

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


def is_standalone(question: str) -> bool:
    """是否为独立问题（不含代词/省略标记）。

    用途：独立问题的"改写后问题"就是原问题本身，因此可以直接用原问题做答案缓存探测，
    命中时无需读取会话历史/实体记忆，也无需进入改写流程（L1 命中延迟降到个位数毫秒）。
    """
    q = (question or "").strip()
    if not q:
        return False
    if _PRONOUN_RE.search(q):
        return False
    # 只有"无实义内容"的极简追问才跳过 L1 直接探测（"可以开发票吗？"属完整问题）
    if len(q) <= 8 and _SHORT_FOLLOWUP_RE.match(q):
        return False
    # 在问订单/物流状态但没带单号 → 依赖上下文，不能走 L1 快速探测
    if has_status_intent(q) and not (
        detect_entities(q)["order_ids"] or detect_entities(q)["tracking_nos"]
    ):
        return False
    return True


def needs_rewrite(question: str, history: list[dict], last_entities: dict) -> bool:
    """判断是否需要结合上下文改写（避免对独立问题做无谓的 LLM 调用）。

    注意（压测中发现的缺陷）：不能仅凭"以 吗/呢 结尾且较短"就判定为追问——
    "可以指定快递公司吗？"是完整问题，一旦被当成追问，LLM 可能把同一 session 里
    **其他用户**的订单号注入改写结果，导致知识问答被误路由为订单查询（403）。
    """
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
    # 极简追问：无实义内容的时间/原因类短问句（"呢？""多久？""什么时候？"）
    if len(q) <= 8 and _SHORT_FOLLOWUP_RE.match(q):
        return True
    # 省略主语的追问：疑似在问订单/物流状态却未带单号，而会话中存在实体记忆
    has_memory = bool(last_entities.get("order_ids") or last_entities.get("tracking_nos"))
    if has_memory and has_status_intent(q):
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
    """剥离 LLM 误注入的订单号/物流单号（非状态类问题不该带实体）。

    例："订单 ORD20240100002 的退货运费由谁承担？" → "退货运费由谁承担？"
    """
    cleaned = ORDER_ID_RE.sub("", text)
    cleaned = TRACKING_RE.sub("", cleaned)
    cleaned = re.sub(r"的?\s*[（(]\s*[)）]", "", cleaned)          # 去掉空括号
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    # 去掉残留的前缀标签与"的"："订单  的退货运费…" → "退货运费…"
    cleaned = re.sub(
        r"^[\s，:：]*((订单号?|物流单号|运单号?|单号|本单|此单|该单)\s*)*的?\s*", "", cleaned
    )
    cleaned = cleaned.strip(" ，,。:：")
    return cleaned or text


def _apply_guards(question: str, rewritten: str, method: str) -> tuple[str, str]:
    """对改写结果应用两道实体护栏（写入侧与缓存读取侧都会调用）。

    护栏1（政策类）：原问题是规则/政策问法时，剥离注入的实体 ——
        否则"那退货运费谁出？"可能被改写为带订单号的句子，被正则误判为精确查询。
    护栏2（非状态类）：原问题不是在问订单/物流状态时，一律不得携带实体 ——
        压测中发现 LLM 会把同一 session 中其他用户的订单号注入 FAQ 问题，
        导致知识问答被误路由为订单查询并触发越权 403。
    """
    if has_knowledge_marker(question):
        stripped = _strip_injected_entities(rewritten)
        if stripped != rewritten:
            logger.info("query rewrite guard(policy) | {!r} -> {!r}", mask_text(rewritten), mask_text(stripped))
            rewritten = stripped
            method = method if method.endswith("policy_guard") else f"{method}+policy_guard"

    orig_ents = detect_entities(question)
    if not (orig_ents["order_ids"] or orig_ents["tracking_nos"]) and not has_status_intent(question):
        stripped = _strip_injected_entities(rewritten)
        if stripped != rewritten:
            logger.info("query rewrite guard(no-status-intent) | {!r} -> {!r}", mask_text(rewritten), mask_text(stripped))
            rewritten = stripped
            method = method if method.endswith("entity_guard") else f"{method}+entity_guard"
    return rewritten, method


def _context_fingerprint(history: list[dict], last_entities: dict) -> str:
    """上下文指纹：会话实体 + 上一轮用户问题，用于保证改写缓存不跨上下文串味。"""
    last_user = ""
    for turn in reversed(history):
        if turn.get("role") == "user":
            last_user = (turn.get("content") or "").strip()[:60]
            break
    orders = ",".join(last_entities.get("order_ids") or [])
    trackings = ",".join(last_entities.get("tracking_nos") or [])
    return f"{orders}|{trackings}|{last_user}"


async def rewrite_query(
    question: str,
    history: list[dict] | None = None,
    last_entities: dict | None = None,
) -> dict:
    """上下文改写入口（结果缓存 24h，避免重复的 LLM 改写调用）。

    返回：{"original": str, "rewritten": str, "applied": bool,
           "method": "none|llm|entity_memory|llm+entity_memory|llm+policy_guard", "cached": bool}
    """
    history = history or []
    last_entities = last_entities or {"order_ids": [], "tracking_nos": []}
    result = {
        "original": question,
        "rewritten": question,
        "applied": False,
        "method": "none",
        "cached": False,
    }

    if not needs_rewrite(question, history, last_entities):
        context_rewrite_total.labels(applied="false", method="none").inc()
        return result

    cache_key = cache_svc.rewrite_cache_key(
        question, f"{_context_fingerprint(history, last_entities)}|{_GUARD_VERSION}"
    )
    cached = await cache_svc.get_json(cache_key, name="query_rewrite")
    if cached and cached.get("rewritten"):
        # 护栏在读取侧同样生效：避免历史缓存（护栏上线前写入）绕过实体剥离
        safe, method = _apply_guards(question, cached["rewritten"], cached.get("method", "llm"))
        if safe != cached["rewritten"]:
            logger.info(
                "query rewrite cache HIT but guarded | {!r} -> {!r}", cached["rewritten"], safe
            )
            await cache_svc.set_json(
                cache_key, {"rewritten": safe, "method": method},
                ttl=settings.rewrite_cache_ttl_seconds, name="query_rewrite",
            )
        context_rewrite_total.labels(applied="true", method=f"{method}+cache").inc()
        result.update(rewritten=safe, applied=True, method=method, cached=True)
        logger.info("query rewrite cache HIT | original={!r} -> rewritten={!r}", mask_text(question), mask_text(safe))
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

    # ---- 护栏（读取侧与写入侧共用同一实现）----
    rewritten, method = _apply_guards(question, rewritten, method)

    # 二次兜底：LLM 改写后仍无实体，但问题确实在问订单/物流状态 → 用记忆补全
    if method.startswith("llm") and "guard" not in method and has_status_intent(question):
        ents = detect_entities(rewritten)
        if not ents["order_ids"] and not ents["tracking_nos"]:
            mem = _rewrite_with_memory(rewritten, last_entities)
            if mem:
                rewritten = mem
                method = "llm+entity_memory"

    context_rewrite_duration.observe(time.perf_counter() - t0)
    context_rewrite_total.labels(applied="true", method=method).inc()
    result.update(rewritten=rewritten, applied=True, method=method)

    # 写入缓存（TTL 24h）：method=cache 的命中不再触发 LLM
    if method not in ("entity_memory",):  # 实体记忆兜底本身零成本，无需缓存
        await cache_svc.set_json(
            cache_key,
            {"rewritten": rewritten, "method": method},
            ttl=settings.rewrite_cache_ttl_seconds,
            name="query_rewrite",
        )

    logger.info(
        "query rewritten | original={!r} -> rewritten={!r} method={}",
        mask_text(question), mask_text(rewritten), method,
    )
    return result
