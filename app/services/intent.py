"""意图识别与路由（需求文档流程1）：
第一层正则快路径 → 第二层关键词信号 → 第三层 LLM few-shot 兜底。
"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm import ainvoke as llm_ainvoke

# 与 scripts/init_db.py 种子数据格式保持一致
ORDER_ID_RE = re.compile(r"(?<![A-Za-z0-9])(ORD\d{8,20})(?![A-Za-z0-9])", re.IGNORECASE)
TRACKING_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:SF|YT|STO|ZTO|JD)\d{10,20})(?![A-Za-z0-9])", re.IGNORECASE
)

# 精确查询类问题的强信号词
PRECISE_KEYWORDS = [
    "订单", "物流", "快递", "发货", "签收", "派送", "运单",
    "到哪", "进度", "查询", "退款到", "退货退款状态",
]

# 政策/规则类标记：命中时**优先**判为知识类，即使问题里带了订单号。
# 例："订单 ORD123 的退货运费谁出？" —— 问的是运费规则，应走 FAQ 而不是查订单状态。
# 说明：本列表由 N4 评测（scripts/eval_accuracy.py）反向校验过——多轮追问
# "那还能换货吗？""那退款多久到账？" 曾因缺少「换货/多久到账」而被误路由到数据库。
KNOWLEDGE_MARKERS = (
    # 退换货与售后政策
    "运费", "换货", "售后", "维修", "保修",
    "怎么退", "如何退", "能不能退", "可以退吗", "能退吗", "退货政策",
    # 时效类（问的是"政策规定的时长"，不是"我这单现在到哪了"）
    "多久到账", "几天到账", "多久能到账", "退款多久",
    # 其他规则类
    "政策", "规则", "规定", "条件", "标准", "流程", "谁出", "谁承担", "谁付",
    "怎么申请", "如何申请", "发票", "优惠券", "会员", "积分",
    "售后政策", "支持吗", "条件是什么",
)

_INTENT_SYSTEM_PROMPT = """你是一个电商客服意图分类器。判断用户问题属于哪类：
- precise_query: 用户查询自己订单/物流/退款的具体状态（通常提到订单号、物流单号，或明显在询问某个订单现在怎么样了）
- knowledge: 询问平台政策、规则、FAQ 知识（退换货政策、运费规则、优惠券、发票、售后流程等）
- other: 闲聊、寒暄或与客服无关的内容

注意：询问"规则/政策/流程/运费归属"的问题属于 knowledge，即使提到了订单号。
只输出 JSON，格式：{"intent": "...", "reason": "一句话理由"}"""


def has_knowledge_marker(question: str) -> bool:
    """是否为政策/规则类问法。"""
    return any(kw in (question or "") for kw in KNOWLEDGE_MARKERS)


# "在问某个订单/物流当前状态"的强信号词（比 PRECISE_KEYWORDS 更严格）。
# 用途：决定"是否允许把会话中的订单号注入改写结果"，以及"问题是否依赖上下文"。
# 注意：不能把"快递""订单"这类主题名词算进来——"可以指定快递公司吗？"是规则问题，
# 若允许注入实体，就会被误路由成订单查询（压测中曾产生跨用户越权 403）。
STATUS_INTENT_KEYWORDS = (
    "到哪", "到货", "进度", "什么时候到", "什么时候能到", "几天到", "几号到",
    "状态", "签收", "派送", "发货了", "发出", "揽收", "停在哪", "在哪",
    "退款进度", "退款到账", "退款了", "运单号", "物流信息",
)


def has_status_intent(question: str) -> bool:
    """是否在询问某个订单/物流的当前状态（可携带上下文实体）。"""
    if has_knowledge_marker(question):
        return False
    return any(kw in (question or "") for kw in STATUS_INTENT_KEYWORDS)


def detect_entities(question: str) -> dict:
    """提取订单号与物流单号。"""
    order_ids = list(dict.fromkeys(m.group(1).upper() for m in ORDER_ID_RE.finditer(question)))
    tracking_nos = list(dict.fromkeys(m.group(1).upper() for m in TRACKING_RE.finditer(question)))
    return {"order_ids": order_ids, "tracking_nos": tracking_nos}


def has_precise_signal(question: str, entities: dict) -> bool:
    """是否存在"在问具体订单/物流状态"的信号（政策问法不算）。"""
    if has_knowledge_marker(question):
        return False
    if entities["order_ids"] or entities["tracking_nos"]:
        return True
    return any(kw in question for kw in PRECISE_KEYWORDS)


async def _classify_with_llm(question: str, history: list[dict]) -> str:
    few_shot = """示例：
用户：我的订单ORD20240101001到哪了？ -> {"intent": "precise_query"}
用户：订单被退回了怎么办？ -> {"intent": "knowledge"}
用户：你们退货运费谁出？ -> {"intent": "knowledge"}
用户：你好，在吗？ -> {"intent": "other"}"""
    hist_text = "\n".join(f"{t['role']}: {t['content']}" for t in history[-4:])
    prompt = f"{few_shot}\n历史对话：\n{hist_text}\n当前问题：{question}"
    try:
        resp = await llm_ainvoke(
            [SystemMessage(content=_INTENT_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            stage="intent",
        )
        text = resp.content.strip().strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        data = json.loads(text)
        intent = data.get("intent", "knowledge")
        if intent not in ("precise_query", "knowledge", "other"):
            intent = "knowledge"
        return intent
    except Exception:
        # LLM 不可用时兜底为 knowledge，避免请求失败
        return "knowledge"


async def classify(question: str, history: list[dict] | None = None) -> dict:
    """返回 {"intent": "precise_query"|"knowledge"|"other", "entities": {...}, "method": "..."}"""
    history = history or []
    entities = detect_entities(question)

    # 第零层：政策/规则类问法优先（即使带订单号也走 FAQ）
    # 例："订单 ORD123 的退货运费谁出？" → knowledge，避免用订单状态答非所问
    if has_knowledge_marker(question):
        return {"intent": "knowledge", "entities": entities, "method": "policy_keyword"}

    # 第一层：正则快路径（订单号/物流单号格式命中）
    if entities["order_ids"] or entities["tracking_nos"]:
        return {"intent": "precise_query", "entities": entities, "method": "regex"}

    # 第二层：关键词信号 → LLM few-shot 确认
    if has_precise_signal(question, entities):
        intent = await _classify_with_llm(question, history)
        # 实体兜底：判定为精确查询但既无订单号也无物流单号时，精确查询根本无法执行
        # （数据库按单号检索）。这类问题通常问的是"发货时效/查询方式"等政策，
        # 应降级走 FAQ 检索，而不是把"我的订单什么时候发货？"直接判成 400。
        if intent == "precise_query" and not (entities["order_ids"] or entities["tracking_nos"]):
            return {"intent": "knowledge", "entities": entities, "method": "llm_no_entity"}
        return {"intent": intent, "entities": entities, "method": "llm"}

    # 无信号：默认知识类
    return {"intent": "knowledge", "entities": entities, "method": "keyword"}
