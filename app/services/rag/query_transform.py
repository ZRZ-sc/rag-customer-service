"""查询改写（需求文档流程2）：Multi-Query 扩展 + HyDE 假设性回答。"""
from langchain_core.messages import HumanMessage

from app.services.llm import get_llm

_MULTI_QUERY_PROMPT = """你是信息检索查询改写助手。请将用户问题改写成{n}个不同角度的检索查询，用于提高知识库召回率。
要求：
1. 保留核心实体与关键词；
2. 从口语表达、正式表达、同义词等不同角度改写；
3. 每行只输出一个查询，不要编号，不要任何解释。
{history_text}用户问题：{question}"""

_HYDE_PROMPT = """你是电商客服专家。请不要直接回答用户，而是先写一段假设性的标准回答（80~150字），用于增强信息检索的召回效果。
要求：覆盖问题涉及的核心概念、政策要点与常见处理方式。

用户问题：{question}"""


async def multi_query(question: str, history: list[dict] | None = None, n: int = 3) -> list[str]:
    """扩展为 n 个查询变体（含原问题）。失败时退化为 [原问题]。"""
    history = history or []
    hist_text = ""
    if history:
        hist_text = "历史对话：\n" + "\n".join(
            f"{t['role']}: {t['content']}" for t in history[-4:]
        ) + "\n"
    try:
        resp = await get_llm().ainvoke(
            [HumanMessage(content=_MULTI_QUERY_PROMPT.format(n=n, history_text=hist_text, question=question))]
        )
        lines = [
            ln.strip().lstrip("-•0123456789.、 ").strip()
            for ln in resp.content.splitlines()
            if ln.strip()
        ]
        variants = [ln for ln in lines if ln][: n - 1]
    except Exception:
        variants = []
    out = [question]
    for v in variants:
        if v != question and v not in out:
            out.append(v)
    return out[:n]


async def hyde(question: str) -> str:
    """生成假设性答案用于检索。失败时返回空串。"""
    try:
        resp = await get_llm().ainvoke(
            [HumanMessage(content=_HYDE_PROMPT.format(question=question))]
        )
        return resp.content.strip()
    except Exception:
        return ""
